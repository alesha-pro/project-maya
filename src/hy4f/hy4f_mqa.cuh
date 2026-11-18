// src/hy4f/hy4f_mqa.cuh - maya-hy4f: the batched absorbed-MLA attention on tensor cores (WMMA fp16, fp32 accumulate).
// MLA absorbed is MQA: every head reads the same 576-wide latent row, so one block per token takes ALL the shard's heads
// and each gathered KV tile is read once for H heads (kb_attn2 read it H times).  Two passes over the rows: the row
// max / sum, then P = exp(S - m) / l and O += P V - no rescaling of the accumulator fragments.
//   SWA (cells null): the window over the ring (positions < pos0) and the chunk buffer (>= pos0); full softmax with sink.
//   Token t of the launch is at position pos0 + t0 + t (an attention tile of the chunk; q / qabs / cells / o / part
//   rows are the tile's); the chunk buffer's rows stay relative to pos0.
//   DSA: cells [T][n_cap]; n_shards > 1: this shard's cells (c % S == shard), partial (o unnormalized, M, L) to part.
#pragma once

#include "hy4f_kern.cuh"

#include <mma.h>

namespace strata::hy4f {
namespace kern {

constexpr int MQA_LD = 584;   // halfs per smem row (576 + 8: 16-byte rows, no 2-way bank pattern on the k loop)
constexpr int MQA_TR = 32;    // KV rows per tile

template <int H> constexpr size_t mqa_smem() {
    return (size_t) H * MQA_LD * 2 + (size_t) MQA_TR * MQA_LD * 2 + 8192 + (size_t) H * 40 * 2 + (size_t) H * 4 * 3 +
           MQA_TR * sizeof(void*) + 2048 * 4 + 64;
}

// Expand each Q8 row once for all overlapping query windows in the chunk.
__global__ void kb_swa_expand(const uint8_t* ring, const uint8_t* chunk,
                              const int* pos0p, int win, int first, __half* expanded) {
    const int r = blockIdx.x, group = threadIdx.x, pos = first + r, pos0 = *pos0p;
    if (group >= KV_ROW / 8) return;
    const uint8_t* row = pos < pos0 ? ring + (size_t)(pos % (win + RING_PAD)) * KV_RB
                                    : chunk + (size_t)(pos - pos0) * KV_RB;
    const int col = group * 8;
    const float scale = __half2float(reinterpret_cast<const __half*>(row + KV_ROW)[col / 64]);
    int2 packed = *reinterpret_cast<const int2*>(row + col);
    const auto* bytes = reinterpret_cast<const int8_t*>(&packed);
    uint4 out;
    auto* halves = reinterpret_cast<__half2*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) halves[i] = __floats2half2_rn(bytes[2*i] * scale, bytes[2*i+1] * scale);
    reinterpret_cast<uint4*>(expanded + (size_t)r * KV_ROW)[group] = out;
}

// With four token positions per selected pool and four interleaved KV shards,
// every complete pool contributes exactly one position to each GPU. The only
// incomplete pool is the causal tail. Preserve the selector's original order.
__device__ __forceinline__ int grouped_rows(int pos, int top_k, int groups, int cap, int shard) {
    const int rows = pos < top_k ? pos + 1 : min(cap, 4 * min(pos / 4, groups) + (pos % 4) + 1);
    return (rows + 3 - shard) / 4;
}

__device__ __forceinline__ int grouped_kv_row(int row, int pos, int top_k, int groups,
                                              const int* cells, const uint16_t* wire) {
    if (pos < top_k) return row;
    if (row >= min(groups, pos / 4)) return pos / 4;
    return wire ? (int)wire[row] : cells[4 * row] / 4;
}

template <int H>
__global__ void __launch_bounds__(256) kb_mqa_attn(const float* __restrict__ qabs, const float* __restrict__ q, int ldq,
                                                   int qs, int nope, float theta, const uint8_t* __restrict__ kv,
                                                   const uint8_t* __restrict__ chunk, const int* __restrict__ cells,
                                                   int n_cap, const int* pos0p, int t0, int win, int n_shards, int shard,
                                                   const float* __restrict__ sinks, float scale, float* __restrict__ o,
                                                   float* __restrict__ part, int W, const __half* expanded = nullptr, int first = 0, bool cache_scores = false,
                                                   int top_k = 0, int groups = 0, const uint16_t* wire = nullptr) {
    const int pos0 = *pos0p;
    using namespace nvcuda;
    constexpr int KVL = 512, ROPE = 64, D = 576, LD = MQA_LD, TR = MQA_TR;
    constexpr int NST = H / 8;          // S tiles (16 heads x 16 rows): (H/16) x 2
    constexpr int KS = 8 / NST;         // k split of S over the warps
    constexpr int KP = D / KS;          // k per part (multiple of 16)
    constexpr int OPW = H / 4;          // O tiles per warp: (H/16) x 32 / 8
    extern __shared__ __align__(128) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + (size_t) H * LD;
    float* Sp = reinterpret_cast<float*>(Ks + (size_t) TR * LD);      // [KS][H][32]
    __half* Ps = reinterpret_cast<__half*>(Sp + 2048);                // [H][40]
    float* sm_m = reinterpret_cast<float*>(Ps + H * 40);
    float* sm_l = sm_m + H;
    float* sm_i = sm_l + H;
    const uint8_t** rowp = reinterpret_cast<const uint8_t**>(sm_i + H);
    int* cl = reinterpret_cast<int*>(rowp + TR);
    __shared__ int wtot[8], s_n;

    const int t = blockIdx.x, pos = pos0 + t0 + t, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    // Q tile: scale * [q_abs | RoPE(q_pe)] as fp16
    for (int i = tid; i < H * D; i += 256) {
        const int h = i / D, c = i % D;
        float v;
        if (c < KVL) v = qabs[((size_t) t * H + h) * KVL + c];
        else {
            const int j = c - KVL, jj = j >> 1;
            const float* qp = q + (size_t) t * ldq + (size_t) h * qs + nope + 2 * jj;
            float sn, cs;
            sincosf((float) pos * powf(theta, -2.0f * jj / ROPE), &sn, &cs);
            v = (j & 1) ? qp[0] * sn + qp[1] * cs : qp[0] * cs - qp[1] * sn;
        }
        Qs[(size_t) h * LD + c] = __float2half_rn(v * scale);
    }
    // the rows: SWA window / this shard's cells (compacted in order)
    int n, lo = 0;
    if (top_k > 0) {
        n = grouped_rows(pos, top_k, groups, n_cap, shard);
    } else if (!cells) {
        lo = max(0, pos - win + 1);
        n = pos - lo + 1;
    } else {
        const int* cr = cells + (size_t) t * n_cap;
        if (tid == 0) s_n = 0;
        __syncthreads();
        for (int b0 = 0; b0 < n_cap; b0 += 256) {
            const int i = b0 + tid;
            const int c = i < n_cap ? cr[i] : -1;
            const bool own = c >= 0 && (n_shards == 1 || c % n_shards == shard);
            const unsigned m = __ballot_sync(0xffffffffu, own);
            if (lane == 0) wtot[w] = __popc(m);
            __syncthreads();
            int base = s_n;
            for (int k = 0; k < w; ++k) base += wtot[k];
            if (own) cl[base + __popc(m & ((1u << lane) - 1))] = n_shards == 1 ? c : c / n_shards;
            __syncthreads();
            if (tid == 0) {
                int s = s_n;
                for (int k = 0; k < 8; ++k) s += wtot[k];
                s_n = s;
            }
            __syncthreads();
        }
        n = s_n;
    }
    if (tid < H) sm_m[tid] = -INFINITY, sm_l[tid] = 0.0f;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[OPW];
#pragma unroll
    for (int i = 0; i < OPW; ++i) wmma::fill_fragment(acc[i], 0.0f);

    // Each CTA can reuse its own partial-output row until the final stores.
    const bool saved = cache_scores && part && n <= W / TR * TR;
    for (int pass = 0; pass < 2; ++pass) {
        for (int r0 = 0; r0 < n; r0 += TR) {
            __syncthreads();   // (the previous tile's readers are done)
            if (expanded && cells) {
                if (tid < TR) {
                    const int r = r0+tid;
                    int row=0;
                    if (r<n) row=top_k ? grouped_kv_row(r,pos,top_k,groups,cells+(size_t)t*n_cap,
                        wire ? wire+(size_t)t*((groups+7)/8*8) : nullptr) : cl[r];
                    rowp[tid]=reinterpret_cast<const uint8_t*>(expanded+(size_t)row*D);
                }
                __syncthreads();
                for(int i=tid;i<TR*(D/8);i+=256) {
                    int row=i/(D/8), col=i%(D/8)*8;
                    __half* dst=Ks+(size_t)row*LD+col;
                    const __half* src=reinterpret_cast<const __half*>(rowp[row])+col;
#if __CUDA_ARCH__ >= 800
                    unsigned address=(unsigned)__cvta_generic_to_shared(dst);
                    int size=r0+row<n ? 16:0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(src),"r"(size));
#else
                    *reinterpret_cast<uint4*>(dst)=r0+row<n ? *reinterpret_cast<const uint4*>(src):make_uint4(0,0,0,0);
#endif
                }
#if __CUDA_ARCH__ >= 800
                asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
            } else if (expanded) {
                for (int i = tid; i < TR * (D / 8); i += 256) {
                    const int row = i / (D / 8), group = i % (D / 8);
                    const int r = r0 + row;
                    uint4 data = make_uint4(0, 0, 0, 0);
                    if (r < n)
                        data = reinterpret_cast<const uint4*>(expanded + (size_t)(lo + r - first) * D)[group];
                    *reinterpret_cast<uint4*>(Ks + (size_t)row * LD + 8 * group) = data;
                }
            } else {
            if (tid < TR) {
                const int s = r0 + tid;
                const uint8_t* p = nullptr;
                if (s < n) {
                    if (!cells) {
                        const int pp = lo + s;
                        p = (chunk && pp >= pos0) ? chunk + (size_t) (pp - pos0) * KV_RB : kv + (size_t) (pp % (win + RING_PAD)) * KV_RB;
                    } else {
                        const int row = top_k > 0 ? grouped_kv_row(s, pos, top_k, groups,
                            cells + (size_t)t * n_cap, wire ? wire + (size_t)t * ((groups+7)/8*8) : nullptr) : cl[s];
                        p = kv + (size_t)row * KV_RB;
                    }
                }
                rowp[tid] = p;
            }
            __syncthreads();
            for (int i = tid; i < TR * (D / 8); i += 256) {
                const int r = i / (D / 8), c = (i % (D / 8)) * 8;
                const uint8_t* p = rowp[r];
                uint4 out = make_uint4(0, 0, 0, 0);
                if (p) {
                    const int2 v = *reinterpret_cast<const int2*>(p + c);
                    const float sc = __half2float(reinterpret_cast<const __half*>(p + KV_ROW)[c >> 6]);
                    const int8_t* b = reinterpret_cast<const int8_t*>(&v);
                    __half2* o2 = reinterpret_cast<__half2*>(&out);
#pragma unroll
                    for (int k = 0; k < 4; ++k) o2[k] = __floats2half2_rn(b[2 * k] * sc, b[2 * k + 1] * sc);
                }
                *reinterpret_cast<uint4*>(Ks + (size_t) r * LD + c) = out;
            }
            }
            __syncthreads();
            if (!saved || pass == 0) {   // S partials
                const int st = w % NST, kp = w / NST, hi = st >> 1, ni = st & 1;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
                wmma::fill_fragment(fs, 0.0f);
#pragma unroll 4
                for (int k = kp * KP; k < (kp + 1) * KP; k += 16) {
                    wmma::load_matrix_sync(fa, Qs + (size_t) hi * 16 * LD + k, LD);
                    wmma::load_matrix_sync(fb, Ks + (size_t) ni * 16 * LD + k, LD);
                    wmma::mma_sync(fs, fa, fb, fs);
                }
                wmma::store_matrix_sync(Sp + (size_t) kp * H * 32 + hi * 16 * 32 + ni * 16, fs, 32, wmma::mem_row_major);
            }
            __syncthreads();
            for (int h = w; h < H; h += 8) {
                float s = 0.0f;
                const size_t si = ((size_t)t * H + h) * W + r0 + lane;
                if (saved && pass == 1) {
                    s = part[si];
                } else {
#pragma unroll
                    for (int k = 0; k < KS; ++k) s += Sp[(size_t) k * H * 32 + h * 32 + lane];
                    if (saved) part[si] = s;
                }
                const bool valid = r0 + lane < n;
                if (pass == 0) {
                    const float mx = warp_max(valid ? s : -INFINITY);
                    const float mo = sm_m[h], mn = fmaxf(mo, mx);
                    const float e = warp_sum(valid ? __expf(s - mn) : 0.0f);
                    if (lane == 0) {
                        sm_l[h] = sm_l[h] * (mo == -INFINITY ? 0.0f : __expf(mo - mn)) + e;
                        sm_m[h] = mn;
                    }
                } else {
                    const float m = sm_m[h];
                    const float p = valid && m != -INFINITY ? __expf(s - m) * sm_i[h] : 0.0f;
                    Ps[h * 40 + lane] = __float2half_rn(p);
                }
            }
            if (pass == 1) {
                __syncthreads();
#pragma unroll
                for (int i = 0; i < OPW; ++i) {
                    const int idx = w * OPW + i, hi = idx >> 5, nj = idx & 31;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
#pragma unroll
                    for (int kk = 0; kk < TR; kk += 16) {
                        wmma::load_matrix_sync(fa, Ps + hi * 16 * 40 + kk, 40);
                        wmma::load_matrix_sync(fb, Ks + (size_t) kk * LD + nj * 16, LD);
                        wmma::mma_sync(acc[i], fa, fb, acc[i]);
                    }
                }
            }
        }
        if (pass == 0) {
            __syncthreads();
            if (tid < H) {
                const bool partial = n_shards > 1 && cells;
                if (partial) {
                    sm_i[tid] = 1.0f;   // unnormalized; M, L exported
                } else {
                    const float m = sm_m[tid], sk = sinks[tid], mf = fmaxf(m, sk);
                    const float l = sm_l[tid] * (m == -INFINITY ? 0.0f : __expf(m - mf)) + __expf(sk - mf);
                    sm_m[tid] = mf;
                    sm_i[tid] = 1.0f / l;
                }
            }
            __syncthreads();
        }
    }
    const bool partial = n_shards > 1 && cells;
#pragma unroll
    for (int i = 0; i < OPW; ++i) {
        const int idx = w * OPW + i, hi = idx >> 5, nj = idx & 31;
        if (partial)
            wmma::store_matrix_sync(part + ((size_t) t * H + hi * 16) * W + nj * 16, acc[i], W, wmma::mem_row_major);
        else
            wmma::store_matrix_sync(o + ((size_t) t * H + hi * 16) * KVL + nj * 16, acc[i], KVL, wmma::mem_row_major);
    }
    if (partial && tid < H) {
        part[((size_t) t * H + tid) * W + KVL] = sm_m[tid];
        part[((size_t) t * H + tid) * W + KVL + 1] = sm_l[tid];
    }
}


// ---- the decode form (a few tokens): split-K over the rows, so a token's work spreads over gridDim.x blocks
// (flash-decoding).  grid (nsplit, T): block (sp, t) takes rows [sp*per, (sp+1)*per) of token t's row list (the SWA
// window / this shard's cells), all H heads, and writes the unnormalized partial (m, l, acc) of each head; kd_mqa_merge
// folds the splits (with the sink, or into the combine's partial).  Same tile math as kb_mqa_attn.
template <int H> constexpr size_t mqd_smem() { return mqa_smem<H>(); }

template <int H, bool DenseDsa = false, int ValueGroups = 1, bool Replica = false>
__global__ void __launch_bounds__(256) kd_mqa_dec(const float* __restrict__ qabs, const float* __restrict__ q, int ldq,
                                                  int qs, int nope, float theta, const uint8_t* __restrict__ kv,
                                                  const int* __restrict__ cells, int n_cap, const int* pos0p, int win,
                                                  int n_shards, int shard, float scale, float* __restrict__ pm,
                                                  float* __restrict__ pl, float* __restrict__ pacc, int top_k = 0, int groups = 0, const __half* packed_q = nullptr, bool local_queries = false, int mirror_base = 0, int mirror_stripe = 0) {
    using namespace nvcuda;
    constexpr int KVL = 512, ROPE = 64, D = 576, LD = MQA_LD, TR = MQA_TR;
    constexpr int NST = H / 8, KS = Replica ? 2 : 8 / NST, KP = D / KS, OPW = H / (4 * ValueGroups);
    extern __shared__ __align__(128) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + (size_t) H * LD;
    float* Sp = reinterpret_cast<float*>(Ks + (size_t) TR * LD);
    __half* Ps = reinterpret_cast<__half*>(Sp + 2048);
    float* sm_m = reinterpret_cast<float*>(Ps + H * 40);
    float* sm_l = sm_m + H;
    float* sm_i = sm_l + H;
    const uint8_t** rowp = reinterpret_cast<const uint8_t**>(sm_i + H);
    int* cl = reinterpret_cast<int*>(rowp + TR);
    __shared__ int wtot[8], s_n;
    const int vg=blockIdx.x % ValueGroups, sp = blockIdx.x / ValueGroups, nsp = gridDim.x / ValueGroups, t = blockIdx.y, pos = *pos0p + t;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int logical_shard = Replica ? (int)blockIdx.z : shard;
    const int head_offset = Replica && !local_queries ? shard * 8 : 0;
    const int input_heads = Replica ? (local_queries ? 8 : 32) : H;
    int n, lo = 0;
    if constexpr (DenseDsa) {
        n = (pos + n_shards - logical_shard) / n_shards;
    } else if (top_k > 0) {
        n = grouped_rows(pos, top_k, groups, n_cap, logical_shard);
    } else if (!cells) {
        lo = max(0, pos - win + 1);
        n = pos - lo + 1;
    } else {
        const int* cr = cells + (size_t) t * n_cap;
        if (tid == 0) s_n = 0;
        __syncthreads();
        for (int b0 = 0; b0 < n_cap; b0 += 256) {
            const int i = b0 + tid;
            const int c = i < n_cap ? cr[i] : -1;
            const bool own = c >= 0 && (n_shards == 1 || c % n_shards == shard);
            const unsigned m = __ballot_sync(0xffffffffu, own);
            if (lane == 0) wtot[w] = __popc(m);
            __syncthreads();
            int base = s_n;
            for (int k = 0; k < w; ++k) base += wtot[k];
            if (own) cl[base + __popc(m & ((1u << lane) - 1))] = n_shards == 1 ? c : c / n_shards;
            __syncthreads();
            if (tid == 0) {
                int s2 = s_n;
                for (int k = 0; k < 8; ++k) s2 += wtot[k];
                s_n = s2;
            }
            __syncthreads();
        }
        n = s_n;
    }
    const int per = ((n + nsp - 1) / nsp + TR - 1) / TR * TR, rb = min(n, sp * per), re = min(n, rb + per);
    const size_t pb = (((Replica ? (size_t)blockIdx.z * gridDim.y : 0) + t) * nsp + sp) * H;
    if (rb == re) {
        if (vg == 0 && tid < H) pm[pb + tid] = -INFINITY, pl[pb + tid] = 0.0f;
        return;  // merge only reads accumulator slices belonging to active splits
    }
    if (packed_q) {
        for (int i=tid;i<H*(D/8);i+=256) {
            const int h=i/(D/8),col=i%(D/8)*8;
            auto* dst=Qs+(size_t)h*LD+col;
            auto* src=packed_q+((size_t)t*input_heads+head_offset+(Replica ? h%8 : h))*D+col;
#if __CUDA_ARCH__ >= 800
            unsigned da=(unsigned)__cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(da),"l"(src));
#else
            *reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(src);
#endif
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    } else for (int i = tid; i < H * D; i += 256) {
        const int h = i / D, c = i % D;
        float v;
        if (c < KVL) v = qabs[((size_t)t*input_heads+head_offset+(Replica ? h%8 : h))*KVL + c];
        else {
            const int j = c - KVL, jj = j >> 1;
            const float* qp = q + (size_t) t * ldq + (size_t)(head_offset+(Replica ? h%8 : h)) * qs + nope + 2 * jj;
            float sn, cs;
            sincosf((float) pos * powf(theta, -2.0f * jj / ROPE), &sn, &cs);
            v = (j & 1) ? qp[0] * sn + qp[1] * cs : qp[0] * cs - qp[1] * sn;
        }
        Qs[(size_t) h * LD + c] = __float2half_rn(v * scale);
    }
    if (tid < H) sm_m[tid] = -INFINITY, sm_l[tid] = 0.0f, sm_i[tid] = 1.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[OPW];
#pragma unroll
    for (int i = 0; i < OPW; ++i) wmma::fill_fragment(acc[i], 0.0f);
    // A one-tile split has its final softmax maximum after its sole QK operation.
    // Keep scores and the same dequantized KV tile for PV instead of repeating QK.
    const bool one_tile = re - rb <= TR;
    for (int pass = 0; pass < (one_tile ? 1 : 2); ++pass) {
        for (int r0 = rb; r0 < re; r0 += TR) {
            __syncthreads();
            if (tid < TR) {
                const int s2 = r0 + tid;
                const uint8_t* p = nullptr;
                if (s2 < re) {
                    if constexpr (Replica && !DenseDsa) {
                        const int row = grouped_kv_row(s2,pos,top_k,groups,cells+(size_t)t*n_cap,nullptr);
                        const int base = replica_row_base(logical_shard,shard,mirror_base,mirror_stripe);
                        p = kv + (size_t)(base+row)*KV_RB;
                    } else if constexpr (Replica) p = kv + ((size_t)s2*n_shards+logical_shard)*KV_RB;
                    else if constexpr (DenseDsa) p = kv + (size_t) s2 * KV_RB;
                    else if (top_k > 0) p = kv + (size_t)grouped_kv_row(s2, pos, top_k, groups, cells+(size_t)t*n_cap, nullptr) * KV_RB;
                    else p = !cells ? kv + (size_t) ((lo + s2) % (win + RING_PAD)) * KV_RB : kv + (size_t) cl[s2] * KV_RB;
                }
                rowp[tid] = p;
            }
            __syncthreads();
            for (int i = tid; i < TR * (D / 8); i += 256) {
                const int r = i / (D / 8), c = (i % (D / 8)) * 8;
                const uint8_t* p = rowp[r];
                uint4 out = make_uint4(0, 0, 0, 0);
                if (p) {
                    const int2 v = *reinterpret_cast<const int2*>(p + c);
                    const float sc = __half2float(reinterpret_cast<const __half*>(p + KV_ROW)[c >> 6]);
                    const int8_t* bq = reinterpret_cast<const int8_t*>(&v);
                    __half2* o2 = reinterpret_cast<__half2*>(&out);
#pragma unroll
                    for (int k = 0; k < 4; ++k) o2[k] = __floats2half2_rn(bq[2 * k] * sc, bq[2 * k + 1] * sc);
                }
                *reinterpret_cast<uint4*>(Ks + (size_t) r * LD + c) = out;
            }
            __syncthreads();
            if (!Replica || w < NST * KS) {
                const int st = w % NST, kp = w / NST, hi = st >> 1, ni = st & 1;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
                wmma::fill_fragment(fs, 0.0f);
#pragma unroll 4
                for (int k = kp * KP; k < (kp + 1) * KP; k += 16) {
                    wmma::load_matrix_sync(fa, Qs + (size_t) hi * 16 * LD + k, LD);
                    wmma::load_matrix_sync(fb, Ks + (size_t) ni * 16 * LD + k, LD);
                    wmma::mma_sync(fs, fa, fb, fs);
                }
                wmma::store_matrix_sync(Sp + (size_t) kp * H * 32 + hi * 16 * 32 + ni * 16, fs, 32, wmma::mem_row_major);
            }
            __syncthreads();
            for (int h = w; h < H; h += 8) {
                float s2 = 0.0f;
#pragma unroll
                for (int k = 0; k < KS; ++k) s2 += Sp[(size_t) k * H * 32 + h * 32 + lane];
                const bool valid = r0 + lane < re;
                if (pass == 0) {
                    const float mx = warp_max(valid ? s2 : -INFINITY);
                    const float mo = sm_m[h], mn = fmaxf(mo, mx);
                    const float weight = valid ? __expf(s2 - mn) : 0.0f;
                    const float e = warp_sum(weight);
                    if (one_tile) Ps[h * 40 + lane] = __float2half_rn(mn != -INFINITY ? weight : 0.0f);
                    if (lane == 0) {
                        sm_l[h] = sm_l[h] * (mo == -INFINITY ? 0.0f : __expf(mo - mn)) + e;
                        sm_m[h] = mn;
                    }
                } else {
                    const float m = sm_m[h];
                    Ps[h * 40 + lane] = __float2half_rn(valid && m != -INFINITY ? __expf(s2 - m) : 0.0f);
                }
            }
            if (pass == 1 || one_tile) {
                __syncthreads();
#pragma unroll
                for (int i = 0; i < OPW; ++i) {
                    const int idx = (vg * 8 + w) * OPW + i, hi = idx >> 5, nj = idx & 31;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
#pragma unroll
                    for (int kk = 0; kk < TR; kk += 16) {
                        wmma::load_matrix_sync(fa, Ps + hi * 16 * 40 + kk, 40);
                        wmma::load_matrix_sync(fb, Ks + (size_t) kk * LD + nj * 16, LD);
                        wmma::mma_sync(acc[i], fa, fb, acc[i]);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < OPW; ++i) {
        const int idx = (vg * 8 + w) * OPW + i, hi = idx >> 5, nj = idx & 31;
        wmma::store_matrix_sync(pacc + (pb + hi * 16) * KVL + nj * 16, acc[i], KVL, wmma::mem_row_major);
    }
    if (vg == 0 && tid < H) pm[pb + tid] = sm_m[tid], pl[pb + tid] = sm_l[tid];
}
// Each virtual shard uses exactly the K splits and row order of the
// context-parallel graph. Fold its partials before combining the four ranks.
__global__ void kd_replica_merge(const float* pm,const float* pl,const float* acc,
                                 int splits,int tokens,const float* sinks,float* out) {
    const int t=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ float factor[4][32], rank_max[4], rank_sum[4], scaling[4], denominator;
    __shared__ int count[4];
    if (warp<4) {
        const size_t row=(((size_t)warp*tokens+t)*splits+lane)*16+h;
        const float m=lane<splits ? pm[row] : -INFINITY;
        const float l=lane<splits ? pl[row] : 0.f;
        const int active=32-__clz(__ballot_sync(~0u,m!=-INFINITY));
        const float mx=warp_max(m);
        const float f=m==-INFINITY ? 0.f : __expf(m-mx);
        factor[warp][lane]=f;
        float sum=0.f;
        for(int i=0;i<active;++i)
            sum+=__shfl_sync(~0u,f,i)*__shfl_sync(~0u,l,i);
        if(lane==0) { rank_max[warp]=mx;rank_sum[warp]=sum;count[warp]=active; }
    }
    __syncthreads();
    if(tid==0) {
        float mx=sinks[h];
        for(int j=0;j<4;++j) mx=fmaxf(mx,rank_max[j]);
        float den=__expf(sinks[h]-mx);
        for(int j=0;j<4;++j) {
            scaling[j]=rank_max[j]==-INFINITY ? 0.f : __expf(rank_max[j]-mx);
            den+=scaling[j]*rank_sum[j];
        }
        denominator=den;
    }
    __syncthreads();
    for(int c=tid;c<512;c+=blockDim.x) {
        float combined=0.f;
        for(int j=0;j<4;++j) {
            float own=0.f;
            for(int sp=0;sp<count[j];++sp) {
                const size_t row=(((size_t)j*tokens+t)*splits+sp)*16+h;
                own+=factor[j][sp]*acc[row*512+c];
            }
            combined+=scaling[j]*own;
        }
        out[((size_t)t*8+h)*512+c]=combined/denominator;
    }
}
// fold the splits of (t, h): full (sinks != null): o[t][h] = acc / (L + e^(sink - M)); else the combine's partial
// part[t][h][W] = (acc, M, L) relative to the splits' common max.  grid (H, T), 256 threads.
__global__ void kd_mqa_merge(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                             int nsp, int H, const float* __restrict__ sinks, float* __restrict__ o, float* __restrict__ part,
                             int W) {
    constexpr int KVL = 512;
    const int h = blockIdx.x, t = blockIdx.y;
    float M = sinks ? sinks[h] : -INFINITY;
    int active = nsp;
    while (active && pm[((size_t) t * nsp + active - 1) * H + h] == -INFINITY) --active;
    for (int sp = 0; sp < active; ++sp) M = fmaxf(M, pm[((size_t) t * nsp + sp) * H + h]);
    float L = sinks && M != -INFINITY ? __expf(sinks[h] - M) : 0.0f, f[32];
    for (int sp = 0; sp < active; ++sp) {
        const float m = pm[((size_t) t * nsp + sp) * H + h];
        f[sp] = m == -INFINITY ? 0.0f : __expf(m - M);
        L += f[sp] * pl[((size_t) t * nsp + sp) * H + h];
    }
    for (int c = threadIdx.x; c < KVL; c += blockDim.x) {
        float a = 0.0f;
        for (int sp = 0; sp < active; ++sp) a += f[sp] * pacc[(((size_t) t * nsp + sp) * H + h) * KVL + c];
        if (sinks) o[((size_t) t * H + h) * KVL + c] = a / L;
        else part[((size_t) t * H + h) * W + c] = a;
    }
    if (!sinks && threadIdx.x == 0) {
        part[((size_t) t * H + h) * W + KVL] = M;
        part[((size_t) t * H + h) * W + KVL + 1] = L;
    }
}


// Compute each split's exponential once per head. The denominator and the
// value sum preserve consecutive split order; only the maximum is parallel.
template<bool Send, bool ByDestination = false>
__global__ void kd_mqa_fold(const float* pm, const float* pl, const float* acc, int nsp, int H,
                            const float* sinks, float* output, float* part, int W, PeersD peers,
                            int me, int ranks, const int* counter) {
    const int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x;
    __shared__ float factor[32], maxval, denominator;
    __shared__ int active;
    if(tid<32) {
        size_t i=((size_t)t*nsp+tid)*H+h;
        const float m=tid<nsp?pm[i]:-INFINITY;
        const float l=tid<nsp?pl[i]:0.f;
        const int live=32-__clz(__ballot_sync(~0u,m!=-INFINITY));
        const float mx=fmaxf(warp_max(m),sinks?sinks[h]:-INFINITY);
        const float f=m==-INFINITY?0.f:__expf(m-mx);
        factor[tid]=f;
        float den=sinks && mx!=-INFINITY?__expf(sinks[h]-mx):0.f;
        for(int j=0;j<live;++j) {
            const float fj=__shfl_sync(~0u,f,j), lj=__shfl_sync(~0u,l,j);
            den+=fj*lj;
        }
        if(tid==0) { active=live; maxval=mx; denominator=den; }
    }
    __syncthreads();
    int dest=me, hl=0, step=0;
    float* row=part ? part+((size_t)t*H+h)*W : nullptr;
    if constexpr (ByDestination) {
        const int heads = H / ranks;
        row = part + (((size_t)(h / heads) * gridDim.y + t) * heads + h % heads) * W;
    }
    if constexpr (Send) {
        const int local_heads=H/ranks;
        dest=h/local_heads; hl=h%local_heads;
        step=*counter+1;
        if(dest!=me) row=peers.buf[dest][step&1]+(((size_t)me*gridDim.y+t)*local_heads+hl)*W;
    }
    for(int c=tid;c<512;c+=blockDim.x) {
        float x=0.f;
        for(int j=0;j<active;++j) x+=factor[j]*acc[(((size_t)t*nsp+j)*H+h)*512+c];
        if(sinks) output[((size_t)t*H+h)*512+c]=x/denominator;
        else row[c]=x;
    }
    if(!sinks && tid==0) { row[512]=maxval; row[513]=denominator; }
    if constexpr (Send) {
        if(dest!=me) {
            __threadfence_system();
            __syncthreads();
            if(tid==0) {
                int* signal=peers.flag[dest]+(t*(H/ranks)+hl)*ranks+me;
                asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(signal), "r"(step) : "memory");
            }
        }
    }
}

// ---- the k-pool indexer scores on tensor cores: score[tt][g] = sum_h relu(q[t][h] . k[g]) * w[t][h] for the score
// tile's tokens tt = 0 .. tt_n-1 (token t0 + tt) and the pooled groups g < pos / kpool.  Block (gs, tb): tokens
// tb*8 .. tb*8+7 (warp = token: its 16 heads are one 16-row WMMA tile, loaded once), groups [gs*per, (gs+1)*per) in
// tiles of 16 (fp16 keys straight from global / L2).  Scores of tokens with pos + 1 <= top_k are not needed.
__global__ void __launch_bounds__(256) kb_score_tc(const float* __restrict__ iq, int ldq, const __half* __restrict__ pooled,
                                                   const float* __restrict__ w, int ldw, int kpool, int top_k,
                                                   const int* pos0p, int t0, int tt_n, int n_pool_max,
                                                   float* __restrict__ score, int ld_score) {
    using namespace nvcuda;
    constexpr int KD = 128, NH = 16, QLD = KD + 8;
    __shared__ __align__(32) __half Qs[8][NH][QLD];
    __shared__ __align__(32) float Ss[8][16][20];
    __shared__ float Ws[8][NH];
    const int tb = blockIdx.y, gs = blockIdx.x, ngs = gridDim.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos0 = *pos0p;
    for (int i = threadIdx.x; i < 8 * NH * KD; i += blockDim.x) {
        const int tk = i / (NH * KD), rem = i % (NH * KD), h = rem / KD, d = rem % KD, tt = tb * 8 + tk;
        Qs[tk][h][d] = __float2half_rn(tt < tt_n ? iq[(size_t) (t0 + tt) * ldq + h * KD + d] : 0.0f);
    }
    if (threadIdx.x < 8 * NH) {
        const int tk = threadIdx.x / NH, h = threadIdx.x % NH, tt = tb * 8 + tk;
        Ws[tk][h] = tt < tt_n ? w[(size_t) (t0 + tt) * ldw + h] : 0.0f;
    }
    __syncthreads();
    if (tt_n <= 4) {   // few tokens (decode): every warp a group sub-range for ALL the tokens, each key tile read once
        const int per_b = ((n_pool_max + ngs - 1) / ngs + 15) / 16 * 16, per = (per_b / 8 + 15) / 16 * 16;
        const int g0 = gs * per_b + warp * per;
        int npool_max_t = 0;
        for (int t = 0; t < tt_n; ++t) {
            const int p = pos0 + t0 + t;
            if (p + 1 > top_k) npool_max_t = max(npool_max_t, p / kpool);
        }
        const int g1 = min(min(npool_max_t, gs * per_b + per_b), g0 + per);
        for (int g = g0; g < g1; g += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) wmma::fill_fragment(acc[t], 0.0f);
#pragma unroll
            for (int k = 0; k < KD / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, pooled + (size_t) g * KD + k * 16, KD);
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (t < tt_n) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa1;
                        wmma::load_matrix_sync(fa1, &Qs[t][0][k * 16], QLD);
                        wmma::mma_sync(acc[t], fa1, fb, acc[t]);
                    }
            }
#pragma unroll
            for (int t = 0; t < 4; ++t)
                if (t < tt_n) {
                    const int p = pos0 + t0 + t, np = p / kpool;
                    wmma::store_matrix_sync(&Ss[warp][0][0], acc[t], 20, wmma::mem_row_major);
                    __syncwarp();
                    if (lane < 16 && g + lane < min(g1, np) && p + 1 > top_k) {
                        float s2 = 0.0f;
#pragma unroll
                        for (int h = 0; h < NH; ++h) s2 += fmaxf(Ss[warp][h][lane], 0.0f) * Ws[t][h];
                        score[(size_t) t * ld_score + g + lane] = s2;
                    }
                    __syncwarp();
                }
        }
        return;
    }
    const int tt = tb * 8 + warp, pos = pos0 + t0 + tt;
    if (tt >= tt_n || pos + 1 <= top_k) return;
    const int npool = pos / kpool;
    const int per = ((n_pool_max + ngs - 1) / ngs + 15) / 16 * 16, g0 = gs * per, g1 = min(npool, g0 + per);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[KD / 16];
#pragma unroll
    for (int k = 0; k < KD / 16; ++k) wmma::load_matrix_sync(fa[k], &Qs[warp][0][k * 16], QLD);
    float* sc = score + (size_t) tt * ld_score;
    for (int g = g0; g < g1; g += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < KD / 16; ++k) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fb, pooled + (size_t) g * KD + k * 16, KD);
            wmma::mma_sync(acc, fa[k], fb, acc);
        }
        wmma::store_matrix_sync(&Ss[warp][0][0], acc, 20, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16 && g + lane < g1) {
            float s2 = 0.0f;
#pragma unroll
            for (int h = 0; h < NH; ++h) s2 += fmaxf(Ss[warp][h][lane], 0.0f) * Ws[warp][h];
            sc[g + lane] = s2;
        }
        __syncwarp();
    }
}

// Four queries share a tile of 64 pooled keys. Each warp keeps its query
// fragments in registers and scores two 16-key subtiles in the original K order.
__global__ void __launch_bounds__(256) kb_score_prefill(const float* iq, int ldq, const __half* pooled,
        const float* weights, int stride_w, int kpool, int top_k, const int* pos0p,
        int t0, int nt, int n_pool, float* score, int ld_score) {
    using namespace nvcuda;
    constexpr int D = 128, LD = 136, H = 16, QT = 4, GK = 64;
    __shared__ __align__(32) __half qs[QT][H][LD], keys[GK][LD];
    __shared__ __align__(32) float dots[8][H][20];
    __shared__ float ws[QT][H];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qb = blockIdx.y * QT;
    for (int i = tid; i < QT * H * D; i += 256) {
        int tk = i / (H*D), h = i / D % H, c = i % D;
        qs[tk][h][c] = __float2half_rn(qb+tk < nt ? iq[(size_t)(t0+qb+tk)*ldq + h*D+c] : 0.0f);
    }
    if (tid < QT*H) ws[tid/H][tid%H] = qb+tid/H < nt ? weights[(size_t)(t0+qb+tid/H)*stride_w+tid%H] : 0.0f;
    __syncthreads();
    int tk = warp >> 1, tt = qb+tk, pos = *pos0p+t0+tt;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a[D/16];
#pragma unroll
    for (int k=0; k<D/16; ++k) wmma::load_matrix_sync(a[k], &qs[tk][0][k*16], LD);
    for (int gb = blockIdx.x*GK; gb < n_pool; gb += gridDim.x*GK) {
        for (int i=tid; i<GK*(D/8); i+=256) {
            const int g=i/(D/8), c=8*(i%(D/8));
            uint4 v = make_uint4(0,0,0,0);
            if (gb+g<n_pool) v = *reinterpret_cast<const uint4*>(pooled+(size_t)(gb+g)*D+c);
            *reinterpret_cast<uint4*>(&keys[g][c])=v;
        }
        __syncthreads();
#pragma unroll
        for (int half=0; half<2; ++half) {
            const int first=32*half+16*(warp&1);
            wmma::fragment<wmma::accumulator,16,16,16,float> accum;
            wmma::fill_fragment(accum,0.0f);
#pragma unroll
            for (int k=0; k<D/16; ++k) {
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                wmma::load_matrix_sync(b, &keys[first][k*16], LD);
                wmma::mma_sync(accum,a[k],b,accum);
            }
            wmma::store_matrix_sync(&dots[warp][0][0],accum,20,wmma::mem_row_major);
            __syncwarp();
            const int g=gb+first+lane;
            if (lane<16 && tt<nt && pos+1>top_k && g<pos/kpool) {
                float v=0;
#pragma unroll
                for (int h=0; h<H; ++h) v += fmaxf(dots[warp][h][lane],0.0f)*ws[tk][h];
                score[(size_t)tt*ld_score+g]=v;
            }
            __syncwarp();
        }
        __syncthreads();
    }
}

// Decode index scores share a coalesced key tile across one to four queries.
// Keep the same WMMA operations and head reduction as kb_score_tc.
template<int Rows>
__global__ void __launch_bounds__(256) kd_score_tile(const float* iq, int ldq, const __half* pooled,
        const float* weights, int stride_w, int kpool, int top_k, const int* pos0p,
        int t0, int nt, int capacity, float* score, int ld_score) {
    using namespace nvcuda;
    constexpr int D=128, LD=136, H=16, KeyWarps=8/Rows, Groups=16*KeyWarps;
    __shared__ __align__(32) __half qs[Rows][H][LD], keys[Groups][LD];
    __shared__ __align__(32) float dots[8][16][16];
    __shared__ float weight[Rows][H];
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int pos0=*pos0p+t0;
    const int groups=min(capacity,(pos0+nt-1)/kpool);
    if ((int)blockIdx.x*Groups>=groups || pos0+nt<=top_k) return;
    const int token=warp/KeyWarps, key_start=16*(warp%KeyWarps);
    for (int i=tid; i<Rows*H*D; i+=256) {
        const int row=i/(H*D), h=i/D%H, d=i%D;
        qs[row][h][d]=row<nt ? __float2half_rn(iq[(size_t)(t0+row)*ldq+h*D+d]) : __float2half(0);
    }
    if (tid<Rows*H) weight[tid/H][tid%H]=tid/H<nt ? weights[(size_t)(t0+tid/H)*stride_w+tid%H] : 0;
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> q[D/16];
#pragma unroll
    for (int k=0; k<D/16; ++k) wmma::load_matrix_sync(q[k],&qs[token][0][k*16],LD);
    for (int gb=blockIdx.x*Groups; gb<groups; gb+=gridDim.x*Groups) {
        for (int i=tid; i<Groups*(D/8); i+=256) {
            const int g=i/(D/8), c=8*(i%(D/8));
            const uint4 v=gb+g<groups ? *reinterpret_cast<const uint4*>(pooled+(size_t)(gb+g)*D+c) : make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(&keys[g][c])=v;
        }
        __syncthreads();
        if (token<nt) {
            wmma::fragment<wmma::accumulator,16,16,16,float> a;
            wmma::fill_fragment(a,0.f);
#pragma unroll
            for (int k=0; k<D/16; ++k) {
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
                wmma::load_matrix_sync(b,&keys[key_start][k*16],LD);
                wmma::mma_sync(a,q[k],b,a);
            }
            wmma::store_matrix_sync(&dots[warp][0][0],a,16,wmma::mem_row_major);
            __syncwarp();
            const int g=gb+key_start+lane;
            if (lane<16 && pos0+token+1>top_k && g<(pos0+token)/kpool) {
                float value=0.f;
#pragma unroll
                for (int h=0; h<H; ++h) value+=fmaxf(dots[warp][h][lane],0.f)*weight[token][h];
                score[(size_t)token*ld_score+g]=value;
            }
        }
        __syncthreads();
    }
}

// (the few-token form of kb_score_tc: every warp a group sub-range for ALL the NT >= tt_n tokens, each key tile read once)
template <int NT>
__device__ __forceinline__ void score_few(const __half* __restrict__ pooled, int kpool, int top_k, int pos0, int t0, int tt_n,
                                          int n_pool_max, float* __restrict__ score, int ld_score, int gs, int ngs, int warp,
                                          int lane, const __half (*Qs)[16][136], float (*Ss)[16][20], const float (*Ws)[16]) {
    using namespace nvcuda;
    constexpr int KD = 128, NH = 16, QLD = KD + 8;
   // few tokens (decode): every warp a group sub-range for ALL the tokens, each key tile read once
        const int per_b = ((n_pool_max + ngs - 1) / ngs + 15) / 16 * 16, per = (per_b / 8 + 15) / 16 * 16;
        const int g0 = gs * per_b + warp * per;
        int npool_max_t = 0;
        for (int t = 0; t < tt_n; ++t) {
            const int p = pos0 + t0 + t;
            if (p + 1 > top_k) npool_max_t = max(npool_max_t, p / kpool);
        }
        const int g1 = min(min(npool_max_t, gs * per_b + per_b), g0 + per);
        for (int g = g0; g < g1; g += 16) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) wmma::fill_fragment(acc[t], 0.0f);
#pragma unroll
            for (int k = 0; k < KD / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, pooled + (size_t) g * KD + k * 16, KD);
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    if (t < tt_n) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa1;
                        wmma::load_matrix_sync(fa1, &Qs[t][0][k * 16], QLD);
                        wmma::mma_sync(acc[t], fa1, fb, acc[t]);
                    }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t)
                if (t < tt_n) {
                    const int p = pos0 + t0 + t, np = p / kpool;
                    wmma::store_matrix_sync(&Ss[warp][0][0], acc[t], 20, wmma::mem_row_major);
                    __syncwarp();
                    if (lane < 16 && g + lane < min(g1, np) && p + 1 > top_k) {
                        float s2 = 0.0f;
#pragma unroll
                        for (int h = 0; h < NH; ++h) s2 += fmaxf(Ss[warp][h][lane], 0.0f) * Ws[t][h];
                        score[(size_t) t * ld_score + g + lane] = s2;
                    }
                    __syncwarp();
                }
        }
        }

// the decode form of kb_score_tc for 5..8 tokens (a kernel of its own: its 8 accumulators would raise the registers of
// kb_score_tc, whose many-token form the prefill runs)
__global__ void __launch_bounds__(256) kd_score_tc8(const float* __restrict__ iq, int ldq, const __half* __restrict__ pooled,
                                                    const float* __restrict__ w, int ldw, int kpool, int top_k,
                                                    const int* pos0p, int t0, int tt_n, int n_pool_max,
                                                    float* __restrict__ score, int ld_score) {
    constexpr int KD = 128, NH = 16, QLD = KD + 8;
    __shared__ __align__(32) __half Qs[8][NH][QLD];
    __shared__ __align__(32) float Ss[8][16][20];
    __shared__ float Ws[8][NH];
    const int gs = blockIdx.x, ngs = gridDim.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos0 = *pos0p;
    for (int i = threadIdx.x; i < 8 * NH * KD; i += blockDim.x) {
        const int tk = i / (NH * KD), rem = i % (NH * KD), h = rem / KD, d = rem % KD;
        Qs[tk][h][d] = __float2half_rn(tk < tt_n ? iq[(size_t) (t0 + tk) * ldq + h * KD + d] : 0.0f);
    }
    if (threadIdx.x < 8 * NH) {
        const int tk = threadIdx.x / NH, h = threadIdx.x % NH;
        Ws[tk][h] = tk < tt_n ? w[(size_t) (t0 + tk) * ldw + h] : 0.0f;
    }
    __syncthreads();
    score_few<8>(pooled, kpool, top_k, pos0, t0, tt_n, n_pool_max, score, ld_score, gs, ngs, warp, lane, Qs, Ss, Ws);
}


}  // namespace kern
}  // namespace strata::hy4f
