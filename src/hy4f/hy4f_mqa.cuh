// src/hy4f/hy4f_mqa.cuh - maya-hy4f: the batched absorbed-MLA attention on tensor cores (WMMA fp16, fp32 accumulate).
// MLA absorbed is MQA: every head reads the same 576-wide latent row, so one block per token takes ALL the shard's heads
// and each gathered KV tile is read once for H heads (kb_attn2 read it H times).  Two passes over the rows: the row
// max / sum, then P = exp(S - m) / l and O += P V - no rescaling of the accumulator fragments.
//   SWA (cells null): the window over the ring (positions < pos0) and the chunk buffer (>= pos0); full softmax with sink.
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

template <int H>
__global__ void __launch_bounds__(256) kb_mqa_attn(const float* __restrict__ qabs, const float* __restrict__ q, int ldq,
                                                   int qs, int nope, float theta, const uint8_t* __restrict__ kv,
                                                   const uint8_t* __restrict__ chunk, const int* __restrict__ cells,
                                                   int n_cap, int pos0, int win, int n_shards, int shard,
                                                   const float* __restrict__ sinks, float scale, float* __restrict__ o,
                                                   float* __restrict__ part, int W) {
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

    const int t = blockIdx.x, pos = pos0 + t, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
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
    if (!cells) {
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

    for (int pass = 0; pass < 2; ++pass) {
        for (int r0 = 0; r0 < n; r0 += TR) {
            __syncthreads();   // (the previous tile's readers are done)
            if (tid < TR) {
                const int s = r0 + tid;
                const uint8_t* p = nullptr;
                if (s < n) {
                    if (!cells) {
                        const int pp = lo + s;
                        p = pp >= pos0 ? chunk + (size_t) (pp - pos0) * KV_RB : kv + (size_t) (pp % win) * KV_RB;
                    } else {
                        p = kv + (size_t) cl[s] * KV_RB;
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
            __syncthreads();
            {   // S partials
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
#pragma unroll
                for (int k = 0; k < KS; ++k) s += Sp[(size_t) k * H * 32 + h * 32 + lane];
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

}  // namespace kern
}  // namespace strata::hy4f
