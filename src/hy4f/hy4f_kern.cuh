// src/hy4f/hy4f_kern.cuh - maya-hy4f v3 kernels: graph-ready (the position and the token read from device memory),
// Q8 latent KV, the tensor-parallel exchange.  Included by hy4f.cu only.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::hy4f {
namespace kern {

// ------------------------------------------------------------------------------------------------ reductions
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float sh[32];
    __shared__ float res;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? sh[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) res = t;
    }
    __syncthreads();
    return res;
}
__device__ __forceinline__ float block_max(float v) {
    __shared__ float sh[32];
    __shared__ float res;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? sh[lane] : -INFINITY;
        t = warp_max(t);
        if (lane == 0) res = t;
    }
    __syncthreads();
    return res;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }
__device__ __forceinline__ float fp8e4m3f(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float f;
    if (e == 0) f = ldexpf((float) m, -9);
    else if (e == 15 && m == 7) f = __int_as_float(0x7fc00000);
    else f = ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -f : f;
}
__device__ __forceinline__ float q8_0_dot(const uint8_t* row, const float* x, int n) {
    float s = 0.0f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) t += (float) q[i] * x[b * 32 + i];
        s += d * t;
    }
    return s;
}

// ------------------------------------------------------------------------------------------------ io
struct Io {   // the step's inputs, copied from pinned host memory at the head of every graph
    int pos;
    int tok;
};

__global__ void k_embed(const uint8_t* emb, size_t rb, const Io* io, float* X, float* R, int E, int hc, int* hist) {
    const int tok = io->tok;
    const uint8_t* row = emb + (size_t) tok * rb;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) {
        const uint8_t* blk = row + (i / 32) * 34;
        const float v = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
        for (int s = 0; s < hc; ++s) X[(size_t) s * E + i] = v, R[(size_t) s * E + i] = v;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) hist[io->pos] = tok;
}

// ------------------------------------------------------------------------------------------------ elementwise
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = block_sum(s);
    const float k = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * k * (w ? w[i] : 1.0f);
}
__global__ void k_f32_gemv(const float* W, const float* x, float* y, int n, int rows) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* w = W + (size_t) r * n;
    float s = 0.0f;
    for (int i = lane; i < n; i += 32) s += w[i] * x[i];
    s = warp_sum(s);
    if (lane == 0) y[r] = s;
}
__global__ void k_sigmul(float* a, const float* g, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] *= sigm(g[i]);
}
__global__ void k_swiglu(float* ga, const float* up, int n, float lim) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = fminf(ga[i], lim), u = fminf(fmaxf(up[i], -lim), lim);
        ga[i] = silu(g) * u;
    }
}
__global__ void k_add(const float* a, const float* b, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}
__global__ void k_copy(const float* a, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a[i];
}
__global__ void k_scale(float* x, int n, float s) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] *= s;
}
// RoPE NORM at the device position
__global__ void k_rope(float* x, int heads, int stride, int off, int n_rot, const Io* io, float theta) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, half = n_rot / 2;
    if (t >= heads * half) return;
    const int h = t / half, j = t % half;
    const float th = (float) io->pos * powf(theta, -2.0f * j / n_rot);
    float c, s;
    sincosf(th, &s, &c);
    float* p = x + (size_t) h * stride + off + 2 * j;
    const float a = p[0], b = p[1];
    p[0] = a * c - b * s;
    p[1] = a * s + b * c;
}

// ------------------------------------------------------------------------------------------------ HC
constexpr int HC_SPLIT = 16;
__global__ void k_hc_mix(const float* X, const float* fn, int n, float* part, float* ss) {
    const int r = blockIdx.x, s = blockIdx.y, chunk = n / HC_SPLIT, i0 = s * chunk;
    const float* w = fn + (size_t) r * n;
    float d = 0.0f, q = 0.0f;
    for (int i = i0 + threadIdx.x; i < i0 + chunk; i += blockDim.x) {
        const float x = X[i];
        d += w[i] * x;
        q += x * x;
    }
    d = block_sum(d);
    q = block_sum(q);
    if (threadIdx.x == 0) {
        part[r * HC_SPLIT + s] = d;
        if (r == 0) ss[s] = q;
    }
}
__global__ void k_hc_apply(const float* X, const float* part, const float* ss, const float* scale, const float* base,
                           int E, int hc, int n_mix, float eps, float hc_eps, float mag, float* cur, float* post) {
    __shared__ float pre[8];
    if (threadIdx.x < 32) {
        float q = threadIdx.x < HC_SPLIT ? ss[threadIdx.x] : 0.0f;
        q = warp_sum(q);
        const float inv = rsqrtf(q / (float) (E * hc) + eps);
        for (int r = 0; r < n_mix; ++r) {
            float m = threadIdx.x < HC_SPLIT ? part[r * HC_SPLIT + threadIdx.x] : 0.0f;
            m = warp_sum(m) * inv;
            if (threadIdx.x == 0) {
                if (r < hc) pre[r] = sigm(m * scale[0] + base[r]) + hc_eps;
                else if (blockIdx.x == 0) post[r - hc] = sigm(m * scale[1] + base[r]) * mag + hc_eps;
            }
        }
    }
    __syncthreads();
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < E; d += gridDim.x * blockDim.x) {
        float v = 0.0f;
        for (int i = 0; i < hc; ++i) v += pre[i] * X[(size_t) i * E + d];
        cur[d] = v;
    }
}
// X[i] = R[i] + post[i/E] * y  (R may equal X)
__global__ void k_hc_post(float* X, const float* R, const float* y, const float* post, int E, int hc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E * hc; i += gridDim.x * blockDim.x)
        X[i] = R[i] + post[i / E] * y[i % E];
}

// ------------------------------------------------------------------------------------------------ Q8 latent KV
// A row: int8 q[576] then half scales[9] (one per 64 values: c 0..511 in 8 groups, k_pe 512..575 in one): 594 bytes,
// padded to 608 (16-byte aligned rows for vector loads).
constexpr int KV_ROW = 576, KV_GROUPS = 9, KV_RB = 608;
constexpr int RING_PAD = 16;
#ifdef HY4F_DT6
constexpr int IRING = 16;   // (the DFly build: verify steps of up to 6 tokens)
#else
constexpr int IRING = 8;
#endif   // indexer ring rows (>= kpool + max step tokens - 1: a rejected draft must not overwrite a row of
                           // the still-open group), position p in row p % IRING   // SWA ring rows = window + RING_PAD: a multi-token step writes its rows before it reads the window   // 576 int8 + 9 half scales, padded to 16-byte rows
__device__ __forceinline__ float kv_val(const uint8_t* row, int j) {
    const __half* sc = reinterpret_cast<const __half*>(row + KV_ROW);
    return (float) reinterpret_cast<const int8_t*>(row)[j] * __half2float(sc[j >> 6]);
}
// quantize the F32 row into its slot: mode 0 = full cache (slot = pos), 1 = SWA ring (slot = pos % ring),
// 2 = context parallel (store only when pos % n_shards == shard, slot = pos / n_shards).  One block of 576 threads.
__global__ void k_kv_store(const float* row, uint8_t* cache, const Io* io, int mode, int ring, int n_shards, int shard) {
    const int pos = io->pos;
    long slot;
    if (mode == 0) slot = pos;
    else if (mode == 1) slot = pos % ring;
    else {
        if (pos % n_shards != shard) return;
        slot = pos / n_shards;
    }
    uint8_t* dst = cache + (size_t) slot * KV_RB;
    const int j = threadIdx.x;
    const float v = j < KV_ROW ? row[j] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float gmax[KV_GROUPS * 2];
    if ((j & 31) == 0) gmax[j >> 5] = a;
    __syncthreads();
    const int g = j >> 6;
    const float amax = fmaxf(gmax[2 * g], gmax[2 * g + 1]);
    const float d = amax / 127.0f;
    if (j < KV_ROW) reinterpret_cast<int8_t*>(dst)[j] = (int8_t) (d > 0.0f ? __float2int_rn(v / d) : 0);
    if (j < KV_GROUPS && (j & 0) == 0) {
        const float am = fmaxf(gmax[2 * j], gmax[2 * j + 1]);
        reinterpret_cast<__half*>(dst + KV_ROW)[j] = __float2half(am / 127.0f);
    }
}

// q_abs for heads [h0, h0+H) of a {nope, kvl, H_all} Q8_0 / BF16 k_b slice already starting at h0
__global__ void k_absorb_q8(const uint8_t* kb, const float* q, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * kvl) return;
    qabs[t] = q8_0_dot(kb + (size_t) t * (nope / 32 * 34), q + (size_t) (t / kvl) * qs, nope);
}
__global__ void k_absorb_bf16(const uint16_t* kb, const float* q, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * kvl) return;
    const uint16_t* w = kb + (size_t) t * nope;
    const float* x = q + (size_t) (t / kvl) * qs;
    float s = 0.0f;
    for (int d = 0; d < nope; ++d) s += bf16f(w[d]) * x[d];
    qabs[t] = s;
}
__global__ void k_vb_q8(const uint8_t* vb, const float* o, float* a, int kvl, int vh, int H) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint8_t* row = vb + (size_t) warp * (kvl / 32 * 34);
    const float* x = o + (size_t) h * kvl;
    float s = 0.0f;
    for (int b = lane; b < kvl / 32; b += 32) {
        const uint8_t* blk = row + b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) t += (float) q[i] * x[b * 32 + i];
        s += d * t;
    }
    s = warp_sum(s);
    if (lane == 0) a[warp] = s;
}
__global__ void k_vb_bf16(const uint16_t* vb, const float* o, float* a, int kvl, int vh, int H) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint16_t* w = vb + (size_t) warp * kvl;
    const float* x = o + (size_t) h * kvl;
    float s = 0.0f;
    for (int j = lane; j < kvl; j += 32) s += bf16f(w[j]) * x[j];
    s = warp_sum(s);
    if (lane == 0) a[warp] = s;
}

// Absorbed MLA over Q8 rows, one block per head, warp per cell for the scores.  Cells: SWA (cells == null) = the
// last `win` positions of the ring; otherwise cells[0..n_cap) with -1 padding, and with n_shards > 1 only the cells
// this shard owns (cell % n_shards == shard, row cell / n_shards) - a partial softmax.  part (n_shards > 1):
// [H][kvl + 2] = (sum_s e^(s - m) c_s, m, sum e^(s - m)); otherwise o = the normalized output with the sink.
template <int NT>
__global__ void k_attn_q8(const float* qabs, const float* q, int qs, int nope, const uint8_t* kv, int kvl, int rope,
                          const int* cells, int n_cap, const Io* io, int win, int n_shards, int shard,
                          const float* sinks, float scale, float* o, float* part, float* sc, int max_sel) {
    const int h = blockIdx.x;
    const float* qa = qabs + (size_t) h * kvl;
    const float* qp = q + (size_t) h * qs + nope;
    float* s_ = sc + (size_t) h * max_sel;
    const int pos = io->pos;
    int n, lo = 0;
    if (!cells) {
        lo = max(0, pos - win + 1);
        n = pos - lo + 1;
    } else {
        n = n_cap;
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = NT / 32;
    float mx = -INFINITY;
    for (int s = wid; s < n; s += nw) {
        int row = -1;
        if (!cells) row = (lo + s) % (win + RING_PAD);
        else {
            const int c = cells[s];
            if (c >= 0 && (n_shards == 1 || c % n_shards == shard)) row = n_shards == 1 ? c : c / n_shards;
        }
        float v = -INFINITY;
        if (row >= 0) {
            const uint8_t* r = kv + (size_t) row * KV_RB;
            float d = 0.0f;
            for (int j = lane; j < kvl; j += 32) d += qa[j] * kv_val(r, j);
            for (int j = lane; j < rope; j += 32) d += qp[j] * kv_val(r, kvl + j);
            d = warp_sum(d);
            v = d * scale;
        }
        if (lane == 0) s_[s] = v;
        mx = fmaxf(mx, v);
    }
    mx = block_max(mx);
    const bool partial = n_shards > 1;
    if (!partial) mx = fmaxf(mx, sinks[h]);
    float den = 0.0f;
    for (int s = threadIdx.x; s < n; s += NT) {
        const float e = s_[s] == -INFINITY || mx == -INFINITY ? 0.0f : __expf(s_[s] - mx);
        s_[s] = e;
        den += e;
    }
    den = block_sum(den);
    if (!partial) den += __expf(sinks[h] - mx);
    for (int j = threadIdx.x; j < kvl; j += NT) {
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
            const float e = s_[s];
            if (e == 0.0f) continue;
            int row;
            if (!cells) row = (lo + s) % (win + RING_PAD);
            else {
                const int c = cells[s];
                row = n_shards == 1 ? c : c / n_shards;
            }
            acc += e * kv_val(kv + (size_t) row * KV_RB, j);
        }
        if (partial) part[(size_t) h * (kvl + 2) + j] = acc;
        else o[(size_t) h * kvl + j] = acc / den;
    }
    if (partial && threadIdx.x == 0) {
        part[(size_t) h * (kvl + 2) + kvl] = mx;
        part[(size_t) h * (kvl + 2) + kvl + 1] = den;
    }
}

// ------------------------------------------------------------------------------------------------ indexer (graph-ready)
// iv / ig of the token into the ring slot pos % kpool (two rows of key_dim)
__global__ void k_ring_store(const float* iv, const float* ig, float* ring, int kd, int kpool, const Io* io) {
    const int slot = io->pos % kpool;
    for (int i = threadIdx.x; i < kd; i += blockDim.x) {
        ring[(size_t) slot * 2 * kd + i] = iv[i];
        ring[(size_t) slot * 2 * kd + kd + i] = ig[i];
    }
}
// when the token completes its group: pooled[g] = rms(sum_m softmax_m(ig_m + ape_m) iv_m) * norm, RoPE NORM on the
// last n_rot dims at the group's first position.  One block of kd threads.
__global__ void k_pool(const float* ring, const float* ape, const float* norm, float* pooled, int kd, int kpool, int n_rot,
                       float eps, float theta, const Io* io) {
    const int pos = io->pos;
    if ((pos + 1) % kpool != 0) return;
    const int g = pos / kpool, d = threadIdx.x;
    __shared__ float vals[256];
    float v = 0.0f;
    if (d < kd) {
        float lg[8], mx = -INFINITY, den = 0.0f;
        for (int m = 0; m < kpool; ++m) {
            lg[m] = ring[(size_t) m * 2 * kd + kd + d] + ape[(size_t) m * kd + d];
            mx = fmaxf(mx, lg[m]);
        }
        for (int m = 0; m < kpool; ++m) den += (lg[m] = __expf(lg[m] - mx));
        for (int m = 0; m < kpool; ++m) v += lg[m] / den * ring[(size_t) m * 2 * kd + d];
    }
    const float k = rsqrtf(block_sum(d < kd ? v * v : 0.0f) / kd + eps);
    if (d < kd) vals[d] = v * k * norm[d];
    __syncthreads();
    const int half = n_rot / 2, base = kd - n_rot;
    if (d < half) {
        const float th = (float) (g * kpool) * powf(theta, -2.0f * d / n_rot);
        float c, s;
        sincosf(th, &s, &c);
        const float a = vals[base + 2 * d], b = vals[base + 2 * d + 1];
        vals[base + 2 * d] = a * c - b * s;
        vals[base + 2 * d + 1] = a * s + b * c;
    }
    __syncthreads();
    if (d < kd) pooled[(size_t) g * kd + d] = vals[d];
}
// score[g] = sum_h relu(iq_h . pooled_g) * w_h over the candidate groups g < pos / kpool (warp per group)
__global__ void k_score(const float* iq, const __half* pooled, const float* w, int kd, int nh, int kpool, int top_k,
                        const Io* io, float* score) {
    const int pos = io->pos;
    if (pos + 1 <= top_k) return;
    const int n_pool = pos / kpool;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_pool) return;
    const __half* pk = pooled + (size_t) warp * kd;
    float pv[4];
    for (int i = 0; i < 4; ++i) pv[i] = lane + 32 * i < kd ? __half2float(pk[lane + 32 * i]) : 0.0f;
    float s = 0.0f;
    for (int h = 0; h < nh; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 4; ++i) d += lane + 32 * i < kd ? iq[(size_t) h * kd + lane + 32 * i] * pv[i] : 0.0f;
        d = warp_sum(d);
        s += fmaxf(d, 0.0f) * w[h];
    }
    if (lane == 0) score[warp] = s;
}

// ------------------------------------------------------------------------------------------------ Engram
__global__ void k_engram_hash(const int* hist, const Io* io, const int64_t* map, int64_t n_map, int64_t pad_id,
                              const uint64_t* mult, const uint64_t* vsz, const uint64_t* off, int ng, int nh,
                              int64_t* ids) {
    const int col = threadIdx.x, ncol = (ng - 1) * nh, pos = io->pos;
    if (col >= ncol) return;
    const int64_t pad = map[pad_id];
    auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= n_map ? -1 : map[t]; };
    const int o_hi = col / nh + 1;
    int64_t c = cid(hist[pos]);
    uint64_t rolling = (uint64_t) (c < 0 ? pad : c) * mult[0];
    for (int oo = 1; oo <= o_hi; ++oo) {
        c = pos - oo < 0 ? -1 : cid(hist[pos - oo]);
        rolling ^= (uint64_t) (c < 0 ? pad : c) * mult[oo];
    }
    ids[col] = (int64_t) (rolling % vsz[col] + off[col]);
}
__global__ void k_engram_rows(const uint8_t* table, const int64_t* ids, const float* lnorm, float* r, int D, float eps) {
    const int col = blockIdx.x, d = threadIdx.x;
    const float v = d < D ? fp8e4m3f(table[(size_t) ids[col] * D + d]) : 0.0f;
    const float k = rsqrtf(block_sum(v * v) / D + eps);
    if (d < D) r[(size_t) col * D + d] = v * k * lnorm[(size_t) col * D + d];
}
__global__ void k_engram_mix(float* X, float* R, const float* kv, const float* kvb, const float* gw, const float* hcs,
                             const float* gamma, int E, int hc, float eps) {
    const float* val = kv + (size_t) hc * E;
    const float* valb = kvb + (size_t) hc * E;
    float s = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        const float v = val[d] + valb[d];
        s += v * v;
    }
    const float kv_ = rsqrtf(block_sum(s) / E + eps);
    for (int i = 0; i < hc; ++i) {
        float* x = X + (size_t) i * E;
        const float* k = kv + (size_t) i * E;
        const float* kb = kvb + (size_t) i * E;
        float sx = 0.0f, sk = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) {
            const float kk = k[d] + kb[d];
            sx += x[d] * x[d];
            sk += kk * kk;
        }
        const float ix = rsqrtf(block_sum(sx) / E + eps), ik = rsqrtf(block_sum(sk) / E + eps);
        float dt = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) dt += x[d] * ix * (k[d] + kb[d]) * ik * gw[(size_t) i * E + d];
        dt = block_sum(dt) * rsqrtf((float) E);
        const float mag = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        const float gate = sigm(dt < 0.0f ? -mag : mag);
        __syncthreads();
        for (int d = threadIdx.x; d < E; d += blockDim.x) {
            x[d] += (val[d] + valb[d]) * kv_ * gamma[d] * (gate + hcs[(size_t) i * E + d]);
            R[(size_t) i * E + d] = x[d];
        }
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ MoE (expert parallel)
// local ids: the experts this shard owns (e0 <= id < e1) keep their local index, the others -1
__global__ void k_local_ids(const int* ids, int* loc, int n, int e0, int e1) {
    const int k = threadIdx.x;
    if (k < n) loc[k] = ids[k] >= e0 && ids[k] < e1 ? ids[k] - e0 : -1;
}
__global__ void k_moe_sum(float* moe, const float* dn, const float* w, const int* loc, int k_n, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < k_n; ++k)
            if (loc[k] >= 0) s += w[k] * dn[(size_t) k * n + i];
        moe[i] = s;
    }
}

// ------------------------------------------------------------------------------------------------ tensor-parallel exchange
// Push model: every shard WRITES its payload (posted P2P stores, all threads at once) into each peer's receive area
// recv[parity][src][payload], fences, stores the step into the peer's flag slot for it, waits for every peer's flag,
// then reads ONLY local memory.  Two receive areas by parity: a peer can be at most one step ahead before it waits on
// our next flag, so the area we read is never overwritten.
// a stalled exchange (a peer died or a launch failed) traps after ~10 s instead of spinning forever: the process gets
// an error the server can restart on
__device__ __forceinline__ unsigned long long hy4f_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void hy4f_spin_guard(unsigned long long t0) {
    if (hy4f_now() - t0 > 10000000000ull) {
        printf("hy4f: exchange stalled (block %d thread %d)\n", (int) blockIdx.x, (int) threadIdx.x);
        __trap();
    }
}
struct Peers {
    float* buf[4][2];   // shard j's receive area of parity p (device pointers, peer-accessible)
    int* flag[4];       // shard j's flag array [4]
};
struct PeersD : Peers {   // the decode exchanges' (hy4f_dkern.cuh): + the all-reduce wire format
    int bf = 0;           // 0 F32 + flags, 1 BF16 + flags, 2 LL (2 BF16 + the step per 8-byte store; every shard rounds its own)
};

__device__ __forceinline__ void peers_signal_wait(const Peers& P, int me, int n_shards, int step) {
    __syncthreads();
    if (threadIdx.x < n_shards) {
        __threadfence_system();
        volatile int* f = P.flag[threadIdx.x];
        f[me] = step;
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[threadIdx.x] < step) hy4f_spin_guard(t0);
    }
    __syncthreads();
    __threadfence_system();
}
// all-reduce of n floats: x (in/out) on every shard ends up the sum over the shards (same order everywhere)
__global__ void k_allreduce(float* x, int n, Peers P, int me, int n_shards, int* counter) {
    __shared__ int step;
    if (threadIdx.x == 0) step = ++counter[0];
    __syncthreads();
    const int par = step & 1;
    for (int j = 0; j < n_shards; ++j) {
        float* dst = P.buf[j][par] + (size_t) me * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = x[i];
    }
    peers_signal_wait(P, me, n_shards, step);
    const volatile float* rv = P.buf[me][par];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = 0.0f;
        for (int j = 0; j < n_shards; ++j) s += rv[(size_t) j * n + i];
        x[i] = s;
    }
}
// the context-parallel attention combine: part [H][kvl+2] for ALL heads on every shard; shard j receives the slices
// of its heads [j*Hl, (j+1)*Hl) from every shard and combines them with the sink:
// o = sum_g e^(m_g - m) p_g / (sum_g e^(m_g - m) s_g + e^(sink - m))
__global__ void k_attn_combine(const float* part, int H, int kvl, int h0, int Hl, const float* sinks_all, Peers P,
                               int me, int n_shards, int* counter, float* o) {
    __shared__ int step;
    if (threadIdx.x == 0) step = ++counter[0];
    __syncthreads();
    const int par = step & 1, W = kvl + 2, slice = Hl * W;
    for (int j = 0; j < n_shards; ++j) {
        float* dst = P.buf[j][par] + (size_t) me * slice;
        const float* src = part + (size_t) j * slice;
        for (int i = threadIdx.x; i < slice; i += blockDim.x) dst[i] = src[i];
    }
    peers_signal_wait(P, me, n_shards, step);
    const volatile float* rv = P.buf[me][par];
    for (int hl = 0; hl < Hl; ++hl) {
        const int h = h0 + hl;
        float m = sinks_all[h];
        for (int j = 0; j < n_shards; ++j) m = fmaxf(m, rv[(size_t) j * slice + (size_t) hl * W + kvl]);
        float den = __expf(sinks_all[h] - m);
        float f[4];
        for (int j = 0; j < n_shards; ++j) {
            const float mj = rv[(size_t) j * slice + (size_t) hl * W + kvl];
            f[j] = mj == -INFINITY ? 0.0f : __expf(mj - m);
            den += f[j] * rv[(size_t) j * slice + (size_t) hl * W + kvl + 1];
        }
        for (int i = threadIdx.x; i < kvl; i += blockDim.x) {
            float acc = 0.0f;
            for (int j = 0; j < n_shards; ++j) acc += f[j] * rv[(size_t) j * slice + (size_t) hl * W + i];
            o[(size_t) hl * kvl + i] = acc / den;
        }
    }
}


// ------------------------------------------------------------------------------------------------ v3.1 fused kernels
struct Q81 {   // Maya's native MMVQ activation block
    half2 ds;
    int8_t qs[32];
};
// quantize one warp's 32 consecutive values (lane i holds v) into a Q8_1 block
__device__ __forceinline__ void q81_store(Q81* blk, float v) {
    const int lane = threadIdx.x & 31;
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    blk->qs[lane] = (int8_t) (amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(v / d))));
    if (lane == 0) blk->ds = make_half2(__float2half(d), __float2half(fmaxf(-65504.0f, fminf(65504.0f, sum))));
}
// HC apply + RMS * w + Q8_1: one block of 1024 threads, n = E (a multiple of 1024 * k).  x (F32) and xq (Q8_1).
__global__ void k_hc_apply_norm(const float* X, const float* part, const float* ss, const float* scale, const float* base,
                                const float* nw, int E, int hc, int n_mix, float eps, float hc_eps, float mag, float* x,
                                Q81* xq, float* post) {
    __shared__ float pre[8];
    if (threadIdx.x < 32) {
        float q = threadIdx.x < HC_SPLIT ? ss[threadIdx.x] : 0.0f;
        q = warp_sum(q);
        const float inv = rsqrtf(q / (float) (E * hc) + eps);
        for (int r = 0; r < n_mix; ++r) {
            float m = threadIdx.x < HC_SPLIT ? part[r * HC_SPLIT + threadIdx.x] : 0.0f;
            m = warp_sum(m) * inv;
            if (threadIdx.x == 0) {
                if (r < hc) pre[r] = sigm(m * scale[0] + base[r]) + hc_eps;
                else post[r - hc] = sigm(m * scale[1] + base[r]) * mag + hc_eps;
            }
        }
    }
    __syncthreads();
    float cur[4];
    float sq = 0.0f;
    const int per = E / blockDim.x;
    for (int k = 0; k < per; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        float v = 0.0f;
        for (int i = 0; i < hc; ++i) v += pre[i] * X[(size_t) i * E + d];
        cur[k] = v;
        sq += v * v;
    }
    const float r = rsqrtf(block_sum(sq) / E + eps);
    for (int k = 0; k < per; ++k) {
        const int d = threadIdx.x + k * blockDim.x;
        const float v = cur[k] * r * nw[d];
        x[d] = v;
        q81_store(xq + d / 32, v);
    }
}
// y = RMS(x) * w -> Q8_1 (n a multiple of 32, one block)
__global__ void k_rms_q8(const float* x, const float* w, int n, float eps, float* y, Q81* yq) {
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq += x[i] * x[i];
    const float r = rsqrtf(block_sum(sq) / n + eps);
    for (int i = threadIdx.x; i < ((n + 31) / 32) * 32; i += blockDim.x) {
        const float v = i < n ? x[i] * r * w[i] : 0.0f;
        if (i < n && y) y[i] = v;
        q81_store(yq + i / 32, v);
    }
}
// any F32 vector -> Q8_1 (n a multiple of 32)
__global__ void k_q8(const float* x, int n, Q81* q) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    if ((i & ~31) < n) q81_store(q + i / 32, v);
}
// the latent row: RoPE on k_pe (its last `rope` values) at the device position, RMS * w on c (kvl), Q8 into the slot
// (mode as k_kv_store).  One block of 576 threads.
__global__ void k_kv_prep(const float* kvin, const float* nw, int kvl, int rope, float eps, float theta, uint8_t* cache,
                          const Io* io, int mode, int ring, int n_shards, int shard, float* kvdump) {
    const int pos = io->pos, j = threadIdx.x;
    __shared__ float row[KV_ROW];
    float v = j < KV_ROW ? kvin[j] : 0.0f;
    const float c2 = j < kvl ? v * v : 0.0f;
    const float r = rsqrtf(block_sum(c2) / kvl + eps);
    if (j < kvl) row[j] = v * r * nw[j];
    else if (j < KV_ROW) row[j] = v;
    __syncthreads();
    if (j >= kvl && j < KV_ROW && ((j - kvl) & 1) == 0) {
        const int jj = (j - kvl) / 2;
        const float th = (float) pos * powf(theta, -2.0f * jj / rope);
        float c, s;
        sincosf(th, &s, &c);
        const float a = row[j], b = row[j + 1];
        row[j] = a * c - b * s;
        row[j + 1] = a * s + b * c;
    }
    __syncthreads();
    if (kvdump && j < KV_ROW) kvdump[j] = row[j];
    long slot;
    if (mode == 0) slot = pos;
    else if (mode == 1) slot = pos % ring;
    else {
        if (pos % n_shards != shard) return;
        slot = pos / n_shards;
    }
    uint8_t* dst = cache + (size_t) slot * KV_RB;
    v = j < KV_ROW ? row[j] : 0.0f;
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float gmax[KV_GROUPS * 2];
    if ((j & 31) == 0) gmax[j >> 5] = a;
    __syncthreads();
    const int g = j >> 6;
    const float d = fmaxf(gmax[2 * g], gmax[2 * g + 1]) / 127.0f;
    if (j < KV_ROW) reinterpret_cast<int8_t*>(dst)[j] = (int8_t) (d > 0.0f ? __float2int_rn(v / d) : 0);
    if (j < KV_GROUPS) reinterpret_cast<__half*>(dst + KV_ROW)[j] = __float2half(fmaxf(gmax[2 * j], gmax[2 * j + 1]) / 127.0f);
}
// Absorbed MLA, one block (8 warps) per head, one pass with an online softmax per warp: lane l owns c dims
// [16 l, 16 l + 16) and k_pe dims {2 l, 2 l + 1}; q_pe is RoPE'd here at the device position.  Cells as k_attn_q8.
// Output: full (n_shards == 1) o = normalized with the sink; partial: part [H][kvl + 2] = (sum e^(s-m) c, m, l).
__global__ void __launch_bounds__(256) k_attn2(const float* qabs, const float* q, int qs, int nope, float theta,
                                               const uint8_t* kv, const int* cells, int n_cap, const Io* io, int win,
                                               int n_shards, int shard, const float* sinks, float scale, float* o,
                                               float* part) {
    constexpr int KVL = 512, ROPE = 64, NW = 8;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, pos = io->pos;
    float qa[16];
    {
        const float4* qa4 = reinterpret_cast<const float4*>(qabs + (size_t) h * KVL + lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 t = qa4[i];
            qa[4 * i] = t.x, qa[4 * i + 1] = t.y, qa[4 * i + 2] = t.z, qa[4 * i + 3] = t.w;
        }
    }
    float qp0, qp1;
    {
        const float* qp = q + (size_t) h * qs + nope + 2 * lane;
        const float th = (float) pos * powf(theta, -2.0f * lane / ROPE);
        float c, s;
        sincosf(th, &s, &c);
        qp0 = qp[0] * c - qp[1] * s;
        qp1 = qp[0] * s + qp[1] * c;
    }
    int n, lo = 0;
    if (!cells) {
        lo = max(0, pos - win + 1);
        n = pos - lo + 1;
    } else {
        n = n_cap;
    }
    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int s = w; s < n; s += NW) {
        int row;
        if (!cells) row = (lo + s) % (win + RING_PAD);
        else {
            const int c = cells[s];
            row = (c >= 0 && (n_shards == 1 || c % n_shards == shard)) ? (n_shards == 1 ? c : c / n_shards) : -1;
        }
        if (row < 0) continue;
        const uint8_t* r = kv + (size_t) row * KV_RB;
        const __half* sc = reinterpret_cast<const __half*>(r + KV_ROW);
        const int4 cv = *reinterpret_cast<const int4*>(r + lane * 16);
        const int8_t* c8 = reinterpret_cast<const int8_t*>(&cv);
        const float cs = __half2float(sc[lane >> 2]), ps = __half2float(sc[8]);
        const char2 p2 = *reinterpret_cast<const char2*>(r + KVL + 2 * lane);
        float cval[16];
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            cval[i] = (float) c8[i] * cs;
            d += qa[i] * cval[i];
        }
        d += qp0 * (float) p2.x * ps + qp1 * (float) p2.y * ps;
        d = warp_sum(d) * scale;
        const float mn = fmaxf(m, d), f = __expf(m - mn), e = __expf(d - mn);
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = acc[i] * f + e * cval[i];
        l = l * f + e;
        m = mn;
    }
    __shared__ float sm[NW], sl[NW], sacc[NW][KVL];
    if (lane == 0) sm[w] = m, sl[w] = l;
#pragma unroll
    for (int i = 0; i < 16; ++i) sacc[w][lane * 16 + i] = acc[i];
    __syncthreads();
    const bool partial = n_shards > 1;
    float M = partial ? -INFINITY : sinks[h];
    for (int i = 0; i < NW; ++i) M = fmaxf(M, sm[i]);
    float fw[NW], L = partial ? 0.0f : __expf(sinks[h] - M);
#pragma unroll
    for (int i = 0; i < NW; ++i) {
        fw[i] = sm[i] == -INFINITY ? 0.0f : __expf(sm[i] - M);
        L += fw[i] * sl[i];
    }
    for (int j = threadIdx.x; j < KVL; j += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < NW; ++i) v += fw[i] * sacc[i][j];
        if (partial) part[(size_t) h * (KVL + 2) + j] = v;
        else o[(size_t) h * KVL + j] = v / L;
    }
    if (partial && threadIdx.x == 0) {
        part[(size_t) h * (KVL + 2) + KVL] = M;
        part[(size_t) h * (KVL + 2) + KVL + 1] = L;
    }
}
// a[h*vh + i] = (sum_j v_b[j, i, h] o[h*kvl + j]) * sigmoid(gate[h*vh + i]); warp per output (Q8_0 v_b)
__global__ void k_vb_gate_q8(const uint8_t* vb, const float* o, const float* gate, float* a, int kvl, int vh, int H) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint8_t* row = vb + (size_t) warp * (kvl / 32 * 34);
    const float* x = o + (size_t) h * kvl;
    float s = 0.0f;
    for (int b = lane; b < kvl / 32; b += 32) {
        const uint8_t* blk = row + b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) t += (float) q[i] * x[b * 32 + i];
        s += d * t;
    }
    s = warp_sum(s);
    if (lane == 0) a[warp] = s * sigm(gate[warp]);
}
__global__ void k_vb_gate_bf16(const uint16_t* vb, const float* o, const float* gate, float* a, int kvl, int vh, int H) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint16_t* w = vb + (size_t) warp * kvl;
    const float* x = o + (size_t) h * kvl;
    float s = 0.0f;
    for (int j = lane; j < kvl; j += 32) s += bf16f(w[j]) * x[j];
    s = warp_sum(s);
    if (lane == 0) a[warp] = s * sigm(gate[warp]);
}
// SwiGLU (+ clamp) -> Q8_1 of n values (n a multiple of 32); ga/up separate arrays
__global__ void k_swiglu_q8(const float* ga, const float* up, int n, float lim, Q81* q, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const float g = fminf(ga[i], lim), u = fminf(fmaxf(up[i], -lim), lim);
        v = silu(g) * u;
        if (out) out[i] = v;
    }
    if ((i & ~31) < n) q81_store(q + i / 32, v);
}
// all-reduce (push model) of x1 (+ x2) -> y = RMS(sum) * w -> X = R + post * y, R = X; one block of 1024 threads,
// n == E <= 4096.  n_shards == 1: no exchange.
__global__ void k_ar_post(const float* x1, const float* x2, int n, Peers P, int me, int n_shards, int* counter,
                          const float* nw, float eps, float* X, float* R, const float* post, int hc, float* ydump) {
    __shared__ float ys[4096];
    for (int i = threadIdx.x; i < n; i += blockDim.x) ys[i] = x1[i] + (x2 ? x2[i] : 0.0f);
    if (n_shards > 1) {
        __shared__ int step;
        if (threadIdx.x == 0) step = ++counter[0];
        __syncthreads();
        const int par = step & 1;
        for (int j = 0; j < n_shards; ++j) {
            float* dst = P.buf[j][par] + (size_t) me * n;
            for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = ys[i];
        }
        peers_signal_wait(P, me, n_shards, step);
        const volatile float* rv = P.buf[me][par];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < n_shards; ++j) s += rv[(size_t) j * n + i];
            ys[i] = s;
        }
    }
    __syncthreads();
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq += ys[i] * ys[i];
    const float r = rsqrtf(block_sum(sq) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = ys[i] * r * nw[i];
        if (ydump) ydump[i] = y;
        for (int s = 0; s < hc; ++s) {
            const float v = R[(size_t) s * n + i] + post[s] * y;
            X[(size_t) s * n + i] = v;
            R[(size_t) s * n + i] = v;
        }
    }
}
// the indexer's value/gate pair into the ring + pooling when the group completes (k_ring_store + k_pool fused)
__global__ void k_ring_pool(const float* ivg, float* ring, const float* ape, const float* norm, __half* pooled, int kd,
                            int kpool, int n_rot, float eps, float theta, const Io* io) {
    const int pos = io->pos, slot = pos % IRING, d = threadIdx.x;
    if (d < kd) {
        ring[(size_t) slot * 2 * kd + d] = ivg[d];
        ring[(size_t) slot * 2 * kd + kd + d] = ivg[kd + d];
    }
    __syncthreads();
    if ((pos + 1) % kpool != 0) return;
    const int g = pos / kpool;
    __shared__ float vals[256];
    float v = 0.0f;
    if (d < kd) {
        float lg[8], mx = -INFINITY, den = 0.0f;
        for (int mm = 0; mm < kpool; ++mm) {
            lg[mm] = ring[(size_t) ((g * kpool + mm) % IRING) * 2 * kd + kd + d] + ape[(size_t) mm * kd + d];
            mx = fmaxf(mx, lg[mm]);
        }
        for (int mm = 0; mm < kpool; ++mm) den += (lg[mm] = __expf(lg[mm] - mx));
        for (int mm = 0; mm < kpool; ++mm) v += lg[mm] / den * ring[(size_t) ((g * kpool + mm) % IRING) * 2 * kd + d];
    }
    const float k = rsqrtf(block_sum(d < kd ? v * v : 0.0f) / kd + eps);
    if (d < kd) vals[d] = v * k * norm[d];
    __syncthreads();
    const int half = n_rot / 2, base = kd - n_rot;
    if (d < half) {
        const float th = (float) (g * kpool) * powf(theta, -2.0f * d / n_rot);
        float c, s;
        sincosf(th, &s, &c);
        const float a = vals[base + 2 * d], b = vals[base + 2 * d + 1];
        vals[base + 2 * d] = a * c - b * s;
        vals[base + 2 * d + 1] = a * s + b * c;
    }
    __syncthreads();
    if (d < kd) pooled[(size_t) g * kd + d] = __float2half_rn(vals[d]);
}

// SwiGLU (+ clamp) of [rows][gate n_ff | up n_ff] -> Q8_1 of rows * n_ff (row-major h)
__global__ void k_swiglu_gu_q8(const float* gu, int n_ff, int n, float lim, Q81* q) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const int r = i / n_ff, c = i % n_ff;
        const float g = fminf(gu[(size_t) r * 2 * n_ff + c], lim), u = fminf(fmaxf(gu[(size_t) r * 2 * n_ff + n_ff + c], -lim), lim);
        v = silu(g) * u;
    }
    if ((i & ~31) < n) q81_store(q + i / 32, v);
}
}  // namespace kern
}  // namespace strata::hy4f
