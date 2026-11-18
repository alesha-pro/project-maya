// src/hy4f/hy4f_dkern.cuh - maya-hy4f: the multi-token decode step's own kernels (T <= 4 tokens: the speculative
// verify and the MTP draft layer).  Exchanges keep the decode's device step counters (graph-capturable).
#pragma once

#include "hy4f_kern.cuh"

#include <climits>

namespace strata::hy4f {
namespace kern {

// all-reduce of T rows (x1 + x2) [T][n], then per row y = RMS(sum) * w and X[t][s] += post[t][s] * y (X = R + post y
// with R = X; hc = 1 and post = 1 for the MTP's plain residual).  One block of 1024 threads, dynamic smem T*n floats.
__global__ void kd_ar_post(const float* x1, const float* x2, int T, int n, Peers P, int me, int n_shards, int* counter,
                           const float* nw, float eps, float* X, const float* post, int hc) {
    extern __shared__ __align__(16) float ys[];
    const int tn = T * n;
    for (int i = threadIdx.x; i < tn; i += blockDim.x) ys[i] = x1[i] + (x2 ? x2[i] : 0.0f);
    __syncthreads();
    if (n_shards > 1) {
        __shared__ int step;
        if (threadIdx.x == 0) step = ++counter[0];
        __syncthreads();
        const int par = step & 1;
        for (int r = 1; r <= n_shards; ++r) {   // one peer at a time
            const int j = (me + r) % n_shards;
            float4* dst = reinterpret_cast<float4*>(P.buf[j][par] + (size_t) me * tn);
            const float4* s4 = reinterpret_cast<const float4*>(ys);
            for (int i = threadIdx.x; i < tn / 4; i += blockDim.x) dst[i] = s4[i];
        }
        peers_signal_wait(P, me, n_shards, step);
        const volatile float* rv = P.buf[me][par];
        for (int i = threadIdx.x; i < tn; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < n_shards; ++j) s += rv[(size_t) j * tn + i];
            ys[i] = s;
        }
        __syncthreads();
    }
    for (int t = 0; t < T; ++t) {
        float sq = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) sq += ys[t * n + i] * ys[t * n + i];
        const float r = rsqrtf(block_sum(sq) / n + eps);
        float* Xt = X + (size_t) t * hc * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float y = ys[t * n + i] * r * nw[i];
            for (int s = 0; s < hc; ++s) Xt[(size_t) s * n + i] += post[t * 8 + s] * y;
        }
    }
}
// the context-parallel combine of T rows: part [T][H][W] (all heads, this shard's cells); shard j gets the head slice
// j of every shard and combines its heads with the sink into o [T][Hl][kvl].  One block of 1024 threads.
__global__ void kd_combine(const float* part, int T, int H, int kvl, int W, int h0, int Hl, const float* sinks_all,
                           Peers P, int me, int n_shards, int* counter, float* o) {
    __shared__ int step;
    if (threadIdx.x == 0) step = ++counter[0];
    __syncthreads();
    const int par = step & 1, slice = Hl * W, per = T * slice;
    for (int r = 1; r <= n_shards; ++r) {
        const int j = (me + r) % n_shards;
        float* dst = P.buf[j][par] + (size_t) me * per;
        for (int i = threadIdx.x; i < per; i += blockDim.x) {
            const int t = i / slice, rem = i % slice;
            dst[i] = part[(size_t) t * H * W + (size_t) j * slice + rem];
        }
    }
    peers_signal_wait(P, me, n_shards, step);
    const volatile float* rv = P.buf[me][par];
    for (int t = 0; t < T; ++t)
        for (int hl = 0; hl < Hl; ++hl) {
            const int h = h0 + hl;
            const size_t b0 = (size_t) t * slice + (size_t) hl * W;
            float m = sinks_all[h];
            for (int j = 0; j < n_shards; ++j) m = fmaxf(m, rv[(size_t) j * per + b0 + kvl]);
            float den = __expf(sinks_all[h] - m), f[4];
            for (int j = 0; j < n_shards; ++j) {
                const float mj = rv[(size_t) j * per + b0 + kvl];
                f[j] = mj == -INFINITY ? 0.0f : __expf(mj - m);
                den += f[j] * rv[(size_t) j * per + b0 + kvl + 1];
            }
            for (int i = threadIdx.x; i < kvl; i += blockDim.x) {
                float acc = 0.0f;
                for (int j = 0; j < n_shards; ++j) acc += f[j] * rv[(size_t) j * per + b0 + i];
                o[((size_t) t * Hl + hl) * kvl + i] = acc / den;
            }
        }
}
// Q8_1 rows for the experts: row r = (token r / K)'s first n values of x (ld floats per token)
__global__ void kd_q8_slots(const float* x, int ld, int n, int K, Q81* q) {
    const int r = blockIdx.y, t = r / K, i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[(size_t) t * ld + i] : 0.0f;
    if ((i & ~31) < n) q81_store(q + (size_t) r * (n / 32) + i / 32, v);
}
__global__ void kd_local_ids(const int* ids, int* loc, int n, int e0, int e1) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
        loc[k] = ids[k] >= e0 && ids[k] < e1 ? ids[k] - e0 : -1;
}
// moe[t][i] = sum over the token's slots this shard ran of w * dn[slot row]  (blockIdx.y = token)
__global__ void kd_moe_sum(float* moe, const float* dn, const float* w, const int* loc, int K, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int r = t * K + k;
            if (loc[r] >= 0) s += w[r] * dn[(size_t) r * n + i];
        }
        moe[(size_t) t * n + i] = s;
    }
}
// y[t][r] = W[r] . x[t] for T <= 4 rows of x, BF16 W [rows][n] read once (warp per row)
template <int T>
__global__ void kd_bf16_gemv(const uint16_t* __restrict__ W, const float* __restrict__ x, int n, int rows,
                             float* __restrict__ y) {
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(W + (size_t) r * n);
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int i = lane; i < n / 2; i += 32) {
        const uint32_t v = w2[i];
        const float a = __uint_as_float(v << 16), b = __uint_as_float(v & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float2 xx = reinterpret_cast<const float2*>(x + (size_t) t * n)[i];
            acc[t] += a * xx.x + b * xx.y;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t) t * rows + r] = s;
    }
}
// per row of y [T][rows]: (max, first index of the max + idx0); blockIdx.x = row, 1024 threads
__global__ void kd_argmax(const float* y, int rows, int idx0, float* oval, int* oidx) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float* r = y + (size_t) t * rows;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        const float v = r[i];
        if (v > bv) bv = v, bi = i;
    }
    auto red = [&]() {
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
        }
    };
    red();
    __shared__ float sv[32];
    __shared__ int si[32];
    if (lane == 0) sv[w] = bv, si[w] = bi;
    __syncthreads();
    if (w == 0) {
        const int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : INT_MAX;
        red();
        if (lane == 0) oval[t] = bv, oidx[t] = bi + idx0;
    }
}
// token rows (Q8_0 embedding) as floats [T][E] (the MTP's next-token input)
__global__ void kd_embed_rows(const uint8_t* emb, size_t rb, const int* toks, int E, float* y) {
    const int t = blockIdx.y;
    const uint8_t* row = emb + (size_t) toks[t] * rb;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) {
        const uint8_t* blk = row + (i / 32) * 34;
        y[(size_t) t * E + i] = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
    }
}

}  // namespace kern
}  // namespace strata::hy4f
