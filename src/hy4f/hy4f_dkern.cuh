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


// ---- multi-block exchanges (the single-block ones were bound by one SM's stores and loads)
// all-reduce of T rows (x1 + x2) [T][n] + post: NB blocks push their slice to the peers (ring order), the last block to
// arrive signals; blocks t < T then wait and do row t: the sum in shard order (the same on every shard), y = RMS * w,
// X[t][s] += post[t][s] y.  counter: the step (read by all, bumped by the last block); arrive: the block count.
__global__ void kd_ar2(const float* __restrict__ x1, const float* __restrict__ x2, int T, int n, Peers P, int me,
                       int n_shards, int* counter, int* arrive, const float* __restrict__ nw, float eps, float* X,
                       const float* __restrict__ post, int hc) {
    const int step = counter[0] + 1, tn = T * n, par = step & 1, n4 = tn / 4;
    const int per = (n4 + gridDim.x - 1) / gridDim.x, b0 = blockIdx.x * per, b1 = min(n4, b0 + per);
    const float4* a4 = reinterpret_cast<const float4*>(x1);
    const float4* c4 = reinterpret_cast<const float4*>(x2);
    for (int r = 1; r < n_shards; ++r) {
        const int j = (me + r) % n_shards;
        float4* dst = reinterpret_cast<float4*>(P.buf[j][par] + (size_t) me * tn);
        for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) {
            float4 v = a4[i];
            if (x2) {
                const float4 w = c4[i];
                v.x += w.x, v.y += w.y, v.z += w.z, v.w += w.w;
            }
            dst[i] = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive, 1);
        if (v == (int) gridDim.x - 1) {
            __threadfence_system();
            for (int j = 0; j < n_shards; ++j)
                if (j != me) reinterpret_cast<volatile int*>(P.flag[j])[me] = step;
            *arrive = 0;
            counter[0] = step;
        }
    }
    if ((int) blockIdx.x >= T) return;
    if (threadIdx.x < n_shards && (int) threadIdx.x != me) {
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[threadIdx.x] < step) hy4f_spin_guard(t0);
    }
    __syncthreads();
    __threadfence_system();
    const int t = blockIdx.x;
    __shared__ float ys[4096];
    const float* rv = P.buf[me][par];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = 0.0f;
        for (int j = 0; j < n_shards; ++j)
            v += j == me ? x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f)
                         : __ldcv(rv + (size_t) j * tn + (size_t) t * n + i);
        ys[i] = v;
        sq += v * v;
    }
    const float r = rsqrtf(block_sum(sq) / n + eps);
    float* Xt = X + (size_t) t * hc * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = ys[i] * r * nw[i];
        for (int s = 0; s < hc; ++s) Xt[(size_t) s * n + i] += post[t * 8 + s] * y;
    }
}
// the combine of T rows over NB blocks: slices pushed in ring order; blocks b < T*Hl combine (t, hl) = (b / Hl, b % Hl)
__global__ void kd_combine2(const float* __restrict__ part, int T, int H, int kvl, int W, int h0, int Hl,
                            const float* __restrict__ sinks_all, Peers P, int me, int n_shards, int* counter, int* arrive,
                            float* __restrict__ o) {
    const int step = counter[0] + 1, par = step & 1, slice = Hl * W, per = T * slice;
    const int pb = (per + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * pb, i1 = min(per, i0 + pb);
    for (int r = 1; r < n_shards; ++r) {
        const int j = (me + r) % n_shards;
        float* dst = P.buf[j][par] + (size_t) me * per;
        for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
            const int t = i / slice, rem = i % slice;
            dst[i] = part[(size_t) t * H * W + (size_t) j * slice + rem];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive, 1);
        if (v == (int) gridDim.x - 1) {
            __threadfence_system();
            for (int j = 0; j < n_shards; ++j)
                if (j != me) reinterpret_cast<volatile int*>(P.flag[j])[me] = step;
            *arrive = 0;
            counter[0] = step;
        }
    }
    if ((int) blockIdx.x >= T * Hl) return;
    if (threadIdx.x < n_shards && (int) threadIdx.x != me) {
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[threadIdx.x] < step) hy4f_spin_guard(t0);
    }
    __syncthreads();
    __threadfence_system();
    const int t = blockIdx.x / Hl, hl = blockIdx.x % Hl, h = h0 + hl;
    const float* rv = P.buf[me][par];
    auto at = [&](int j, int c) -> float {
        return j == me ? part[(size_t) t * H * W + (size_t) h * W + c]
                       : __ldcv(rv + (size_t) j * per + (size_t) t * slice + (size_t) hl * W + c);
    };
    float m = sinks_all[h];
    for (int j = 0; j < n_shards; ++j) m = fmaxf(m, at(j, kvl));
    float den = __expf(sinks_all[h] - m), f[4];
    for (int j = 0; j < n_shards; ++j) {
        const float mj = at(j, kvl);
        f[j] = mj == -INFINITY ? 0.0f : __expf(mj - m);
        den += f[j] * at(j, kvl + 1);
    }
    for (int c = threadIdx.x; c < kvl; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_shards; ++j) acc += f[j] * at(j, c);
        o[((size_t) t * Hl + hl) * kvl + c] = acc / den;
    }
}
// a[t][r] = (v_b[r] . o[t][r / vh]) * sigmoid(gate[t][r]) for T rows, v_b Q8_0 [Hl*vh][kvl] read once (warp per r)
template <int T>
__global__ void kd_vb_gate(const uint8_t* __restrict__ vb, const float* __restrict__ o, const float* __restrict__ gate,
                           int ldg, float* __restrict__ a, int kvl, int vh, int H) {
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= H * vh) return;
    const int h = r / vh, nbk = kvl / 32;
    const uint8_t* row = vb + (size_t) r * nbk * 34;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int e = lane; e < kvl; e += 32) {   // lane e: one element of each 32-block (coalesced)
        const uint8_t* blk = row + (e / 32) * 34;
        const float w = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[e % 32];
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] += w * o[((size_t) t * H + h) * kvl + e];
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) a[(size_t) t * H * vh + r] = s * sigm(gate[(size_t) t * ldg + r]);
    }
}
// qabs[t][h][j] = k_b[h*kvl + j] . q_nope[t][h] for T rows, k_b Q8_0 [Ha*kvl][nope] read once (warp per row)
template <int T>
__global__ void kd_absorb(const uint8_t* __restrict__ kb, const float* __restrict__ q, int ldq, int qs,
                          float* __restrict__ qabs, int nope, int kvl, int H) {
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= H * kvl) return;
    const int h = r / kvl, nbk = nope / 32;
    const uint8_t* row = kb + (size_t) r * nbk * 34;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int e = lane; e < nope; e += 32) {
        const uint8_t* blk = row + (e / 32) * 34;
        const float w = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[e % 32];
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] += w * q[(size_t) t * ldq + (size_t) h * qs + e];
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) qabs[((size_t) t * H + h) * kvl + (r % kvl)] = s;
    }
}


// ---- exchanges as one push kernel per peer (several blocks, one peer at a time: a single SM's remote stores were
// the all-reduce's cost; several SMs storing to different peers at once is this board's slow pattern) + a post kernel.
// counter[0] = the last finished step (the post kernel's last block bumps it); arrive[0..3] the push blocks per peer,
// arrive[4] the post blocks.
// push of x1 (+ x2) [tn] (mode 0) or of the head slice j of part [T][H][W] (mode 1: slice [T][Hl][W]) to peer j
__global__ void kd_push1(const float* __restrict__ x1, const float* __restrict__ x2, int tn, int mode, int T, int H, int W,
                         int Hl, Peers P, int me, int j, const int* counter, int* arrive) {
    const int step = counter[0] + 1, par = step & 1;
    float* dst = P.buf[j][par] + (size_t) me * tn;
    if (mode == 0) {
        const float4* a4 = reinterpret_cast<const float4*>(x1);
        const float4* c4 = reinterpret_cast<const float4*>(x2);
        float4* d4 = reinterpret_cast<float4*>(dst);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tn / 4; i += gridDim.x * blockDim.x) {
            float4 v = a4[i];
            if (x2) {
                const float4 w = c4[i];
                v.x += w.x, v.y += w.y, v.z += w.z, v.w += w.w;
            }
            d4[i] = v;
        }
    } else {
        const int slice = Hl * W;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tn; i += gridDim.x * blockDim.x) {
            const int t = i / slice, rem = i % slice;
            dst[i] = x1[(size_t) t * H * W + (size_t) j * slice + rem];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive + j, 1);
        if (v == (int) gridDim.x - 1) {
            __threadfence_system();
            reinterpret_cast<volatile int*>(P.flag[j])[me] = step;
            arrive[j] = 0;
        }
    }
}
__device__ __forceinline__ void kd_wait_peers(const Peers& P, int me, int n_shards, int step) {
    if (threadIdx.x < n_shards && (int) threadIdx.x != me) {
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[threadIdx.x] < step) hy4f_spin_guard(t0);
    }
    __syncthreads();
    __threadfence_system();
}
__device__ __forceinline__ void kd_post_done(int* counter, int* arrive, int step) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive, 1);
        if (v == (int) gridDim.x - 1) {
            arrive[0] = 0;
            counter[0] = step;
        }
    }
}
// all-reduce tail: block t = row t: the sum in shard order (local partial at me), y = RMS * w, X[t][s] += post y
__global__ void kd_ar_tail(const float* __restrict__ x1, const float* __restrict__ x2, int T, int n, Peers P, int me,
                           int n_shards, int* counter, int* arrive, const float* __restrict__ nw, float eps, float* X,
                           const float* __restrict__ post, int hc) {
    const int step = counter[0] + 1, tn = T * n, t = blockIdx.x;
    kd_wait_peers(P, me, n_shards, step);
    const float* rv = P.buf[me][step & 1];
    __shared__ float ys[4096];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = 0.0f;
        for (int j = 0; j < n_shards; ++j)
            v += j == me ? x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f)
                         : __ldcv(rv + (size_t) j * tn + (size_t) t * n + i);
        ys[i] = v;
        sq += v * v;
    }
    const float r = rsqrtf(block_sum(sq) / n + eps);
    float* Xt = X + (size_t) t * hc * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = ys[i] * r * nw[i];
        for (int s = 0; s < hc; ++s) Xt[(size_t) s * n + i] += post[t * 8 + s] * y;
    }
    kd_post_done(counter, arrive, step);
}
// combine tail: block b = (t, hl)
__global__ void kd_cb_tail(const float* __restrict__ part, int T, int H, int kvl, int W, int h0, int Hl,
                           const float* __restrict__ sinks_all, Peers P, int me, int n_shards, int* counter, int* arrive,
                           float* __restrict__ o) {
    const int step = counter[0] + 1, slice = Hl * W, per = T * slice;
    kd_wait_peers(P, me, n_shards, step);
    const int t = blockIdx.x / Hl, hl = blockIdx.x % Hl, h = h0 + hl;
    const float* rv = P.buf[me][step & 1];
    auto at = [&](int j, int c) -> float {
        return j == me ? part[(size_t) t * H * W + (size_t) h * W + c]
                       : __ldcv(rv + (size_t) j * per + (size_t) t * slice + (size_t) hl * W + c);
    };
    float m = sinks_all[h];
    for (int j = 0; j < n_shards; ++j) m = fmaxf(m, at(j, kvl));
    float den = __expf(sinks_all[h] - m), f[4];
    for (int j = 0; j < n_shards; ++j) {
        const float mj = at(j, kvl);
        f[j] = mj == -INFINITY ? 0.0f : __expf(mj - m);
        den += f[j] * at(j, kvl + 1);
    }
    for (int c = threadIdx.x; c < kvl; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n_shards; ++j) acc += f[j] * at(j, c);
        o[((size_t) t * Hl + hl) * kvl + c] = acc / den;
    }
    kd_post_done(counter, arrive, step);
}


// ---- the k-pool selection for a few tokens (decode) over gridDim.x blocks per token instead of one: the same set of
// cells as kb_select_kpool (top k groups by hy4f_key, ties taken in index order, then the current group's tail), the
// groups in index order.  st [T][4] = prefix, mask, kk (ties left), -; hist [T][2048]; bc [T][NB][2] = (gt, eq) per block.
constexpr int SEL_NB = 32;
__global__ void kd_sel_init(const int* pos0p, int kpool, int top_pools, unsigned* st, unsigned* hist) {
    const int t = blockIdx.x, pos = *pos0p + t, n_pool = pos / kpool;
    if (threadIdx.x == 0) st[t * 4] = 0, st[t * 4 + 1] = 0, st[t * 4 + 2] = (unsigned) min(top_pools, n_pool), st[t * 4 + 3] = 0;
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) hist[(size_t) t * 2048 + i] = 0;
}
__global__ void kd_sel_hist(const float* __restrict__ score, int ld, const int* pos0p, int kpool, int top_k,
                            const unsigned* st, unsigned* hist, int sh, int nbits) {
    const int t = blockIdx.y, pos = *pos0p + t;
    if (pos + 1 <= top_k) return;
    const int n_pool = pos / kpool, nbins = 1 << nbits;
    __shared__ unsigned h[2048];
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const unsigned prefix = st[t * 4], mask = st[t * 4 + 1];
    const int per = (n_pool + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n_pool, i0 + per);
    const float* sc = score + (size_t) t * ld;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const unsigned key = hy4f_key(sc[i]);
        if ((key & mask) == prefix) atomicAdd(&h[(key >> sh) & (nbins - 1)], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        if (h[i]) atomicAdd(&hist[(size_t) t * 2048 + i], h[i]);
}
__global__ void kd_sel_resolve(const int* pos0p, int top_k, unsigned* st, unsigned* hist, int sh, int nbits) {
    const int t = blockIdx.x, pos = *pos0p + t, tid = threadIdx.x;
    if (pos + 1 <= top_k) return;
    const int nbins = 1 << nbits;
    using Scan = cub::BlockScan<int, 1024>;
    __shared__ typename Scan::TempStorage tmp;
    unsigned* hh = hist + (size_t) t * 2048;
    const unsigned prefix = st[t * 4], mask = st[t * 4 + 1];
    const int kk = (int) st[t * 4 + 2];
    const int r0 = nbins - 1 - 2 * tid, r1 = r0 - 1;
    const int c0 = r0 >= 0 ? (int) hh[r0] : 0, c1 = r1 >= 0 ? (int) hh[r1] : 0;
    int above, total;
    Scan(tmp).ExclusiveSum(c0 + c1, above, total);
    __syncthreads();
    if (r0 >= 0 && above < kk && kk <= above + c0) {
        st[t * 4] = prefix | ((unsigned) r0 << sh);
        st[t * 4 + 1] = mask | ((unsigned) (nbins - 1) << sh);
        st[t * 4 + 2] = (unsigned) (kk - above);
    } else if (r1 >= 0 && above + c0 < kk && kk <= above + c0 + c1) {
        st[t * 4] = prefix | ((unsigned) r1 << sh);
        st[t * 4 + 1] = mask | ((unsigned) (nbins - 1) << sh);
        st[t * 4 + 2] = (unsigned) (kk - above - c0);
    }
    for (int i = tid; i < 2048; i += blockDim.x) hh[i] = 0;
}
__global__ void kd_sel_count(const float* __restrict__ score, int ld, const int* pos0p, int kpool, int top_k,
                             const unsigned* st, int* bc) {
    const int t = blockIdx.y, pos = *pos0p + t;
    if (pos + 1 <= top_k) return;
    const int n_pool = pos / kpool;
    const unsigned thr = st[t * 4];
    const int per = (n_pool + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n_pool, i0 + per);
    const float* sc = score + (size_t) t * ld;
    int gt = 0, eq = 0;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        const unsigned key = hy4f_key(sc[i]);
        gt += key > thr;
        eq += key == thr;
    }
    gt = (int) block_sum((float) gt);
    eq = (int) block_sum((float) eq);
    if (threadIdx.x == 0) bc[((size_t) t * gridDim.x + blockIdx.x) * 2] = gt, bc[((size_t) t * gridDim.x + blockIdx.x) * 2 + 1] = eq;
}
__global__ void kd_sel_write(const float* __restrict__ score, int ld, const int* pos0p, int kpool, int top_k, int top_pools,
                             int n_cap, const unsigned* st, const int* bc, int* __restrict__ cells_all) {
    const int t = blockIdx.y, pos = *pos0p + t, tid = threadIdx.x;
    int* cells = cells_all + (size_t) t * n_cap;
    if (pos + 1 <= top_k) {
        if (blockIdx.x == 0)
            for (int i = tid; i < n_cap; i += blockDim.x) cells[i] = i <= pos ? i : -1;
        return;
    }
    const int n_pool = pos / kpool, k = min(top_pools, n_pool), gcur = n_pool;
    const unsigned thr = st[t * 4];
    const int ties = (int) st[t * 4 + 2];
    int base_gt = 0, base_eq = 0, n_gt = 0;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const int g = bc[((size_t) t * gridDim.x + b) * 2], e = bc[((size_t) t * gridDim.x + b) * 2 + 1];
        if (b < (int) blockIdx.x) base_gt += g, base_eq += e;
        n_gt += g;
    }
    const int per = (n_pool + gridDim.x - 1) / gridDim.x, i0 = blockIdx.x * per, i1 = min(n_pool, i0 + per);
    const float* sc = score + (size_t) t * ld;
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage tmp;
    int run_gt = 0, run_eq = 0;
    for (int c0 = i0; c0 < i1; c0 += 256) {
        const int i = c0 + tid;
        const unsigned key = i < i1 ? hy4f_key(sc[i]) : 0u;
        const int fg = (i < i1 && key > thr) ? 1 : 0, fe = (i < i1 && key == thr) ? 1 : 0;
        int rg, tg, re, te;
        Scan(tmp).ExclusiveSum(fg, rg, tg);
        __syncthreads();
        Scan(tmp).ExclusiveSum(fe, re, te);
        __syncthreads();
        if (fg) {
            const int slot = base_gt + run_gt + rg;
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
        if (fe) {
            const int r = base_eq + run_eq + re;
            if (r < ties) {
                const int slot = n_gt + r;
                for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
            }
        }
        run_gt += tg;
        run_eq += te;
    }
    if (blockIdx.x == 0) {
        const int n_top = k * kpool;
        for (int i = n_top + tid; i < n_cap; i += blockDim.x) {
            const int c = gcur * kpool + (i - n_top);
            cells[i] = c <= pos ? c : -1;
        }
    }
}

// the sigmoid router: top-k by (sigmoid(l) + bias) descending, index ascending (router_sigmoid's rank order) through a
// bitonic sort in shared memory instead of an O(n^2) rank count; the weights from the unbiased p summed in double on
// one thread in rank order (router_sigmoid's arithmetic).  NE = n_expert, a power of two <= 1024; blockIdx.x = token,
// NE threads.
template <int NE>
__global__ void __launch_bounds__(NE) kd_router(const float* __restrict__ logits, const float* __restrict__ bias, int k,
                                                float w_scale, int* __restrict__ ids, float* __restrict__ weights) {
    __shared__ float sp[NE], key[NE];
    __shared__ int idx[NE];
    const int t = blockIdx.x, i = threadIdx.x;
    {
        const float x = logits[(size_t) t * NE + i];
        const float p = 1.0f / (1.0f + expf(-x));
        sp[i] = p;
        key[i] = bias ? p + bias[i] : p;
        idx[i] = i;
    }
    __syncthreads();
    for (int size = 2; size <= NE; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int j = i ^ stride;
            if (j > i) {
                const float ka = key[i], kb = key[j];
                const int ia = idx[i], ib = idx[j];
                const bool before = ka > kb || (ka == kb && ia < ib);
                const bool up = (i & size) == 0;
                if (before != up) {
                    key[i] = kb, key[j] = ka;
                    idx[i] = ib, idx[j] = ia;
                }
            }
            __syncthreads();
        }
    if (i < k) ids[(size_t) t * k + i] = idx[i];
    if (i == 0) {
        double sum = 0.0;
        for (int r = 0; r < k; ++r) sum += (double) sp[idx[r]];
        const float inv = (float) (1.0 / fmax(sum, 6.103515625e-5));
        for (int r = 0; r < k; ++r) weights[(size_t) t * k + r] = (float) ((double) sp[idx[r]] * inv * w_scale);
    }
}

}  // namespace kern
}  // namespace strata::hy4f
