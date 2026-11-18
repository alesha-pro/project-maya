// src/hy4f/hy4f_dkern.cuh - maya-hy4f: the multi-token decode step's own kernels (T <= 4 tokens: the speculative
// verify and the MTP draft layer).  Exchanges keep the decode's device step counters (graph-capturable).
#pragma once

#include "hy4f_kern.cuh"

#include <climits>

namespace strata::hy4f {
namespace kern {

__device__ __forceinline__ float ar_own(const PeersD& P, float v) { return P.bf ? __bfloat162float(__float2bfloat16_rn(v)) : v; }
// P.bf == 2: the LL form (NCCL's low-latency protocol): every 8-byte word = 2 BF16 values + the step, stored by the
// sender in one 8-byte store with no fence and no flag; the receiver polls the word until it carries its step
__device__ __forceinline__ float ar_peer(const PeersD& P, int step, const float* rv, size_t idx) {
    if (P.bf == 2) {
        const volatile uint2* w = reinterpret_cast<const volatile uint2*>(rv) + idx / 2;
        uint2 u;
        unsigned long long t0 = 0;
        for (;;) {
            asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(u.x), "=r"(u.y) : "l"(w));
            if ((int) u.y == step) break;
            if (!t0) t0 = hy4f_now();
            hy4f_spin_guard(t0);
        }
        const unsigned short h = (idx & 1) ? (unsigned short) (u.x >> 16) : (unsigned short) (u.x & 0xffffu);
        return __bfloat162float(__ushort_as_bfloat16(h));
    }
    if (P.bf) return __bfloat162float(__ushort_as_bfloat16(__ldcv(reinterpret_cast<const unsigned short*>(rv) + idx)));
    return __ldcv(rv + idx);
}


// all-reduce of T rows (x1 + x2) [T][n], then per row y = RMS(sum) * w and X[t][s] += post[t][s] * y (X = R + post y
// with R = X; hc = 1 and post = 1 for the MTP's plain residual).  One block of 1024 threads, dynamic smem T*n floats.
__global__ void kd_ar_post(const float* x1, const float* x2, int T, int n, PeersD P, int me, int n_shards, int* counter,
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
                           PeersD P, int me, int n_shards, int* counter, float* o) {
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
__global__ void kd_ar2(const float* __restrict__ x1, const float* __restrict__ x2, int T, int n, PeersD P, int me,
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
            v += j == me ? ar_own(P, x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f))
                         : ar_peer(P, step, rv, (size_t) j * tn + (size_t) t * n + i);
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
                            const float* __restrict__ sinks_all, PeersD P, int me, int n_shards, int* counter, int* arrive,
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
                         int Hl, PeersD P, int me, int j, const int* counter, int* arrive) {
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
__global__ void kd_ar_tail(const float* __restrict__ x1, const float* __restrict__ x2, int T, int n, PeersD P, int me,
                           int n_shards, int* counter, int* arrive, const float* __restrict__ nw, float eps, float* X,
                           const float* __restrict__ post, int hc) {
    const int step = counter[0] + 1, tn = T * n, t = blockIdx.x;
    if (P.bf != 2) kd_wait_peers(P, me, n_shards, step);
    const float* rv = P.buf[me][step & 1];
    __shared__ float ys[4096];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = 0.0f;
        for (int j = 0; j < n_shards; ++j)
            v += j == me ? ar_own(P, x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f))
                         : ar_peer(P, step, rv, (size_t) j * tn + (size_t) t * n + i);
        ys[i] = v;
        sq += v * v;
    }
    const float r = nw ? rsqrtf(block_sum(sq) / n + eps) : 1.0f;   // (nw null: a plain residual add)
    float* Xt = X + (size_t) t * hc * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = nw ? ys[i] * r * nw[i] : ys[i];
        for (int s = 0; s < hc; ++s) Xt[(size_t) s * n + i] += post[t * 8 + s] * y;
    }
    kd_post_done(counter, arrive, step);
}
// Stream-ordered after the peer DMA operation. The following sublayer's
// all-reduce orders every receiver before this receive slot is reused.
__global__ void kd_cb_publish(int* flag, int me, const int* counter) {
    if (threadIdx.x == 0) {
        const int step = *counter + 1;
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag+me), "r"(step) : "memory");
    }
}
// combine tail: block b = (t, hl)
__global__ void kd_cb_tail(const float* __restrict__ part, int T, int H, int kvl, int W, int h0, int Hl,
                           const float* __restrict__ sinks_all, PeersD P, int me, int n_shards, int* counter, int* arrive,
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
// Before sparse selection starts, every DSA layer sees the same consecutive prefix.
__global__ void kd_dense_cells(const int* pos0p, int cap, int* out) {
    const int t = blockIdx.y, pos = *pos0p + t;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cap) out[(size_t) t * cap + i] = i <= pos ? i : -1;
}

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


// the distinct local experts of a step and their slots (eloc [n]: local id or -1; n <= 64): groups in the order of
// their first slot, a group's slots in slot order.  One block of 64 threads.  ng[0] = group count.
__global__ void kd_moe_groups(const int* __restrict__ eloc, int n, int* ng, int* gexp, int* goff, int* gslot) {
    __shared__ int id[128], leader[128], cnt[128], gidx[128];
    const int i = threadIdx.x;
    id[i] = i < n ? eloc[i] : -1;
    __syncthreads();
    int ld = -1, rank = 0;
    if (id[i] >= 0) {
        for (int j = 0; j < i; ++j)
            if (id[j] == id[i]) {
                if (ld < 0) ld = j;
                ++rank;
            }
        if (ld < 0) ld = i;
    }
    leader[i] = (id[i] >= 0 && ld == i) ? 1 : 0;
    int c = 0;
    if (leader[i])
        for (int j = i; j < n; ++j) c += id[j] == id[i];
    cnt[i] = c;
    __syncthreads();
    int g = 0;
    for (int j = 0; j < i; ++j) g += leader[j];
    gidx[i] = g;
    __syncthreads();
    if (leader[i]) {
        int off = 0;
        for (int j = 0; j < i; ++j) off += cnt[j];
        gexp[g] = id[i];
        goff[g] = off;
    }
    if (i == blockDim.x - 1) {
        int tot = 0, ngr = 0;
        for (int j = 0; j < (int) blockDim.x; ++j) tot += cnt[j], ngr += leader[j];
        ng[0] = ngr;
        goff[ngr] = tot;
    }
    __syncthreads();
    if (id[i] >= 0) {
        int off = 0;
        for (int j = 0; j < ld; ++j) off += cnt[j];
        gslot[off + rank] = i;
    }
}


// all peers in one launch: blocks [r*nbp, (r+1)*nbp) push x1 (+ x2) [tn] to peer (me + 1 + r) % n_shards (each peer its
// own blocks, written concurrently); the last block per peer signals it.  mode 1: the head slice like kd_push1.
__global__ void kd_push_all(const float* __restrict__ x1, const float* __restrict__ x2, int tn, int mode, int T, int H,
                            int W, int Hl, PeersD P, int me, int n_shards, int nbp, const int* counter, int* arrive) {
    const int r = blockIdx.x / nbp, bb = blockIdx.x % nbp, j = (me + 1 + r) % n_shards;
    const int step = counter[0] + 1, par = step & 1;
    float* dst = P.buf[j][par] + (size_t) me * tn;
    if (mode == 0) {
        const float4* a4 = reinterpret_cast<const float4*>(x1);
        const float4* c4 = reinterpret_cast<const float4*>(x2);
        float4* d4 = reinterpret_cast<float4*>(dst);
        for (int i = bb * blockDim.x + threadIdx.x; i < tn / 4; i += nbp * blockDim.x) {
            float4 v = a4[i];
            if (x2) {
                const float4 w = c4[i];
                v.x += w.x, v.y += w.y, v.z += w.z, v.w += w.w;
            }
            d4[i] = v;
        }
    } else {
        const int slice = Hl * W;
        for (int i = bb * blockDim.x + threadIdx.x; i < tn; i += nbp * blockDim.x) {
            const int t = i / slice, rem = i % slice;
            dst[i] = x1[(size_t) t * H * W + (size_t) j * slice + rem];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive + j, 1);
        if (v == nbp - 1) {
            __threadfence_system();
            reinterpret_cast<volatile int*>(P.flag[j])[me] = step;
            arrive[j] = 0;
        }
    }
}


// the peers one after another inside one launch (a single launch per exchange instead of one per peer; still one peer
// at a time per block): each block writes its slice to peer me+1, fences, counts in; then me+2, ...
__global__ void kd_push_seq(const float* __restrict__ x1, const float* __restrict__ x2, int tn, int mode, int T, int H,
                            int W, int Hl, PeersD P, int me, int n_shards, const int* counter, int* arrive) {
    const int step = counter[0] + 1, par = step & 1;
    for (int r = 1; r < n_shards; ++r) {
        const int j = (me + r) % n_shards;
        float* dst = P.buf[j][par] + (size_t) me * tn;
        if (mode == 0 && P.bf == 2) {   // (LL: no fence, no flag)
            uint2* d8 = reinterpret_cast<uint2*>(P.buf[j][par]) + (size_t) me * (tn / 2);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tn / 2; i += gridDim.x * blockDim.x) {
                float a = x1[2 * i], b = x1[2 * i + 1];
                if (x2) a += x2[2 * i], b += x2[2 * i + 1];
                const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
                const unsigned bits = *reinterpret_cast<const unsigned*>(&v);
                asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" :: "l"(d8 + i), "r"(bits), "r"((unsigned) step) : "memory");
            }
            continue;
        }
        if (mode == 0 && P.bf) {
            const float4* a4 = reinterpret_cast<const float4*>(x1);
            const float4* c4 = reinterpret_cast<const float4*>(x2);
            uint2* d8 = reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(P.buf[j][par]) + (size_t) me * tn);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tn / 4; i += gridDim.x * blockDim.x) {
                float4 v = a4[i];
                if (x2) {
                    const float4 w = c4[i];
                    v.x += w.x, v.y += w.y, v.z += w.z, v.w += w.w;
                }
                const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
                d8[i] = make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
            }
        } else if (mode == 0) {
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
}

// kd_push_seq with one system fence: the data to every peer, one fence, then the flags (fewer PCIe round trips)
__global__ void kd_push_seq1(const float* __restrict__ x1, const float* __restrict__ x2, int tn, int mode, int T, int H,
                             int W, int Hl, PeersD P, int me, int n_shards, const int* counter, int* arrive) {
    const int step = counter[0] + 1, par = step & 1;
    for (int r = 1; r < n_shards; ++r) {
        const int j = (me + r) % n_shards;
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
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive, 1);
        if (v == (int) gridDim.x - 1) {
            __threadfence_system();
            for (int r = 1; r < n_shards; ++r) reinterpret_cast<volatile int*>(P.flag[(me + r) % n_shards])[me] = step;
            arrive[0] = 0;
        }
    }
}


// the all-reduce tail fused with the next sublayer's input (HC pre-mix + RMS norm + Q8_1): block t = row t.  After
// X[t][s] += post[t][s] * RMS(sum) * w (the tail), the next mix m_r = fn[r] . X[t] / rms(X[t]) gives pre / post (post is
// overwritten with the next sublayer's), x = RMS(sum_s pre[s] X[t][s]) * nw_next -> xo [T][E] fp32 and xq Q8_1.
// E == 3 * 1024, hc <= 4, n_mix <= 8 (the Hy4 geometry; the caller checks).
struct HcNext {
    const float *fn = nullptr, *scale = nullptr, *base = nullptr, *nw = nullptr;
    int n_mix = 0;
    float mag = 0.0f;
};
__global__ void __launch_bounds__(1024) kd_ar_tail_hc(const float* __restrict__ x1, const float* __restrict__ x2, int T,
                                                      int n, PeersD P, int me, int n_shards, int* counter, int* arrive,
                                                      const float* __restrict__ nw, float eps, float* X, float* post, int hc,
                                                      HcNext nx, float hc_eps, float* __restrict__ xo, Q81* __restrict__ xq) {
    constexpr int PER = 3;
    const int step = counter[0] + 1, tn = T * n, t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    if (P.bf != 2) kd_wait_peers(P, me, n_shards, step);
    const float* rv = P.buf[me][step & 1];
    float ysum[PER], sq = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * 1024;
        float v = 0.0f;
        for (int j = 0; j < n_shards; ++j)
            v += j == me ? ar_own(P, x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f))
                         : ar_peer(P, step, rv, (size_t) j * tn + (size_t) t * n + i);
        ysum[k] = v;
        sq += v * v;
    }
    const float r = rsqrtf(block_sum(sq) / n + eps);
    float pv[4];
    for (int s2 = 0; s2 < 4; ++s2) pv[s2] = s2 < hc ? post[t * 8 + s2] : 0.0f;
    float Xv[4][PER];
    float* Xt = X + (size_t) t * hc * n;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * 1024;
        const float y = ysum[k] * r * nw[i];
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2)
            if (s2 < hc) {
                const float v = Xt[(size_t) s2 * n + i] + pv[s2] * y;
                Xt[(size_t) s2 * n + i] = v;
                Xv[s2][k] = v;
            }
    }
    // the next mix: 8 dots over the hc * n state + its square sum
    float acc[9];
#pragma unroll
    for (int q = 0; q < 9; ++q) acc[q] = 0.0f;
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2)
        if (s2 < hc)
#pragma unroll
            for (int k = 0; k < PER; ++k) {
                const int i = s2 * n + tid + k * 1024;
                const float xv = Xv[s2][k];
                acc[8] += xv * xv;
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (q < nx.n_mix) acc[q] += nx.fn[(size_t) q * hc * n + i] * xv;
            }
    __shared__ float red[9][32], res[9], pre[4];
#pragma unroll
    for (int q = 0; q < 9; ++q) {
        const float v = warp_sum(acc[q]);
        if (lane == 0) red[q][wid] = v;
    }
    __syncthreads();
    if (tid < 9) {
        float v = 0.0f;
        for (int w2 = 0; w2 < 32; ++w2) v += red[tid][w2];
        res[tid] = v;
    }
    __syncthreads();
    if (tid == 0) {
        const float inv = rsqrtf(res[8] / (float) (n * hc) + eps);
        for (int q = 0; q < nx.n_mix; ++q) {
            const float m = res[q] * inv;
            if (q < hc) pre[q] = sigm(m * nx.scale[0] + nx.base[q]) + hc_eps;
            else post[t * 8 + q - hc] = sigm(m * nx.scale[1] + nx.base[q]) * nx.mag + hc_eps;
        }
    }
    __syncthreads();
    float cur[PER], sq2 = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        float v = 0.0f;
#pragma unroll
        for (int s2 = 0; s2 < 4; ++s2)
            if (s2 < hc) v += pre[s2] * Xv[s2][k];
        cur[k] = v;
        sq2 += v * v;
    }
    const float r2 = rsqrtf(block_sum(sq2) / n + eps);
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = tid + k * 1024;
        const float v = cur[k] * r2 * nx.nw[i];
        xo[(size_t) t * n + i] = v;
        q81_store(xq + ((size_t) t * n + i) / 32, v);
    }
    kd_post_done(counter, arrive, step);
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


// No normal rounding: each halfword is an original FP32 router coefficient's
// high bits, and the loader proved the low bits were zero.
__global__ void kd_router_gemv(const uint16_t* __restrict__ packed, const float* __restrict__ x,
                               float* __restrict__ out, int cols, int rows) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* a = x + size_t(blockIdx.y) * cols + lane;
    const uint4* w = reinterpret_cast<const uint4*>(packed + size_t(r) * cols) + lane;
    float acc = 0.0f;
    for (int c = 0; c < cols; c += 256) {
        const uint4 v = w[c / 8];
        const unsigned four[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += __uint_as_float(four[j] << 16) * a[c + 64*j];
            acc += __uint_as_float(four[j] & 0xffff0000u) * a[c + 64*j + 32];
        }
    }
    acc = warp_sum(acc);
    if (!lane) out[size_t(blockIdx.y) * rows + r] = acc;
}
__global__ void kb_router_unpack(const uint16_t* __restrict__ packed, float* __restrict__ w, size_t size) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        const size_t orig = (i & ~size_t(255)) + ((i & 7) << 5) + ((i >> 3) & 31);
        w[orig] = __uint_as_float(unsigned(packed[i]) << 16);
    }
}
// Exact top-16 router. Merge sixteen warp-local runs four times, keeping only
// their best sixteen entries after each merge. The low bits break ties by ID.
__device__ __forceinline__ unsigned long long router_winner(unsigned long long v, int dist, bool want_max) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, v, dist);
    return (other > v) == want_max ? other : v;
}
__global__ void kd_router16(const float* logits, const float* bias, float scale, int e0, int e1,
                            int* local, float* weights) {
    __shared__ float prob[512];
    __shared__ unsigned long long run[2][256];
    const int i = threadIdx.x, warp = i >> 5, lane = i & 31, t = blockIdx.x;
    const float p = 1.0f / (1.0f + expf(-logits[t * 512 + i]));
    prob[i] = p;
    const float score = bias ? p + bias[i] : p;
    const unsigned bits = __float_as_uint(score == 0.0f ? 0.0f : score);
    const unsigned ordered = bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
    unsigned long long v = (static_cast<unsigned long long>(ordered) << 32) | (511 - i);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
#pragma unroll
        for (int d = size >> 1; d; d >>= 1)
            v = router_winner(v, d, ((lane & size) == 0) == ((lane & d) == 0));
    if (lane < 16) run[0][warp * 16 + lane] = v;
#pragma unroll
    for (int stage = 0; stage < 4; ++stage) {
        __syncthreads();
        if (warp < (8 >> stage)) {
            // Lower half is descending, upper half ascending, hence bitonic.
            v = run[stage & 1][warp * 32 + (lane < 16 ? lane : (47 - lane))];
#pragma unroll
            for (int d = 16; d; d >>= 1) v = router_winner(v, d, (lane & d) == 0);
            if (lane < 16) run[(stage + 1) & 1][warp * 16 + lane] = v;
        }
    }
    if (warp == 0) {
        const int id = 511 - static_cast<int>(static_cast<unsigned>(v));
        const float selected = prob[id];
        double sum = 0.0;
#pragma unroll
        for (int k = 0; k < 16; ++k) sum += (double)__shfl_sync(0xffffffffu, selected, k);
        if (lane < 16) {
            const float norm = (float)(1.0 / fmax(sum, 6.103515625e-5));
            weights[t * 16 + lane] = (float)((double)selected * norm * scale);
            local[t * 16 + lane] = id >= e0 && id < e1 ? id - e0 : -1;
        }
    }
}

// Both latent and shared FFN projections consume Q8_1, so write the
// quantized rows directly. The summation is in the original slot order.
__global__ void kd_ffn_q8(const float* dn, const float* weights, const int* local, int K, int latent,
                          const float* shared_gu, int ld, int fl, Q81* routed_q, Q81* shared_q) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < latent) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int r = t * K + k;
            if (local[r] >= 0) sum += weights[r] * dn[(size_t)r * latent + i];
        }
        q81_store(routed_q + ((size_t)t * latent + i) / 32, sum);
    } else if (i < latent + fl) {
        const int j = i - latent;
        const float* gu = shared_gu + (size_t)t * ld;
        q81_store(shared_q + ((size_t)t * fl + j) / 32, silu(gu[j]) * gu[fl + j]);
    }
}

// ---- DFly2 drafter (hy4f_dfly.inc).  io = the step's IoT {pos = anchor, n = source row, toks}.
// the draft block's cell lists: row t = the target's top-k cells of the source row (positions < anchor, -1 unused) +
// the block's own positions anchor .. anchor + t (causal inside the block).  blockIdx.x = t.
__global__ void kd_df_cells(const int* __restrict__ snap, int n_cap, const int* __restrict__ io, int* __restrict__ dc, int ncd) {
    const int anchor = io[0], row = io[1], t = blockIdx.x;
    const int* src = snap + (size_t) row * n_cap;
    for (int i = threadIdx.x; i < ncd; i += blockDim.x) {
        int v;
        if (i < n_cap) {
            v = src[i];
            if (v >= anchor) v = -1;
        } else {
            v = i - n_cap <= t ? anchor + (i - n_cap) : -1;
        }
        dc[(size_t) t * ncd + i] = v;
    }
}
// cat = [RMS(a) * wa, RMS(b) * wb] (one block of 1024 threads); b = b0 + io[1] * n when io (the source row's hidden)
__global__ void kd_df_rms2(const float* __restrict__ a, const float* __restrict__ wa, const float* __restrict__ b0,
                           const int* __restrict__ io, const float* __restrict__ wb, int n, float eps, float* __restrict__ cat) {
    const float* b = b0 + (io ? (size_t) io[1] * n : 0);
    float sa = 0.0f, sb = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sa += a[i] * a[i], sb += b[i] * b[i];
    const float ra = rsqrtf(block_sum(sa) / n + eps);
    __syncthreads();
    const float rb = rsqrtf(block_sum(sb) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) cat[i] = a[i] * ra * wa[i], cat[n + i] = b[i] * rb * wb[i];
}
// the block's input rows: mr[0] = emb(bonus) + gate * eh_proj(...), mr[t >= 1] = emb(mask)
__global__ void kd_df_init(const float* __restrict__ me, const float* __restrict__ fz, const float* __restrict__ gate, int n,
                           int T, float* __restrict__ mr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * n; i += gridDim.x * blockDim.x)
        mr[i] = me[i] + (i < n ? gate[i] * fz[i] : 0.0f);
}
__global__ void kd_add(const float* __restrict__ a, const float* __restrict__ b, int n, float* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}


// ---- the fused all-reduce tail in two wide kernels (kd_ar_tail_hc's math over 12 x T blocks instead of T blocks).
// Every reduction the tail needs is a sum over the row of terms known before the norm r is: with y' = ysum * nw,
// X_s the old streams, p_s = post_s:  r = rsqrt(sum ysum^2 / n + eps);  fn_q . X_new = A_q + r B_q (A_q = fn_q . X,
// B_q = sum_s p_s fn_qs . y');  |X_new|^2 = sum_s G_ss + 2 r p_s H_s + r^2 p_s^2 K (G = X_s . X_s', H_s = X_s . y',
// K = y' . y');  cur = sum_s pre_s X_new_s = sum_s pre_s X_s + r c y' (c = sum_s pre_s p_s) so |cur|^2 = sum pre pre' G +
// 2 r c sum pre_s H_s + r^2 c^2 K.  kd_art_a: wait for the peers, ysum (scratch Y), the 32 partial sums per block;
// kd_art_b: the totals (fixed order), the scalars, X_new, cur, x and its Q8_1.  hc <= 4, n = gridDim.x * 256.
__device__ __forceinline__ int art_g(int s, int s2) {   // index of G[s][s2] (s <= s2) among the 10
    return s == 0 ? s2 : s == 1 ? 3 + s2 : s == 2 ? 5 + s2 : 9;
}
// The existing X-dependent HC partials can run alongside the sublayer.
__global__ void kd_art_prepare(const float* X, int n, int hc, HcNext nx, float* part) {
    const int t=blockIdx.y,b=blockIdx.x,tid=threadIdx.x,i=b*256+tid;
    float xs[4];
#pragma unroll
    for (int h=0;h<4;++h) xs[h]=h<hc ? X[((size_t)t*hc+h)*n+i] : 0.f;
    __shared__ float sums[8][19];
#pragma unroll
    for (int j=0;j<8;++j) {
        float a=0.f;
        if (j<nx.n_mix)
#pragma unroll
            for (int h=0;h<4;++h)
                if (h<hc) a+=nx.fn[((size_t)j*hc+h)*n+i]*xs[h];
        const float w=warp_sum(a);
        if ((tid&31)==0) sums[tid/32][j]=w;
    }
#pragma unroll
    for (int h=0;h<4;++h)
#pragma unroll
        for (int j=h;j<4;++j) {
            const float w=warp_sum(xs[h]*xs[j]);
            if ((tid&31)==0) sums[tid/32][8+art_g(h,j)]=w;
        }
    __syncthreads();
    if(tid<18) {
        float v=0.f;
#pragma unroll
        for (int j=0;j<8;++j) v+=sums[j][tid];
        part[((size_t)t*gridDim.x+b)*32+(tid<8?1+tid:9+tid)]=v;
    }
}
template<bool FusedSend = false, bool Prepared = false>
__global__ void __launch_bounds__(256) kd_art_a(const float* __restrict__ x1, const float* __restrict__ x2, int T, int n,
                                                PeersD P, int me, int n_shards, int* counter, int* arrive,
                                                const float* __restrict__ nw, const float* __restrict__ X,
                                                const float* __restrict__ post, int hc, HcNext nx, float* __restrict__ Y,
                                                float* __restrict__ part, float* __restrict__ ppost) {
    const int step = counter[0] + 1, tn = T * n, t = blockIdx.y, b = blockIdx.x, nbk = gridDim.x - int(FusedSend);
    if constexpr (FusedSend) {
        // One dedicated CTA publishes the words while the other CTAs consume
        // ready portions. A cooperative launch guarantees all blocks can run.
        if (b == nbk) {
            if (t == 0) {
                for (int r = 1; r < n_shards; ++r) {
                    const int j = (me + r) % n_shards;
                    auto* dst = reinterpret_cast<uint2*>(P.buf[j][step & 1]) + (size_t)me * (tn / 2);
                    for (int q = threadIdx.x; q < tn / 2; q += blockDim.x) {
                        float2 pair = reinterpret_cast<const float2*>(x1)[q];
                        if (x2) {
                            const float2 second = reinterpret_cast<const float2*>(x2)[q];
                            pair.x += second.x;
                            pair.y += second.y;
                        }
                        const auto packed = __floats2bfloat162_rn(pair.x, pair.y);
                        const unsigned bits = *reinterpret_cast<const unsigned*>(&packed);
                        asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};"
                                     :: "l"(dst + q), "r"(bits), "r"((unsigned)step) : "memory");
                    }
                }
            }
            return;
        }
    }
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, i = b * blockDim.x + tid;
    if (P.bf != 2) kd_wait_peers(P, me, n_shards, step);
    const float* rv = P.buf[me][step & 1];
    float v = 0.0f;
    for (int j = 0; j < n_shards; ++j)
        v += j == me ? ar_own(P, x1[(size_t) t * n + i] + (x2 ? x2[(size_t) t * n + i] : 0.0f))
                     : ar_peer(P, step, rv, (size_t) j * tn + (size_t) t * n + i);
    Y[(size_t) t * n + i] = v;
    const float yp = v * nw[i];
    float xs[4], pp[4];
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2) {
        xs[s2] = s2 < hc ? X[((size_t) t * hc + s2) * n + i] : 0.0f;
        pp[s2] = s2 < hc ? post[t * 8 + s2] : 0.0f;
    }
    float a[32];
    a[0] = v * v;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        float A = 0.0f, B = 0.0f;
        if (q < nx.n_mix)
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2)
                if (s2 < hc) {
                    const float f = nx.fn[(size_t) q * hc * n + (size_t) s2 * n + i];
                    if constexpr (!Prepared) A += f * xs[s2];
                    B += pp[s2] * (f * yp);
                }
        a[1 + q] = A;
        a[9 + q] = B;
    }
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2)
#pragma unroll
        for (int s3 = s2; s3 < 4; ++s3) if constexpr (!Prepared) a[17 + art_g(s2, s3)] = xs[s2] * xs[s3];
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2) a[27 + s2] = xs[s2] * yp;
    a[31] = yp * yp;
    __shared__ float red[8][33];
#pragma unroll
    for (int q = 0; q < 32; ++q) {
        if constexpr (Prepared) if ((q >= 1 && q <= 8) || (q >= 17 && q <= 26)) continue;
        const float w = warp_sum(a[q]);
        if (lane == 0) red[wid][q] = w;
    }
    __syncthreads();
    if (tid < 32 && (!Prepared || !((tid >= 1 && tid <= 8) || (tid >= 17 && tid <= 26)))) {
        float w = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) w += red[k][tid];
        part[((size_t) t * nbk + b) * 32 + tid] = w;
    }
    if (b == 0 && tid < 4) ppost[t * 4 + tid] = pp[tid];
    __syncthreads();
    if (!FusedSend && tid == 0) {
        const int c = atomicAdd(arrive, 1);
        if (c == (int) (gridDim.x * gridDim.y) - 1) {
            arrive[0] = 0;
            counter[0] = step;
        }
    }
}
__global__ void __launch_bounds__(256) kd_art_b(int n, const float* __restrict__ nw, float eps, float* X, float* post, int hc,
                                                HcNext nx, float hc_eps, const float* __restrict__ Y,
                                                const float* __restrict__ part, const float* __restrict__ ppost,
                                                float* __restrict__ xo, Q81* __restrict__ xq, int* advance = nullptr) {
    if (advance && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) ++*advance;
    const int t = blockIdx.y, nbk = gridDim.x, tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    __shared__ float tot[32], sc[12];   // sc: r, pre[4], c, r2, p[4]
    if (tid < 32) {
        float w = 0.0f;
        for (int k = 0; k < nbk; ++k) w += part[((size_t) t * nbk + k) * 32 + tid];
        tot[tid] = w;
    }
    __syncthreads();
    if (tid == 0) {
        const float r = rsqrtf(tot[0] / n + eps);
        float p[4], pre[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int s2 = 0; s2 < 4; ++s2) p[s2] = s2 < hc ? ppost[t * 4 + s2] : 0.0f;
        const float K = tot[31];
        float ss = 0.0f;
        for (int s2 = 0; s2 < hc; ++s2) ss += tot[17 + art_g(s2, s2)] + 2.0f * r * p[s2] * tot[27 + s2] + r * r * p[s2] * p[s2] * K;
        const float inv = rsqrtf(ss / (float) (n * hc) + eps);
        for (int q = 0; q < nx.n_mix; ++q) {
            const float m = (tot[1 + q] + r * tot[9 + q]) * inv;
            if (q < hc) pre[q] = sigm(m * nx.scale[0] + nx.base[q]) + hc_eps;
            else if (blockIdx.x == 0) post[t * 8 + q - hc] = sigm(m * nx.scale[1] + nx.base[q]) * nx.mag + hc_eps;
        }
        float c = 0.0f, gh = 0.0f, gg = 0.0f;
        for (int s2 = 0; s2 < hc; ++s2) {
            c += pre[s2] * p[s2];
            gh += pre[s2] * tot[27 + s2];
            for (int s3 = 0; s3 < hc; ++s3) gg += pre[s2] * pre[s3] * tot[17 + (s2 <= s3 ? art_g(s2, s3) : art_g(s3, s2))];
        }
        const float cc = fmaxf(gg + 2.0f * r * c * gh + r * r * c * c * K, 0.0f);
        sc[0] = r;
        for (int s2 = 0; s2 < 4; ++s2) sc[1 + s2] = pre[s2], sc[8 + s2] = p[s2];
        sc[5] = c;
        sc[6] = rsqrtf(cc / n + eps);
    }
    __syncthreads();
    const float r = sc[0], yp = Y[(size_t) t * n + i] * nw[i];
    float cur = 0.0f;
#pragma unroll
    for (int s2 = 0; s2 < 4; ++s2)
        if (s2 < hc) {
            float* xp = X + ((size_t) t * hc + s2) * n + i;
            const float xn = *xp + sc[8 + s2] * r * yp;
            *xp = xn;
            cur += sc[1 + s2] * xn;
        }
    const float v = cur * sc[6] * nx.nw[i];
    xo[(size_t) t * n + i] = v;
    q81_store(xq + ((size_t) t * n + i) / 32, v);
}

}  // namespace kern
}  // namespace strata::hy4f
