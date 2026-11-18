// src/hy4f/hy4f_bkern.cuh - maya-hy4f batched kernels (a chunk of T tokens at positions pos0 .. pos0 + T - 1): the
// prompt path and the speculative verify.  blockIdx.y (or blockIdx.x where noted) is the token.  Included by hy4f.cu.
#pragma once

#include "hy4f_kern.cuh"

#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>

namespace strata::hy4f {
namespace kern {

__global__ void kb_embed(const uint8_t* emb, size_t rb, const int* toks, const int* pos0p, float* X, float* R, int E, int hc,
                         int* hist) {
    const int pos0 = *pos0p;
    const int t = blockIdx.y, tok = toks[t];
    const uint8_t* row = emb + (size_t) tok * rb;
    float* Xt = X + (size_t) t * hc * E;
    float* Rt = R + (size_t) t * hc * E;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) {
        const uint8_t* blk = row + (i / 32) * 34;
        const float v = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
        for (int s = 0; s < hc; ++s) Xt[(size_t) s * E + i] = v, Rt[(size_t) s * E + i] = v;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) hist[pos0 + t] = tok;
}
// Engram, per token (blockIdx.x = token): hash, then rows
__global__ void kb_engram_hash(const int* hist, const int* pos0p, const int64_t* map, int64_t n_map, int64_t pad_id,
                               const uint64_t* mult, const uint64_t* vsz, const uint64_t* off, int ng, int nh, int64_t* ids) {
    const int pos0 = *pos0p;
    const int t = blockIdx.x, col = threadIdx.x, ncol = (ng - 1) * nh, pos = pos0 + t;
    if (col >= ncol) return;
    const int64_t pad = map[pad_id];
    auto cid = [&](int64_t tk) -> int64_t { return tk < 0 || tk >= n_map ? -1 : map[tk]; };
    const int o_hi = col / nh + 1;
    int64_t c = cid(hist[pos]);
    uint64_t rolling = (uint64_t) (c < 0 ? pad : c) * mult[0];
    for (int oo = 1; oo <= o_hi; ++oo) {
        c = pos - oo < 0 ? -1 : cid(hist[pos - oo]);
        rolling ^= (uint64_t) (c < 0 ? pad : c) * mult[oo];
    }
    ids[(size_t) t * ncol + col] = (int64_t) (rolling % vsz[col] + off[col]);
}
__global__ void kb_engram_rows(const uint8_t* table, const int64_t* ids, const float* lnorm, float* r, int D, int ncol,
                               float eps) {
    const int col = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float v = d < D ? fp8e4m3f(table[(size_t) ids[(size_t) t * ncol + col] * D + d]) : 0.0f;
    const float k = rsqrtf(block_sum(v * v) / D + eps);
    if (d < D) r[((size_t) t * ncol + col) * D + d] = v * k * lnorm[(size_t) col * D + d];
}
__global__ void kb_engram_mix(float* X, float* R, const float* kv, const float* kvb, const float* gw, const float* hcs,
                              const float* gamma, int E, int hc, float eps, const int* mask) {
    const int t = blockIdx.x;
    if (mask && mask[t]) return;   // (an embedding row: no n-gram, the Engram adds nothing)
    X += (size_t) t * hc * E;
    R += (size_t) t * hc * E;
    kv += (size_t) t * (hc + 1) * E;
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
// embedding rows (an image) over the chunk's token rows [t0, t0 + n): X[t][s] = row for every hc stream, the history
// entry -1 (an embedding has no token id: the Engram n-gram of later tokens sees the pad there) and the Engram mask
__global__ void kb_embed_rows(const float* rows, int t0, const int* pos0p, float* X, int E, int hc, int* hist, int* mask) {
    const int t = t0 + blockIdx.y;
    const float* r = rows + (size_t) blockIdx.y * E;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x)
        for (int s = 0; s < hc; ++s) X[((size_t) t * hc + s) * E + i] = r[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        hist[*pos0p + t] = -1;
        mask[t] = 1;
    }
}
// HC mix partials, all n_mix rows per block (blockIdx.x = split, blockIdx.y = token): part [T][8][HC_SPLIT], ss [T][HC_SPLIT]
__global__ void kb_hc_mix(const float* X, const float* fn, int n, int n_mix, float* part, float* ss) {
    const int s = blockIdx.x, t = blockIdx.y, chunk = n / HC_SPLIT, i0 = s * chunk;
    X += (size_t) t * n;
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0}, q = 0.0f;
    for (int i = i0 + threadIdx.x; i < i0 + chunk; i += blockDim.x) {
        const float x = X[i];
        q += x * x;
#pragma unroll
        for (int r = 0; r < 8; ++r)
            if (r < n_mix) d[r] += fn[(size_t) r * n + i] * x;
    }
    __shared__ float red[9][32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float v = warp_sum(d[r]);
        if (lane == 0) red[r][w] = v;
    }
    q = warp_sum(q);
    if (lane == 0) red[8][w] = q;
    __syncthreads();
    if (threadIdx.x < 9) {
        float v = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) v += red[threadIdx.x][k];
        if (threadIdx.x < 8) {
            if (threadIdx.x < n_mix) part[((size_t) t * 8 + threadIdx.x) * HC_SPLIT + s] = v;
        } else {
            ss[(size_t) t * HC_SPLIT + s] = v;
        }
    }
}
// HC apply + RMS * w per token (blockIdx.x = token, 1024 threads, E a multiple of 1024): x [T][E], post [T][8]
__global__ void kb_hc_apply_norm(const float* X, const float* part, const float* ss, const float* scale, const float* base,
                                 const float* nw, int E, int hc, int n_mix, float eps, float hc_eps, float mag, float* x,
                                 float* post) {
    const int t = blockIdx.x;
    X += (size_t) t * hc * E;
    part += (size_t) t * 8 * HC_SPLIT;
    ss += (size_t) t * HC_SPLIT;
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
                else post[(size_t) t * 8 + r - hc] = sigm(m * scale[1] + base[r]) * mag + hc_eps;
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
        x[(size_t) t * E + d] = cur[k] * r * nw[d];
    }
}
// y[t] = RMS(x[t]) * w (rows with strides), blockIdx.x = token
__global__ void kb_rms(const float* x, int ldx, const float* w, int n, float eps, float* y, int ldy) {
    const int t = blockIdx.x;
    const float* xr = x + (size_t) t * ldx;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(sq) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) t * ldy + i] = xr[i] * r * w[i];
}
// the latent rows of the chunk: RoPE(k_pe) at pos0 + t, RMS * w on c, Q8 into: mode 1 (SWA) the chunk buffer row t,
// mode 2 (DSA, context parallel) the cache slot when this shard owns the position, mode 3 the ring slot pos % rw.  blockIdx.x = token, 576 threads.
__global__ void kb_kv_prep(const float* kvin, int ld, const float* nw, int kvl, int rope, float eps, float theta,
                           uint8_t* dst_base, const int* pos0p, int mode, int n_shards, int shard, int rw) {
    const int pos0 = *pos0p;
    const int t = blockIdx.x, pos = pos0 + t, j = threadIdx.x;
    __shared__ float row[KV_ROW];
    float v = j < KV_ROW ? kvin[(size_t) t * ld + j] : 0.0f;
    const float r = rsqrtf(block_sum(j < kvl ? v * v : 0.0f) / kvl + eps);
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
    long slot;
    if (mode == 1) slot = t;
    else if (mode == 3) slot = pos % rw;   // (SWA ring, a multi-token decode step)
    else {
        if (pos % n_shards != shard) return;
        slot = pos / n_shards;
    }
    uint8_t* dst = dst_base + (size_t) slot * KV_RB;
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
// the chunk's last min(T, win) SWA rows into the ring (after the chunk's attention)
__global__ void kb_ring_write(const uint8_t* chunk, uint8_t* ring, int T, const int* pos0p, int win) {
    const int pos0 = *pos0p;
    const int t0 = max(0, T - win);
    for (int t = t0 + blockIdx.x; t < T; t += gridDim.x) {
        const uint8_t* src = chunk + (size_t) t * KV_RB;
        uint8_t* dst = ring + (size_t) ((pos0 + t) % (win + RING_PAD)) * KV_RB;
        for (int i = threadIdx.x; i < KV_RB / 4; i += blockDim.x)
            reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
    }
}
// RoPE NORM per token at pos0 + t (blockIdx.y = token)
__global__ void kb_rope(float* x, int ld, int heads, int stride, int off, int n_rot, const int* pos0p, float theta) {
    const int pos0 = *pos0p;
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x, half = n_rot / 2;
    if (i >= heads * half) return;
    const int h = i / half, j = i % half;
    const float th = (float) (pos0 + t) * powf(theta, -2.0f * j / n_rot);
    float c, s;
    sincosf(th, &s, &c);
    float* p = x + (size_t) t * ld + (size_t) h * stride + off + 2 * j;
    const float a = p[0], b = p[1];
    p[0] = a * c - b * s;
    p[1] = a * s + b * c;
}
// q_abs[t][h][j] (blockIdx.y = token)
__global__ void kb_absorb_q8(const uint8_t* kb, const float* q, int ldq, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * kvl) return;
    qabs[(size_t) t * H * kvl + i] = q8_0_dot(kb + (size_t) i * (nope / 32 * 34), q + (size_t) t * ldq + (size_t) (i / kvl) * qs, nope);
}
__global__ void kb_absorb_bf16(const uint16_t* kb, const float* q, int ldq, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * kvl) return;
    const uint16_t* w = kb + (size_t) i * nope;
    const float* x = q + (size_t) t * ldq + (size_t) (i / kvl) * qs;
    float s = 0.0f;
    for (int d = 0; d < nope; ++d) s += bf16f(w[d]) * x[d];
    qabs[(size_t) t * H * kvl + i] = s;
}
// batched absorbed MLA, grid (H, T), 256 threads: token t at pos0 + t.  SWA (cells null): the window over the ring
// (positions < pos0) and the chunk buffer (>= pos0).  DSA: cells [T][n_cap], context parallel as k_attn2.
__global__ void __launch_bounds__(256) kb_attn2(const float* qabs, const float* q, int ldq, int qs, int nope, float theta,
                                                const uint8_t* kv, const uint8_t* chunk, const int* cells, int n_cap,
                                                const int* pos0p, int win, int n_shards, int shard, const float* sinks,
                                                float scale, float* o, float* part, int H) {
    const int pos0 = *pos0p;
    constexpr int KVL = 512, ROPE = 64, NW = 8;
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5, pos = pos0 + t;
    float qa[16];
    {
        const float4* qa4 = reinterpret_cast<const float4*>(qabs + ((size_t) t * H + h) * KVL + lane * 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = qa4[i];
            qa[4 * i] = v.x, qa[4 * i + 1] = v.y, qa[4 * i + 2] = v.z, qa[4 * i + 3] = v.w;
        }
    }
    float qp0, qp1;
    {
        const float* qp = q + (size_t) t * ldq + (size_t) h * qs + nope + 2 * lane;
        const float th = (float) pos * powf(theta, -2.0f * lane / ROPE);
        float c, s;
        sincosf(th, &s, &c);
        qp0 = qp[0] * c - qp[1] * s;
        qp1 = qp[0] * s + qp[1] * c;
    }
    int n, lo = 0;
    const int* cr = cells ? cells + (size_t) t * n_cap : nullptr;
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
        const uint8_t* r;
        if (!cells) {
            const int p = lo + s;
            r = (chunk && p >= pos0) ? chunk + (size_t) (p - pos0) * KV_RB : kv + (size_t) (p % (win + RING_PAD)) * KV_RB;
        } else {
            const int c = cr[s];
            if (c < 0 || (n_shards > 1 && c % n_shards != shard)) continue;
            r = kv + (size_t) (n_shards == 1 ? c : c / n_shards) * KV_RB;
        }
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
        if (partial) part[((size_t) t * H + h) * (KVL + 4) + j] = v;
        else o[((size_t) t * H + h) * KVL + j] = v / L;
    }
    if (partial && threadIdx.x == 0) {
        part[((size_t) t * H + h) * (KVL + 4) + KVL] = M;
        part[((size_t) t * H + h) * (KVL + 4) + KVL + 1] = L;
    }
}
// a[t][h*vh + i] = (v_b . o[t][h]) * sigmoid(gate[t][h*vh + i]); warp per output, blockIdx.y = token
__global__ void kb_vb_gate_q8(const uint8_t* vb, const float* o, const float* gate, int ldg, float* a, int kvl, int vh, int H) {
    const int t = blockIdx.y, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint8_t* row = vb + (size_t) warp * (kvl / 32 * 34);
    const float* x = o + ((size_t) t * H + h) * kvl;
    float s = 0.0f;
    for (int b = lane; b < kvl / 32; b += 32) {
        const uint8_t* blk = row + b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        float tt = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) tt += (float) q[i] * x[b * 32 + i];
        s += d * tt;
    }
    s = warp_sum(s);
    if (lane == 0) a[(size_t) t * H * vh + warp] = s * sigm(gate[(size_t) t * ldg + warp]);
}
__global__ void kb_vb_gate_bf16(const uint16_t* vb, const float* o, const float* gate, int ldg, float* a, int kvl, int vh, int H) {
    const int t = blockIdx.y, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * vh) return;
    const int h = warp / vh;
    const uint16_t* w = vb + (size_t) warp * kvl;
    const float* x = o + ((size_t) t * H + h) * kvl;
    float s = 0.0f;
    for (int j = lane; j < kvl; j += 32) s += bf16f(w[j]) * x[j];
    s = warp_sum(s);
    if (lane == 0) a[(size_t) t * H * vh + warp] = s * sigm(gate[(size_t) t * ldg + warp]);
}
// small Q8_0 / F32 matrix times T rows: y[t][r] = W[r] . x[t] (warp per (r, t)); W [rows][n]
__global__ void kb_q8_gemv(const uint8_t* W, const float* x, int ldx, float* y, int ldy, int n, int rows) {
    const int t = blockIdx.y, r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint8_t* row = W + (size_t) r * (n / 32 * 34);
    const float* xr = x + (size_t) t * ldx;
    float s = 0.0f;
    for (int b = lane; b < n / 32; b += 32) {
        const uint8_t* blk = row + b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* q = reinterpret_cast<const int8_t*>(blk + 2);
        float tt = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) tt += (float) q[i] * xr[b * 32 + i];
        s += d * tt;
    }
    s = warp_sum(s);
    if (lane == 0) y[(size_t) t * ldy + r] = s;
}
__global__ void kb_f32_gemv(const float* W, const float* x, int ldx, float* y, int ldy, int n, int rows, float alpha) {
    const int t = blockIdx.y, r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* w = W + (size_t) r * n;
    const float* xr = x + (size_t) t * ldx;
    float s = 0.0f;
    for (int i = lane; i < n; i += 32) s += w[i] * xr[i];
    s = warp_sum(s);
    if (lane == 0) y[(size_t) t * ldy + r] = s * alpha;
}
// indexer: pool the groups that complete inside the chunk; members before pos0 come from the ring (the incomplete
// group carried over), members inside from the chunk's ivg rows.  blockIdx.x = group index from g_first, 128 threads.
__global__ void kb_pool_groups(const float* ivg, const float* ring, const int* pos0p, int T, const float* ape,
                               const float* norm, __half* pooled, int kd, int kpool, int n_rot, float eps, float theta) {
    const int pos0 = *pos0p, g = pos0 / kpool + blockIdx.x, d = threadIdx.x;
    if (g >= (pos0 + T) / kpool) return;
    __shared__ float vals[256];
    float v = 0.0f;
    if (d < kd) {
        float lg[8], vv[8], mx = -INFINITY, den = 0.0f;
        for (int mm = 0; mm < kpool; ++mm) {
            const int p = g * kpool + mm;
            const float* src = p >= pos0 ? ivg + (size_t) (p - pos0) * 2 * kd : ring + (size_t) (p % IRING) * 2 * kd;
            vv[mm] = src[d];
            lg[mm] = src[kd + d] + ape[(size_t) mm * kd + d];
            mx = fmaxf(mx, lg[mm]);
        }
        for (int mm = 0; mm < kpool; ++mm) den += (lg[mm] = __expf(lg[mm] - mx));
        for (int mm = 0; mm < kpool; ++mm) v += lg[mm] / den * vv[mm];
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
// the chunk's last IRING tokens' value/gate rows into the ring (slot = pos % IRING), after the pooling; IRING blocks
__global__ void kb_ring_chunk(const float* ivg, float* ring, int T, const int* pos0p, int kd, int kpool) {
    const int pos0 = *pos0p;
    const int t = T - IRING + (int) blockIdx.x;
    if (t < 0) return;
    const int slot = (pos0 + t) % IRING;
    for (int i = threadIdx.x; i < 2 * kd; i += blockDim.x) ring[(size_t) slot * 2 * kd + i] = ivg[(size_t) t * 2 * kd + i];
}
// scores of the tokens [t0, t0 + Tt) over their candidate groups g < pos / kpool (warp per group, blockIdx.y = token)
__global__ void kb_score(const float* iq, int ldq, const __half* pooled, const float* w, int ldw, int kd, int nh, int kpool,
                         int top_k, const int* pos0p, int t0, float* score, int ld_score) {
    const int pos0 = *pos0p;
    const int tt = blockIdx.y, t = t0 + tt, pos = pos0 + t;
    if (pos + 1 <= top_k) return;
    const int n_pool = pos / kpool;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_pool) return;
    const __half* pk = pooled + (size_t) warp * kd;
    const float* q = iq + (size_t) t * ldq;
    float pv[4];
    for (int i = 0; i < 4; ++i) pv[i] = lane + 32 * i < kd ? __half2float(pk[lane + 32 * i]) : 0.0f;
    float s = 0.0f;
    for (int h = 0; h < nh; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 4; ++i) d += lane + 32 * i < kd ? q[(size_t) h * kd + lane + 32 * i] * pv[i] : 0.0f;
        d = warp_sum(d);
        s += fmaxf(d, 0.0f) * w[(size_t) t * ldw + h];
    }
    if (lane == 0) score[(size_t) tt * ld_score + warp] = s;
}
// the moe grouping of this shard's experts, rows ordered by local expert then by (token, slot):
// kb_moe_count (blockIdx.x = local expert) -> kb_moe_scan (one block) -> kb_moe_place (blockIdx.x = local expert).
// bounds [n_local + 1], rows_tok [rows] (the activation row of each), row_of [T*k] (each (token, slot)'s row, -1 if
// another shard's expert).
__global__ void kb_moe_count(const int* ids, int n, int e0, int* cnt) {
    const int e = blockIdx.x + e0;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) c += ids[i] == e;
    c = (int) block_sum((float) c);
    if (threadIdx.x == 0) cnt[blockIdx.x] = c;
}
__global__ void kb_moe_scan(const int* cnt, int n_local, int* bounds) {
    __shared__ int v[1024];
    const int e = threadIdx.x;
    v[e] = e < n_local ? cnt[e] : 0;
    __syncthreads();
    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < n_local; ++i) {
            bounds[i] = acc;
            acc += v[i];
        }
        bounds[n_local] = acc;
    }
}
__global__ void kb_moe_place(const int* ids, int n, int k, int e0, int n_local, const int* bounds, int* rows_tok, int* row_of) {
    const int el = blockIdx.x, e = el + e0, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __shared__ int wt[32], s_base;
    if (threadIdx.x == 0) s_base = bounds[el];
    __syncthreads();
    for (int b0 = 0; b0 < n; b0 += blockDim.x) {
        const int i = b0 + threadIdx.x;
        const int id = i < n ? ids[i] : -1;
        const bool hit = id == e;
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) wt[w] = __popc(m);
        __syncthreads();
        int base = s_base;
        for (int q = 0; q < w; ++q) base += wt[q];
        if (hit) {
            const int r = base + __popc(m & ((1u << lane) - 1));
            rows_tok[r] = i / k;
            row_of[i] = r;
        }
        if (i < n && el == 0) {   // the other shards' slots
            const int loc = id - e0;
            if (loc < 0 || loc >= n_local) row_of[i] = -1;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int sacc = s_base;
            for (int q = 0; q < nw; ++q) sacc += wt[q];
            s_base = sacc;
        }
        __syncthreads();
    }
}
// shift a bounds slice to start at 0
__global__ void kb_shift_bounds(const int* src, int* dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) dst[i] = src[i] - src[0];
}
// moe[t][i] = sum over the token's slots this shard ran of w * D[row]  (blockIdx.y = token)
// (rows [ra, rb) of D only, D's row 0 = row ra; accumulate: add to moe)
__global__ void kb_moe_combine(const float* D, int ldd, const int* row_of, const float* w, int k, float* moe, int n, int ra,
                               int rb, int accumulate) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = accumulate ? moe[(size_t) t * n + i] : 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            const int r = row_of[t * k + kk];
            if (r >= ra && r < rb) s += w[t * k + kk] * D[(size_t) (r - ra) * ldd + i];
        }
        moe[(size_t) t * n + i] = s;
    }
}
// SwiGLU of rows [gate | up] (ld) -> h [rows][n_ff], no clamp (dense / shared expert)
__global__ void kb_swiglu_rows(const float* gu, int ld, int n_ff, int rows, float* h) {
    const int r = blockIdx.y;
    if (r >= rows) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ff; i += gridDim.x * blockDim.x)
        h[(size_t) r * n_ff + i] = silu(gu[(size_t) r * ld + i]) * gu[(size_t) r * ld + n_ff + i];
}

// ------------------------------------------------------------------------------------------------ batched exchange
// Push to ONE peer j (any number of blocks): this shard's share of the payload into j's receive area [src][per_dst];
// the last block to arrive signals j.  The caller launches it once per peer in ring order (j = me+1, me+2, .., me):
// a GPU's stores interleaved over three peers at once run at ~0.8 GB/s on this PCIe 3.0 board, one peer at a time at
// ~12.7 GB/s.  mode 0: the same payload to everyone (all-reduce); mode 1: j gets the head slice j of part
// [T][H_all][W] (slice = [T][Hl][W]).  float4 moves (per_dst and the slice a multiple of 4 floats).
__global__ void kb_push(const float* src, size_t per_dst, int mode, int T, int Hl, int W, Peers P, int me, int n_shards,
                        int step, int* arrive, int j) {
    const int par = step & 1;
    float4* dst = reinterpret_cast<float4*>(P.buf[j][par] + (size_t) me * per_dst);
    const float4* s4 = reinterpret_cast<const float4*>(src);
    const size_t n4 = per_dst / 4;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        size_t si = i;
        if (mode == 1) {
            const size_t slice = (size_t) Hl * W / 4, t = i / slice, rem = i % slice;
            si = t * (size_t) n_shards * slice + (size_t) j * slice + rem;
        }
        dst[i] = s4[si];
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
// the combine partials to ONE peer j as fp16: per (t, hl) a 1040-byte slot = acc / L (512 halfs: |acc / L| <= max |c|,
// so the half range is safe) + (M, L) as floats.  part [T][H][W] fp32 (acc, M, L), this shard's cells.  Half the
// fp32 payload; the last block signals j.
constexpr int CB_SLOT = 1040;
__global__ void kb_push_cb16(const float* part, int T, int H, int W, int Hl, Peers P, int me, int step, int* arrive, int j) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(P.buf[j][step & 1]) + (size_t) me * T * Hl * CB_SLOT;
    const int n = T * Hl * 256;   // half2 per slot: 256
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int slot = i / 256, c2 = i % 256, t = slot / Hl, hl = slot % Hl;
        const float* src = part + ((size_t) t * H + (size_t) j * Hl + hl) * W;
        const float L = src[513], inv = L > 0.0f ? 1.0f / L : 0.0f;
        reinterpret_cast<__half2*>(dst + (size_t) slot * CB_SLOT)[c2] = __floats2half2_rn(src[2 * c2] * inv, src[2 * c2 + 1] * inv);
        if (c2 == 0) *reinterpret_cast<float2*>(dst + (size_t) slot * CB_SLOT + 1024) = make_float2(src[512], L);
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
// The indexer selects ordered k-pools; their members are consecutive token
// positions. Up to 256K context, transmitting one 16-bit group id per pool
// reproduces every selected int32 cell exactly, including ties and the tail.
__global__ void kb_pack_groups(const int* __restrict__ cells, uint16_t* __restrict__ wire,
                               int rows, int cap, int kpool, int groups, int stride) {
    const int t = blockIdx.x;
    for (int g = threadIdx.x; g < stride; g += blockDim.x)
        wire[(size_t)t * stride + g] = g < groups ? (uint16_t)(cells[(size_t)t * cap + g * kpool] / kpool) : 0;
}
__global__ void kb_unpack_groups(const uint16_t* __restrict__ wire, int* __restrict__ cells, int cap,
                                 int kpool, int groups, int stride, const int* pos0p, int t0) {
    const int t = blockIdx.x, pos = *pos0p + t0 + t, current = pos / kpool;
    for (int i = threadIdx.x; i < cap; i += blockDim.x) {
        const int g = i / kpool;
        const int c = g < groups ? (int)wire[(size_t)t * stride + g] * kpool + (i % kpool)
                                 : current * kpool + i - groups * kpool;
        cells[(size_t)t * cap + i] = c <= pos ? c : -1;
    }
}
// dequantize a Q8_0 / BF16 matrix into F32 (the batched products' cuBLAS operand)
__global__ void kb_deq(const void* w, int type, size_t n, float* y) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        if (type == 8) {
            const uint8_t* blk = static_cast<const uint8_t*>(w) + (i / 32) * 34;
            y[i] = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
        } else {
            y[i] = bf16f(static_cast<const uint16_t*>(w)[i]);
        }
    }
}
// a[t][k] *= sigmoid(gate[t][k])  (k < n, gate rows ldg apart)
__global__ void kb_gate_mul(float* a, int n, const float* gate, int ldg, int T) {
    const int t = blockIdx.y;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
        a[(size_t) t * n + k] *= sigm(gate[(size_t) t * ldg + k]);
}
__device__ __forceinline__ void kb_wait(const Peers& P, int me, int n_shards, int step) {
    if (threadIdx.x < n_shards) {
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[threadIdx.x] < step) hy4f_spin_guard(t0);
    }
    __syncthreads();
    __threadfence_system();
}
__global__ void kb_wait_k(Peers P, int me, int n_shards, int step) { kb_wait(P, me, n_shards, step); }
// wait for one peer's flag (a paired phase: both partners finish phase r before either starts r + 1)
__global__ void kb_wait_one(Peers P, int me, int src, int step) {
    if (threadIdx.x == 0) {
        volatile int* mine = P.flag[me];
        const unsigned long long t0 = hy4f_now();
        while (mine[src] < step) hy4f_spin_guard(t0);
        __threadfence_system();
    }
}
// all-reduce tail per token (blockIdx.x = token, 1024 threads): y = RMS(sum over shards [+ x2 local... folded in the
// payload]) * w, X = R + post * y, R = X
__global__ void kb_ar_post(const float* local, int n, Peers P, int me, int n_shards, int step, const float* nw, float eps,
                           float* X, float* R, const float* post, int hc, size_t per_dst) {
    const int t = blockIdx.x;
    __shared__ float ys[4096];
    if (n_shards > 1) {
        kb_wait(P, me, n_shards, step);
        const volatile float* rv = P.buf[me][step & 1];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            float s = 0.0f;
            for (int j = 0; j < n_shards; ++j) s += rv[(size_t) j * per_dst + (size_t) t * n + i];
            ys[i] = s;
        }
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) ys[i] = local[(size_t) t * n + i];
    }
    __syncthreads();
    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq += ys[i] * ys[i];
    const float r = rsqrtf(block_sum(sq) / n + eps);
    X += (size_t) t * hc * n;
    R += (size_t) t * hc * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float y = ys[i] * r * nw[i];
        for (int s = 0; s < hc; ++s) {
            const float v = R[(size_t) s * n + i] + post[(size_t) t * 8 + s] * y;
            X[(size_t) s * n + i] = v;
            R[(size_t) s * n + i] = v;
        }
    }
}
// the combine tail per token (blockIdx.x = token): o[t][hl] for this shard's heads from the shards' partials
__global__ void kb_combine(int kvl, int W, int h0, int Hl, const float* sinks_all, Peers P, int me, int n_shards, int step,
                           size_t per_dst, float* o) {
    const int t = blockIdx.x;
    kb_wait(P, me, n_shards, step);
    const volatile float* rv = P.buf[me][step & 1];
    for (int hl = 0; hl < Hl; ++hl) {
        const int h = h0 + hl;
        float m = sinks_all[h];
        for (int j = 0; j < n_shards; ++j) m = fmaxf(m, rv[(size_t) j * per_dst + ((size_t) t * Hl + hl) * W + kvl]);
        float den = __expf(sinks_all[h] - m);
        float f[4];
        for (int j = 0; j < n_shards; ++j) {
            const size_t b = (size_t) j * per_dst + ((size_t) t * Hl + hl) * W;
            const float mj = rv[b + kvl];
            f[j] = mj == -INFINITY ? 0.0f : __expf(mj - m);
            den += f[j] * rv[b + kvl + 1];
        }
        for (int i = threadIdx.x; i < kvl; i += blockDim.x) {
            float acc = 0.0f;
            for (int j = 0; j < n_shards; ++j) acc += f[j] * rv[(size_t) j * per_dst + ((size_t) t * Hl + hl) * W + i];
            o[((size_t) t * Hl + hl) * kvl + i] = acc / den;
        }
    }
}
// ---- the all-reduce as reduce-scatter (bf16 partials) + all-gather (fp16 normalized rows), by token slices: shard j
// owns tokens [j*ts, min(T, (j+1)*ts)).  ~1/8 of the fp32 push-to-everyone traffic.
// Stage contiguous local payloads for the PCIe copy engines. All numeric conversions
// have the same order and rounding as the direct-to-peer push kernels.
// The reduce-scatter wire format rounds the F32 branch sum once to BF16.
__global__ void kb_rs_pack_add(const float4* __restrict__ routed, const float4* __restrict__ shared,
                               __nv_bfloat162* __restrict__ dst, int quads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= quads) return;
    const float4 a = routed[i], b = shared[i];
    dst[2 * i] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
    dst[2 * i + 1] = __floats2bfloat162_rn(a.z + b.z, a.w + b.w);
}

__global__ void kb_rs_pack(const float* __restrict__ y, __nv_bfloat162* __restrict__ dst, int pairs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pairs) {
        const float2 v = reinterpret_cast<const float2*>(y)[i];
        dst[i] = __floats2bfloat162_rn(v.x, v.y);
    }
}
__global__ void kb_pack_cb16(const float* __restrict__ part, int T, int H, int W, int Hl, uint8_t* __restrict__ dst) {
    const int t = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    const int peer = h / Hl, hl = h % Hl;
    const float* row = part + ((size_t) t * H + h) * W;
    uint8_t* out = dst + (((size_t) peer * T + t) * Hl + hl) * CB_SLOT;
    const float L = row[513], inv = L > 0.0f ? 1.0f / L : 0.0f;
    const float2 v = reinterpret_cast<const float2*>(row)[c];
    reinterpret_cast<__half2*>(out)[c] = __floats2half2_rn(v.x * inv, v.y * inv);
    if (c == 0) *reinterpret_cast<float2*>(out + 1024) = make_float2(row[512], L);
}
// The preceding D2D copy and this kernel are ordered by the source stream.
__global__ void kb_copy_done(int* peer_flags, int from, int step) {
    if (threadIdx.x == 0) reinterpret_cast<volatile int*>(peer_flags)[from] = step;
}
__global__ void kb_ag_residual(const __half* __restrict__ yn, float* __restrict__ X,
                               const float* __restrict__ post, int E, int hc) {
    const int t = blockIdx.y, h = blockIdx.z;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= E) return;
    const size_t col = (size_t)t * E + i;
    const __half2* y = reinterpret_cast<const __half2*>(yn + col);
    const float2 a = __half22float2(y[0]), b = __half22float2(y[1]);
    float4* dst = reinterpret_cast<float4*>(X + ((size_t)t * hc + h) * E + i);
    const float4 v = *dst;
    const float w = post[t * 8 + h];
    *dst = make_float4(v.x + w * a.x, v.y + w * a.y, v.z + w * b.x, v.w + w * b.y);
}
// RS push to peer j: rows of j's slice of y [T][E] (fp32) as bf16 into j's area [me][ts][E]
__global__ void kb_rs_push(const float* y, int T, int E, int ts, Peers P, int me, int step, int* arrive, int j) {
    const int r0 = j * ts, nr = max(0, min(T, r0 + ts) - r0);
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(P.buf[j][step & 1]) + (size_t) me * ts * E;
    const size_t n8 = (size_t) nr * E / 8;
    const float4* src = reinterpret_cast<const float4*>(y + (size_t) r0 * E);
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (size_t) gridDim.x * blockDim.x) {
        const float4 a = src[2 * i], b = src[2 * i + 1];
        __nv_bfloat162 v[4] = {__floats2bfloat162_rn(a.x, a.y), __floats2bfloat162_rn(a.z, a.w),
                               __floats2bfloat162_rn(b.x, b.y), __floats2bfloat162_rn(b.z, b.w)};
        reinterpret_cast<uint4*>(dst)[i] = *reinterpret_cast<uint4*>(v);
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
// RS tail (blockIdx.x = row of this shard's slice, 1024 threads): sum the shards' partials, RMS * w -> yn (fp16)
__global__ void kb_rs_post(int T, int E, int ts, Peers P, int me, int n_shards, int step, const float* nw, float eps,
                           __half* yn) {
    const int r = blockIdx.x, t = me * ts + r;
    if (t >= T) return;
    kb_wait(P, me, n_shards, step);
    const __nv_bfloat16* rv = reinterpret_cast<const __nv_bfloat16*>(P.buf[me][step & 1]);
    __shared__ float ys[4096];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float v = 0.0f;
        for (int j = 0; j < n_shards; ++j) v += __bfloat162float(rv[((size_t) j * ts + r) * E + i]);
        ys[i] = v;
        sq += v * v;
    }
    const float k = rsqrtf(block_sum(sq) / E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) yn[(size_t) r * E + i] = __float2half_rn(ys[i] * k * nw[i]);
}
// AG push to peer j: this shard's yn rows into j's area [T][E] at rows me*ts ..
__global__ void kb_ag_push(const __half* yn, int T, int E, int ts, Peers P, int me, int step, int* arrive, int j) {
    const int r0 = me * ts, nr = max(0, min(T, r0 + ts) - r0);
    __half* dst = reinterpret_cast<__half*>(P.buf[j][step & 1]) + (size_t) r0 * E;
    const size_t n8 = (size_t) nr * E / 8;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += (size_t) gridDim.x * blockDim.x)
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(yn)[i];
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
// AG tail (blockIdx.x = token): X = R + post * y, R = X
__global__ void kb_ag_post(int E, Peers P, int me, int n_shards, int step, float* X, float* R, const float* post, int hc) {
    const int t = blockIdx.x;
    kb_wait(P, me, n_shards, step);
    const __half* y = reinterpret_cast<const __half*>(P.buf[me][step & 1]) + (size_t) t * E;
    X += (size_t) t * hc * E;
    R += (size_t) t * hc * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = __half2float(y[i]);
        for (int s = 0; s < hc; ++s) {
            const float x = R[(size_t) s * E + i] + post[(size_t) t * 8 + s] * v;
            X[(size_t) s * E + i] = x;
            R[(size_t) s * E + i] = x;
        }
    }
}

// One CTA owns one residual row. The 16 HC subreductions keep the
// original 256-thread summation tree; the final 1024-thread RMS uses
// exactly the same column assignment as kb_hc_apply_norm.
template<bool ApplyResidual>
__global__ void __launch_bounds__(1024) kb_hc_norm_fused(float* X, const __half* yn, const float* old_post,
                    const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base,
                    const float* __restrict__ nw, int n_mix, float eps, float hc_eps, float mag,
                    float* __restrict__ output, float* post) {
    constexpr int E = 3072, H = 4, Splits = 16, Subgroup = 256, Chunk = (E * H) / Splits;
    __shared__ union {
        struct { float warp[Splits][9][8], partial[9][Splits]; } reduction;
        float cur[E];
    } sm;
    __shared__ float pre[H];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid % 32;
    const int g = tid / Subgroup, local = tid % Subgroup, warp = local / 32;
    const int col = g * Chunk + local;
    X += (size_t)t * H * E;
    // Keeping two streams in shared memory avoids register spills at 1024 threads.
    __shared__ float shared_x[2 * E];
    float keep[H - 2][3];
#pragma unroll
    for (int h = 0; h < H; ++h) {
        const int seg = h * 4 + g;
        float sq = 0.0f, dots[8] = {};
        const float weight = ApplyResidual ? old_post[(size_t)t * 8 + h] : 0.0f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int c = col + j * Subgroup, i = h * E + c;
            float v = X[i];
            if constexpr (ApplyResidual) {
                v += weight * __half2float(yn[(size_t)t * E + c]);
                X[i] = v;
            }
            if (h < 2) shared_x[h * E + c] = v;
            else keep[h - 2][j] = v;
            sq += v * v;
#pragma unroll
            for (int r = 0; r < 8; ++r)
                if (r < n_mix) dots[r] += fn[(size_t)r * E * H + i] * v;
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float v = warp_sum(dots[r]);
            if (!lane) sm.reduction.warp[seg][r][warp] = v;
        }
        sq = warp_sum(sq);
        if (!lane) sm.reduction.warp[seg][8][warp] = sq;
    }
    __syncthreads();
    if (tid < 9 * Splits) {
        const int r = tid / Splits, seg = tid % Splits;
        float v = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) v += sm.reduction.warp[seg][r][i];
        sm.reduction.partial[r][seg] = v;
    }
    __syncthreads();
    if (tid < 32) {
        const float sq = warp_sum(tid < Splits ? sm.reduction.partial[8][tid] : 0.0f);
        const float inv = rsqrtf(sq / (E * H) + eps);
        for (int r = 0; r < n_mix; ++r) {
            const float v = warp_sum(tid < Splits ? sm.reduction.partial[r][tid] : 0.0f) * inv;
            if (!tid) {
                if (r < H) pre[r] = sigm(v * scale[0] + base[r]) + hc_eps;
                else post[(size_t)t * 8 + r - H] = sigm(v * scale[1] + base[r]) * mag + hc_eps;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float v = 0;
#pragma unroll
        for (int h = 0; h < H; ++h) {
            v += pre[h] * (h < 2 ? shared_x[h * E + col + j * Subgroup] : keep[h - 2][j]);
        }
        sm.cur[col + j * Subgroup] = v;
    }
    __syncthreads();
    float sq = 0;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float v = sm.cur[tid + j * 1024];
        sq += v * v;
    }
    const float r = rsqrtf(block_sum(sq) / E + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int c = tid + j * 1024;
        output[(size_t)t * E + c] = sm.cur[c] * r * nw[c];
    }
}


// the combine tail over fp16 slots (kb_push_cb16), per token (blockIdx.x = token): o[t][hl] for this shard's heads
__global__ void kb_combine16(int kvl, int h0, int Hl, const float* sinks_all, Peers P, int me, int n_shards, int step,
                             int T, float* o) {
    const int t = blockIdx.x;
    kb_wait(P, me, n_shards, step);
    const uint8_t* rv = reinterpret_cast<const uint8_t*>(P.buf[me][step & 1]);
    const size_t per = (size_t) T * Hl * CB_SLOT;
    for (int hl = 0; hl < Hl; ++hl) {
        const int h = h0 + hl;
        float mj[4], lj[4], m = sinks_all[h];
        for (int j = 0; j < n_shards; ++j) {
            const float2 ml = __ldcv(reinterpret_cast<const float2*>(rv + j * per + ((size_t) t * Hl + hl) * CB_SLOT + 1024));
            mj[j] = ml.x, lj[j] = ml.y;
            m = fmaxf(m, mj[j]);
        }
        float den = __expf(sinks_all[h] - m), f[4];
        for (int j = 0; j < n_shards; ++j) {
            f[j] = mj[j] == -INFINITY ? 0.0f : __expf(mj[j] - m) * lj[j];   // (the slot holds acc / L)
            den += f[j];
        }
        for (int c2 = threadIdx.x; c2 < kvl / 2; c2 += blockDim.x) {
            float ax = 0.0f, ay = 0.0f;
            for (int j = 0; j < n_shards; ++j) {
                const __half2 v = reinterpret_cast<const __half2*>(rv + j * per + ((size_t) t * Hl + hl) * CB_SLOT)[c2];
                const float2 fv = __half22float2(v);
                ax += f[j] * fv.x, ay += f[j] * fv.y;
            }
            o[((size_t) t * Hl + hl) * kvl + 2 * c2] = ax / den;
            o[((size_t) t * Hl + hl) * kvl + 2 * c2 + 1] = ay / den;
        }
    }
}

__global__ void kb_add(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}

}  // namespace kern

// The sample estimates the candidate count only. When fewer than k or more than
// the bounded candidate capacity qualify, the original full selector takes over.
constexpr int SELECT_CAND_LIMIT = 2048;

__global__ void kb_select_pivot(const float* __restrict__ scores, int ld, const int* pos0p, int t0,
                                int pool, int k, unsigned* pivots, unsigned* counts, int* fallback) {
    constexpr int NT = 256, NS = 1024;
    using Sort = cub::BlockRadixSort<unsigned, NT, NS / NT>;
    __shared__ typename Sort::TempStorage tmp;
    __shared__ unsigned extremes[2];
    const int t = blockIdx.x, n = (*pos0p + t0 + t) / pool, tid = threadIdx.x;
    if (k < 1 || k > 512 || n > 65536 || n < NS) {
        if (!tid) { counts[t] = 0; fallback[t] = 1; }
        return;
    }
    unsigned buf[NS / NT];
#pragma unroll
    for (int i = 0; i < NS / NT; ++i) {
        const int rank = tid * (NS / NT) + i;
        buf[i] = hy4f_key(scores[(size_t)t * ld + (long long)rank * n / NS]);
    }
    Sort(tmp).SortDescending(buf);
    const int rank = min(NS - 1, (int)(((long long)k * NS * 7 / 4 + n - 1) / n));
    if (tid == rank / (NS / NT)) pivots[t] = buf[rank % (NS / NT)];
    if (tid == 0) extremes[0] = buf[0];
    if (tid == NT - 1) extremes[1] = buf[NS / NT - 1];
    __syncthreads();
    if (!tid) {
        counts[t] = 0;
        fallback[t] = extremes[0] == extremes[1];
    }
}

__global__ void kb_select_filter(const float* __restrict__ scores, int ld, const int* pos0p, int t0, int pool,
                                 const unsigned* pivots, unsigned* counts, const int* fallback, uint2* candidates) {
    const int t = blockIdx.y, tid = threadIdx.x, lane = tid % 32;
    if (fallback[t]) return;
    const int n = (*pos0p + t0 + t) / pool;
    const unsigned pivot = pivots[t];
    uint2* dst = candidates + (size_t)t * SELECT_CAND_LIMIT;
    for (int i = blockIdx.x * blockDim.x + tid; i - lane < n; i += blockDim.x * gridDim.x) {
        const unsigned key = i < n ? hy4f_key(scores[(size_t)t * ld + i]) : 0;
        const bool take = i < n && key >= pivot;
        const unsigned mask = __ballot_sync(0xffffffffu, take);
        if (mask) {
            unsigned base = 0;
            if (!lane) base = atomicAdd(counts + t, __popc(mask));
            base = __shfl_sync(0xffffffffu, base, 0);
            const unsigned off = base + __popc(mask & ((1u << lane) - 1));
            if (take && off < SELECT_CAND_LIMIT) dst[off] = make_uint2(key, i);
        }
    }
}

__global__ void kb_select_finish(const uint2* __restrict__ candidates, const unsigned* counts, int* fallback,
                                 const int* pos0p, int t0, int tb, int pool, int k, int cap, int* cells_all) {
    constexpr int NT = 256, Items = SELECT_CAND_LIMIT / NT;
    using Sort = cub::BlockRadixSort<unsigned, NT, Items>;
    __shared__ typename Sort::TempStorage tmp;
    __shared__ unsigned kth;
    const int t = blockIdx.x, tid = threadIdx.x, pos = *pos0p + t0 + t, n = pos / pool;
    const unsigned count = counts[t];
    if (fallback[t] || count < (unsigned)k || count > SELECT_CAND_LIMIT || n > 65536) {
        if (!tid) fallback[t] = 1;
        return;
    }
    unsigned tags[Items], keys[Items], before[Items];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = tid * Items + i;
        const uint2 v = rank < count ? candidates[(size_t)t * SELECT_CAND_LIMIT + rank] : make_uint2(0, 0x3ffffu);
        keys[i] = before[i] = v.x;
        tags[i] = v.y;
    }
    Sort(tmp).SortDescending(keys);
    if (tid == (k - 1) / Items) kth = keys[(k - 1) % Items];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        if (tags[i] != 0x3ffffu)
            tags[i] = before[i] > kth ? tags[i] : before[i] == kth ? (0x10000u | tags[i]) : 0x3ffffu;
    }
    // Preserve full-selector order: all greater keys by index, then equal
    // keys by index. Excess equal keys fall strictly beyond the first k.
    Sort(tmp).Sort(tags, 0, 18);
    int* cells = cells_all + (size_t)(t0 + t - tb) * cap;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = tid * Items + i;
        if (rank < k)
            for (int j = 0; j < pool; ++j) cells[rank * pool + j] = int(tags[i] % 65536u) * pool + j;
    }
    for (int j = k * pool + tid; j < cap; j += NT) {
        const int c = n * pool + j - k * pool;
        cells[j] = c <= pos ? c : -1;
    }
}


// the batched k-pool selection (blockIdx.x = token of the tile): as k_select_kpool for pos = pos0 + t0 + blockIdx.x
template <int NT>
__global__ void kb_select_kpool(const float* __restrict__ score_tile, int ld_score, const int* pos0p, int t0, int tb, int kpool, int top_k,
                                int top_pools, int n_cap, int* __restrict__ cells_all, const int* fallback = nullptr) {
    if (fallback && !fallback[blockIdx.x]) return;
    const int pos0 = *pos0p;
    const int tt = blockIdx.x, t = t0 + tt, pos = pos0 + t;
    const float* score = score_tile + (size_t) tt * ld_score;
    int* cells = cells_all + (size_t) (t - tb) * n_cap;   // (rows of the attention tile that starts at token tb)
    const int tid = threadIdx.x;
    if (pos + 1 <= top_k) {
        for (int i = tid; i < n_cap; i += NT) cells[i] = i <= pos ? i : -1;
        return;
    }
    const int n_pool = pos / kpool, gcur = n_pool;
    const int k = min(top_pools, n_pool);
    __shared__ unsigned hist[2048];
    __shared__ unsigned s_prefix, s_mask;
    __shared__ int s_kk;
    if (tid == 0) s_prefix = 0, s_mask = 0, s_kk = k;
    __syncthreads();
    using SelScan = cub::BlockScan<int, NT>;
    __shared__ typename SelScan::TempStorage sel_tmp;
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
        const int sh = shifts[pass], nbins = 1 << widths[pass];
        for (int i = tid; i < nbins; i += NT) hist[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_pool; i += NT) {
            const unsigned key = hy4f_key(score[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> sh) & (nbins - 1)], 1u);
        }
        __syncthreads();
        {   // the bin where the count from the top crosses kk: a block scan over the bins in descending order (two per
            // thread) instead of one thread walking up to 2048 bins
            const int kk = s_kk;
            const int r0 = nbins - 1 - 2 * tid, r1 = r0 - 1;
            const int c0 = r0 >= 0 ? (int) hist[r0] : 0, c1 = r1 >= 0 ? (int) hist[r1] : 0;
            int above, total;
            SelScan(sel_tmp).ExclusiveSum(c0 + c1, above, total);
            __syncthreads();   // (every thread has read s_kk and its bins before the winner writes)
            if (r0 >= 0 && above < kk && kk <= above + c0) {
                s_prefix = prefix | ((unsigned) r0 << sh);
                s_mask = mask | ((unsigned) (nbins - 1) << sh);
                s_kk = kk - above;
            } else if (r1 >= 0 && above + c0 < kk && kk <= above + c0 + c1) {
                s_prefix = prefix | ((unsigned) r1 << sh);
                s_mask = mask | ((unsigned) (nbins - 1) << sh);
                s_kk = kk - above - c0;
            }
        }
        __syncthreads();
    }
    const unsigned thr = s_prefix;
    const int ties = s_kk;
    // the groups above the threshold, then the first `ties` at it, both in index order (block scans: deterministic,
    // so the cells - and the attention's summation order - are the same every run and in every path)
    using Scan = cub::BlockScan<int, NT>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ int s_gt, s_eq, s_ngt;
    if (tid == 0) s_gt = 0, s_eq = 0;
    __syncthreads();
    int ngt_local = 0;
    for (int i = tid; i < n_pool; i += NT) ngt_local += hy4f_key(score[i]) > thr;
    {
        int dummy, tot;
        Scan(tmp).ExclusiveSum(ngt_local, dummy, tot);
        if (tid == 0) s_ngt = tot;
    }
    __syncthreads();
    const int n_gt = s_ngt;
    for (int c0 = 0; c0 < n_pool; c0 += NT) {
        const int i = c0 + tid;
        const unsigned key = i < n_pool ? hy4f_key(score[i]) : 0u;
        const int fg = (i < n_pool && key > thr) ? 1 : 0, fe = (i < n_pool && key == thr) ? 1 : 0;
        int rg, tg, re, te;
        Scan(tmp).ExclusiveSum(fg, rg, tg);
        __syncthreads();
        Scan(tmp).ExclusiveSum(fe, re, te);
        const int bg = s_gt, be = s_eq;
        if (fg) {
            const int slot = bg + rg;
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
        if (fe && be + re < ties) {
            const int slot = n_gt + be + re;
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
        __syncthreads();
        if (tid == 0) s_gt = bg + tg, s_eq = be + te;
        __syncthreads();
    }
    const int n_top = k * kpool;
    for (int i = n_top + tid; i < n_cap; i += NT) {
        const int c = gcur * kpool + (i - n_top);
        cells[i] = c <= pos ? c : -1;
    }
}

}  // namespace strata::hy4f
