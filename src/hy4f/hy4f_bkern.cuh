// src/hy4f/hy4f_bkern.cuh - maya-hy4f batched kernels (a chunk of T tokens at positions pos0 .. pos0 + T - 1): the
// prompt path and the speculative verify.  blockIdx.y (or blockIdx.x where noted) is the token.  Included by hy4f.cu.
#pragma once

#include "hy4f_kern.cuh"

namespace strata::hy4f {
namespace kern {

__global__ void kb_embed(const uint8_t* emb, size_t rb, const int* toks, int pos0, float* X, float* R, int E, int hc,
                         int* hist) {
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
__global__ void kb_engram_hash(const int* hist, int pos0, const int64_t* map, int64_t n_map, int64_t pad_id,
                               const uint64_t* mult, const uint64_t* vsz, const uint64_t* off, int ng, int nh, int64_t* ids) {
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
                              const float* gamma, int E, int hc, float eps) {
    const int t = blockIdx.x;
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
// HC (blockIdx.z = token)
__global__ void kb_hc_mix(const float* X, const float* fn, int n, float* part, float* ss) {
    const int r = blockIdx.x, s = blockIdx.y, t = blockIdx.z, chunk = n / HC_SPLIT, i0 = s * chunk;
    X += (size_t) t * n;
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
        part[((size_t) t * 8 + r) * HC_SPLIT + s] = d;
        if (r == 0) ss[(size_t) t * HC_SPLIT + s] = q;
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
// mode 2 (DSA, context parallel) the cache slot when this shard owns the position.  blockIdx.x = token, 576 threads.
__global__ void kb_kv_prep(const float* kvin, int ld, const float* nw, int kvl, int rope, float eps, float theta,
                           uint8_t* dst_base, int pos0, int mode, int n_shards, int shard) {
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
__global__ void kb_ring_write(const uint8_t* chunk, uint8_t* ring, int T, int pos0, int win) {
    const int t0 = max(0, T - win);
    for (int t = t0 + blockIdx.x; t < T; t += gridDim.x) {
        const uint8_t* src = chunk + (size_t) t * KV_RB;
        uint8_t* dst = ring + (size_t) ((pos0 + t) % win) * KV_RB;
        for (int i = threadIdx.x; i < KV_RB / 4; i += blockDim.x)
            reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
    }
}
// RoPE NORM per token at pos0 + t (blockIdx.y = token)
__global__ void kb_rope(float* x, int ld, int heads, int stride, int off, int n_rot, int pos0, float theta) {
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
                                                int pos0, int win, int n_shards, int shard, const float* sinks,
                                                float scale, float* o, float* part, int H) {
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
            r = p >= pos0 ? chunk + (size_t) (p - pos0) * KV_RB : kv + (size_t) (p % win) * KV_RB;
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
        if (partial) part[((size_t) t * H + h) * (KVL + 2) + j] = v;
        else o[((size_t) t * H + h) * KVL + j] = v / L;
    }
    if (partial && threadIdx.x == 0) {
        part[((size_t) t * H + h) * (KVL + 2) + KVL] = M;
        part[((size_t) t * H + h) * (KVL + 2) + KVL + 1] = L;
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
__global__ void kb_pool_groups(const float* ivg, const float* ring, int pos0, int g_first, const float* ape,
                               const float* norm, float* pooled, int kd, int kpool, int n_rot, float eps, float theta) {
    const int g = g_first + blockIdx.x, d = threadIdx.x;
    __shared__ float vals[256];
    float v = 0.0f;
    if (d < kd) {
        float lg[8], vv[8], mx = -INFINITY, den = 0.0f;
        for (int mm = 0; mm < kpool; ++mm) {
            const int p = g * kpool + mm;
            const float* src = p >= pos0 ? ivg + (size_t) (p - pos0) * 2 * kd : ring + (size_t) (p % kpool) * 2 * kd;
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
    if (d < kd) pooled[(size_t) g * kd + d] = vals[d];
}
// the chunk's last tokens' value/gate rows into the ring (slot = pos % kpool), after the pooling
__global__ void kb_ring_chunk(const float* ivg, float* ring, int T, int pos0, int kd, int kpool) {
    const int t = T - kpool + (int) blockIdx.x;
    if (t < 0) return;
    const int slot = (pos0 + t) % kpool;
    for (int i = threadIdx.x; i < 2 * kd; i += blockDim.x) ring[(size_t) slot * 2 * kd + i] = ivg[(size_t) t * 2 * kd + i];
}
// scores of the tokens [t0, t0 + Tt) over their candidate groups g < pos / kpool (warp per group, blockIdx.y = token)
__global__ void kb_score(const float* iq, int ldq, const float* pooled, const float* w, int ldw, int kd, int nh, int kpool,
                         int top_k, int pos0, int t0, float* score, int ld_score) {
    const int tt = blockIdx.y, t = t0 + tt, pos = pos0 + t;
    if (pos + 1 <= top_k) return;
    const int n_pool = pos / kpool;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_pool) return;
    const float* pk = pooled + (size_t) warp * kd;
    const float* q = iq + (size_t) t * ldq;
    float pv[4];
    for (int i = 0; i < 4; ++i) pv[i] = lane + 32 * i < kd ? pk[lane + 32 * i] : 0.0f;
    float s = 0.0f;
    for (int h = 0; h < nh; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 4; ++i) d += lane + 32 * i < kd ? q[(size_t) h * kd + lane + 32 * i] * pv[i] : 0.0f;
        d = warp_sum(d);
        s += fmaxf(d, 0.0f) * w[(size_t) t * ldw + h];
    }
    if (lane == 0) score[(size_t) tt * ld_score + warp] = s;
}
// the moe grouping of this shard's experts: rows ordered by local expert, then by (token, slot); one block, one thread
// per local expert (n_local <= blockDim).  bounds [n_local + 1], rows_tok [rows] (the activation row), row_of [T*k]
// (the row of each (token, slot), -1 elsewhere).
__global__ void kb_moe_group(const int* ids, int T, int k, int e0, int n_local, int* bounds, int* rows_tok, int* row_of) {
    __shared__ int cnt[1024];
    const int e = threadIdx.x;
    int c = 0;
    if (e < n_local)
        for (int i = 0; i < T * k; ++i) c += ids[i] - e0 == e;
    cnt[e] = c;
    __syncthreads();
    if (e == 0) {
        int acc = 0;
        for (int i = 0; i < n_local; ++i) {
            bounds[i] = acc;
            acc += cnt[i];
        }
        bounds[n_local] = acc;
    }
    __syncthreads();
    if (e < n_local) {
        int r = bounds[e];
        for (int i = 0; i < T * k; ++i)
            if (ids[i] - e0 == e) rows_tok[r] = i / k, row_of[i] = r++;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T * k; i += blockDim.x) {
        const int loc = ids[i] - e0;
        if (loc < 0 || loc >= n_local) row_of[i] = -1;
    }
}
// moe[t][i] = sum over the token's slots this shard ran of w * D[row]  (blockIdx.y = token)
__global__ void kb_moe_combine(const float* D, int ldd, const int* row_of, const float* w, int k, float* moe, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            const int r = row_of[t * k + kk];
            if (r >= 0) s += w[t * k + kk] * D[(size_t) r * ldd + i];
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
// Push (any number of blocks): each block copies its share of the payload into every peer's receive area
// [src][per_dst]; the last block to arrive signals the peers.  mode 0: the same payload to everyone (all-reduce);
// mode 1: shard j gets the head slice j of part [T][H_all][W] (slice = [T][Hl][W]).
__global__ void kb_push(const float* src, size_t per_dst, int mode, int T, int Hl, int W, Peers P, int me, int n_shards,
                        int step, int* arrive) {
    const int par = step & 1;
    for (int j = 0; j < n_shards; ++j) {
        float* dst = P.buf[j][par] + (size_t) me * per_dst;
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < per_dst; i += (size_t) gridDim.x * blockDim.x) {
            size_t si = i;
            if (mode == 1) {
                const size_t slice = (size_t) Hl * W, t = i / slice, rem = i % slice;
                si = t * (size_t) n_shards * slice + (size_t) j * slice + rem;
            }
            dst[i] = src[si];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int v = atomicAdd(arrive, 1);
        if (v == (int) gridDim.x - 1) {
            __threadfence_system();
            for (int j = 0; j < n_shards; ++j) reinterpret_cast<volatile int*>(P.flag[j])[me] = step;
            *arrive = 0;
        }
    }
}
__device__ __forceinline__ void kb_wait(const Peers& P, int me, int n_shards, int step) {
    if (threadIdx.x < n_shards) {
        volatile int* mine = P.flag[me];
        while (mine[threadIdx.x] < step) {
        }
    }
    __syncthreads();
    __threadfence_system();
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
__global__ void kb_combine(int kvl, int h0, int Hl, const float* sinks_all, Peers P, int me, int n_shards, int step,
                           size_t per_dst, float* o) {
    const int t = blockIdx.x, W = kvl + 2;
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
__global__ void kb_add(const float* a, const float* b, float* y, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}

}  // namespace kern

// the batched k-pool selection (blockIdx.x = token of the tile): as k_select_kpool for pos = pos0 + t0 + blockIdx.x
template <int NT>
__global__ void kb_select_kpool(const float* __restrict__ score_tile, int ld_score, int pos0, int t0, int kpool, int top_k,
                                int top_pools, int n_cap, int* __restrict__ cells_all) {
    const int tt = blockIdx.x, t = t0 + tt, pos = pos0 + t;
    const float* score = score_tile + (size_t) tt * ld_score;
    int* cells = cells_all + (size_t) t * n_cap;
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
        if (tid == 0) {
            int kk = s_kk;
            for (int b = nbins - 1; b >= 0; --b) {
                const int c = (int) hist[b];
                if (c >= kk) {
                    s_prefix = prefix | ((unsigned) b << sh);
                    s_mask = mask | ((unsigned) (nbins - 1) << sh);
                    break;
                }
                kk -= c;
            }
            s_kk = kk;
        }
        __syncthreads();
    }
    const unsigned thr = s_prefix;
    const int ties = s_kk;
    __shared__ int s_out;
    if (tid == 0) s_out = 0;
    __syncthreads();
    for (int i = tid; i < n_pool; i += NT)
        if (hy4f_key(score[i]) > thr) {
            const int slot = atomicAdd(&s_out, 1);
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
    __syncthreads();
    using Scan = cub::BlockScan<int, NT>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ int s_taken;
    if (tid == 0) s_taken = 0;
    __syncthreads();
    const int base = s_out;
    for (int c0 = 0; c0 < n_pool; c0 += NT) {
        const int i = c0 + tid;
        const int flag = (i < n_pool && hy4f_key(score[i]) == thr) ? 1 : 0;
        int rank, total;
        Scan(tmp).ExclusiveSum(flag, rank, total);
        const int taken = s_taken;
        if (flag && taken + rank < ties) {
            const int slot = base + taken + rank;
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
        __syncthreads();
        if (tid == 0) s_taken = taken + total;
        __syncthreads();
        if (s_taken >= ties) break;
    }
    const int n_top = k * kpool;
    for (int i = n_top + tid; i < n_cap; i += NT) {
        const int c = gcur * kpool + (i - n_top);
        cells[i] = c <= pos ? c : -1;
    }
}

}  // namespace strata::hy4f
