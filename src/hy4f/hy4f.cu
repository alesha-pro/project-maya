// src/hy4f/hy4f.cu - maya-hy4f v0: the Hy4 flash decode (see hy4f.hpp).  One token per step through every layer.
//
// Per layer (docs/HY4F-SPEC.md): [Engram] -> HC pre (hc 4, pre/post weights only) -> rms*attn_norm -> MLA attention
// (DSA layers: k-pool indexer selection; SWA layers: the last 512 positions) with a learnable sink and a sigmoid gate
// -> rms*post_attention_norm -> HC post -> HC pre -> rms*ffn_norm -> dense FFN (layer 0) or LatentMoE (+ shared) ->
// rms*post_ffw_norm -> HC post.  Head: HC head pre -> rms*output_norm -> output (BF16).
//
// Weights stay in their GGUF blocks on the device: Q8_0 / Q3_K / Q4_K / Q6_K / Q4_0 through Maya's native MMVQ, BF16
// through bf16_gemv_fp32_mmvf, F32 through a plain GEMV.  The per-head absorbed k_b / v_b (Q8_0, 3-D) have their own
// kernels.  The Engram table (F8_E4M3, 63 GB) lives in pinned host RAM, mapped: the GPUs gather a token's 64 rows over
// PCIe in place.  v0 keeps a few host round trips (the routed experts' ids, the indexer's top pools past top_k, the
// Engram hash) - they go to the device in the fast path.
#include "hy4f.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/glm_dsa.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/router_sigmoid.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace strata::hy4f {

namespace kn = strata::kernels;

namespace {

#define HY_CK(x)                                                                                                        \
    do {                                                                                                                \
        cudaError_t e_ = (x);                                                                                           \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)

// GGUF type ids as the AngelSlim GGUF writes them (ggml numbering, F8_E4M3 = 43 in the patched ggml)
enum : int { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q3_K = 11, T_Q4_K = 12, T_Q6_K = 14, T_BF16 = 30, T_F8 = 43 };

size_t row_bytes(int type, int64_t n) {
    switch (type) {
    case T_F32: return (size_t) n * 4;
    case T_F16:
    case T_BF16: return (size_t) n * 2;
    case T_F8: return (size_t) n;
    case T_Q4_0: return (size_t) n / 32 * 18;
    case T_Q8_0: return (size_t) n / 32 * 34;
    case T_Q3_K: return (size_t) n / 256 * 110;
    case T_Q4_K: return (size_t) n / 256 * 144;
    case T_Q6_K: return (size_t) n / 256 * 210;
    default: throw std::runtime_error("hy4f: unsupported tensor type " + std::to_string(type));
    }
}

// ------------------------------------------------------------------------------------------------ kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum (blockDim multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v) {
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
__device__ float block_max(float v) {
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

__device__ __forceinline__ float q8_0_dot(const uint8_t* row, const float* x, int n) {   // one thread, whole row
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
__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((uint32_t) h << 16); }
__device__ __forceinline__ float fp8e4m3f(uint8_t v) {   // OCP E4M3FN
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float f;
    if (e == 0) f = ldexpf((float) m, -9);                       // subnormal: m/8 * 2^-6
    else if (e == 15 && m == 7) f = __int_as_float(0x7fc00000);  // NaN
    else f = ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -f : f;
}

__global__ void k_q8_0_row(const uint8_t* row, float* out, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint8_t* blk = row + (i / 32) * 34;
        out[i] = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) reinterpret_cast<const int8_t*>(blk + 2)[i % 32];
    }
}
__global__ void k_fill(float* x, int n, float v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] = v;
}
__global__ void k_rep(const float* e, float* X, int E, int hc) {   // X[i] = e for every stream
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E * hc; i += gridDim.x * blockDim.x) X[i] = e[i % E];
}
// y = x / rms(x) * w (w may be null), one block
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    s = block_sum(s);
    const float k = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * k * (w ? w[i] : 1.0f);
}
// y[r] = sum_i W[r][i] x[i], W F32 row-major [rows][n]; one warp per row
__global__ void k_f32_gemv(const float* W, const float* x, float* y, int n, int rows) {
    const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* w = W + (size_t) r * n;
    float s = 0.0f;
    for (int i = lane; i < n; i += 32) s += w[i] * x[i];
    s = warp_sum(s);
    if (lane == 0) y[r] = s;
}
// HC pre: mixes = fn [n_mix x hc*E] . rms(X); pre_i = sigmoid(m_i * scale[0] + base_i) + hc_eps;
// post_i = sigmoid(m_{hc+i} * scale[1] + base_{hc+i}) * mag + hc_eps (n_mix == 2*hc); cur = sum_i pre_i X_i.  One block.
__global__ void k_hc_pre(const float* X, const float* fn, const float* scale, const float* base, int E, int hc,
                         int n_mix, float eps, float hc_eps, float mag, float* cur, float* post) {
    __shared__ float pre[8];
    const int n = E * hc;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += X[i] * X[i];
    const float inv = rsqrtf(block_sum(s) / n + eps);
    for (int r = 0; r < n_mix; ++r) {
        float d = 0.0f;
        const float* w = fn + (size_t) r * n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) d += w[i] * X[i];
        d = block_sum(d) * inv;
        if (threadIdx.x == 0) {
            if (r < hc) pre[r] = sigm(d * scale[0] + base[r]) + hc_eps;
            else post[r - hc] = sigm(d * scale[1] + base[r]) * mag + hc_eps;
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < E; d += blockDim.x) {
        float v = 0.0f;
        for (int i = 0; i < hc; ++i) v += pre[i] * X[(size_t) i * E + d];
        cur[d] = v;
    }
}
__global__ void k_hc_post(float* X, const float* R, const float* y, const float* post, int E, int hc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E * hc; i += gridDim.x * blockDim.x)
        X[i] = R[i] + post[i / E] * y[i % E];
}
// RoPE NORM on x[h*stride + off + 0..n_rot) for every head, position pos, base theta (llama.cpp's theta_scale form)
__global__ void k_rope(float* x, int heads, int stride, int off, int n_rot, float pos, float theta) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, half = n_rot / 2;
    if (t >= heads * half) return;
    const int h = t / half, j = t % half;
    const float th = pos * powf(theta, -2.0f * j / n_rot);
    float c, s;
    sincosf(th, &s, &c);
    float* p = x + (size_t) h * stride + off + 2 * j;
    const float a = p[0], b = p[1];
    p[0] = a * c - b * s;
    p[1] = a * s + b * c;
}
// q_abs[h][j] = sum_d k_b[d, j, h] q[h*qs + d]  (k_b Q8_0 {nope, kvl, H}: row (h*kvl + j) has nope values)
__global__ void k_absorb_q8(const uint8_t* kb, const float* q, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * kvl) return;
    const int h = t / kvl;
    qabs[t] = q8_0_dot(kb + (size_t) t * (nope / 32 * 34), q + (size_t) h * qs, nope);
}
__global__ void k_absorb_bf16(const uint16_t* kb, const float* q, int qs, float* qabs, int nope, int kvl, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * kvl) return;
    const int h = t / kvl;
    const uint16_t* w = kb + (size_t) t * nope;
    float s = 0.0f;
    for (int d = 0; d < nope; ++d) s += bf16f(w[d]) * q[(size_t) h * qs + d];
    qabs[t] = s;
}
// a[h*vh + i] = sum_j v_b[j, i, h] o[h*kvl + j]  (v_b {kvl, vh, H}: row (h*vh + i) has kvl values)
__global__ void k_vb_q8(const uint8_t* vb, const float* o, float* a, int kvl, int vh, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * vh) return;
    const int h = t / vh;
    a[t] = q8_0_dot(vb + (size_t) t * (kvl / 32 * 34), o + (size_t) h * kvl, kvl);
}
__global__ void k_vb_bf16(const uint16_t* vb, const float* o, float* a, int kvl, int vh, int H) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * vh) return;
    const int h = t / vh;
    const uint16_t* w = vb + (size_t) t * kvl;
    float s = 0.0f;
    for (int j = 0; j < kvl; ++j) s += bf16f(w[j]) * o[(size_t) h * kvl + j];
    a[t] = s;
}
// Absorbed MLA with a sink over the selected cells (cells == null: the range lo .. lo+n_sel-1).  One block per head.
// score = (q_abs_h . c_s + q_pe_h . kpe_s) * scale; w = softmax over the cells with exp(sink_h) in the denominator;
// o_h = sum_s w_s c_s.  kv rows: [c (kvl) | kpe (rope)].  sc: scratch [H][max_sel].
__global__ void k_attn(const float* qabs, const float* q, int qs, int nope, const float* kv, int kvl, int rope,
                       const int* cells, int lo, int n_sel, const float* sinks, float scale, float* o, float* sc,
                       int max_sel) {
    const int h = blockIdx.x, KR = kvl + rope;
    const float* qa = qabs + (size_t) h * kvl;
    const float* qp = q + (size_t) h * qs + nope;
    float* s_ = sc + (size_t) h * max_sel;
    float mx = -INFINITY;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const int c = cells ? cells[s] : lo + s;
        float v = -INFINITY;
        if (c >= 0) {
            const float* K = kv + (size_t) c * KR;
            float d = 0.0f;
            for (int j = 0; j < kvl; ++j) d += qa[j] * K[j];
            for (int r = 0; r < rope; ++r) d += qp[r] * K[kvl + r];
            v = d * scale;
        }
        s_[s] = v;
        mx = fmaxf(mx, v);
    }
    mx = fmaxf(block_max(mx), sinks[h]);
    float den = 0.0f;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const float e = s_[s] == -INFINITY ? 0.0f : __expf(s_[s] - mx);
        s_[s] = e;
        den += e;
    }
    den = block_sum(den) + __expf(sinks[h] - mx);
    const float inv = 1.0f / den;
    for (int j = threadIdx.x; j < kvl; j += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int c = cells ? cells[s] : lo + s;
            if (c >= 0) acc += s_[s] * kv[(size_t) c * KR + j];
        }
        o[(size_t) h * kvl + j] = acc * inv;
    }
}
__global__ void k_sigmul(float* a, const float* g, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) a[i] *= sigm(g[i]);
}
__global__ void k_swiglu(float* ga, const float* up, int n, float lim) {   // lim = INFINITY: plain SwiGLU
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = fminf(ga[i], lim), u = fminf(fmaxf(up[i], -lim), lim);
        ga[i] = silu(g) * u;
    }
}
__global__ void k_axpy_k(float* acc, const float* x, int n, const float* w, int k) {   // acc += w[k] * x
    const float a = w[k];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc[i] += a * x[i];
}
__global__ void k_add(const float* a, const float* b, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}
__global__ void k_scale(float* x, int n, float s) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) x[i] *= s;
}
// a pooled indexer key: rms * norm, then RoPE NORM on its last n_rot dims at position pos0.  One block.
__global__ void k_pool_finish(float* p, const float* w, int n, float eps, int n_rot, float pos0, float theta) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += p[i] * p[i];
    const float k = rsqrtf(block_sum(s) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = p[i] * k * w[i];
    __syncthreads();
    const int half = n_rot / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        const float th = pos0 * powf(theta, -2.0f * j / n_rot);
        float c, sn;
        sincosf(th, &sn, &c);
        float* q = p + (n - n_rot) + 2 * j;
        const float a = q[0], b = q[1];
        q[0] = a * c - b * sn;
        q[1] = a * sn + b * c;
    }
}
// Engram rows: r[col] = rms(fp8 row ids[col]) * lnorm[col]  (one block per column, D <= blockDim)
__global__ void k_engram_rows(const uint8_t* table, const int64_t* ids, const float* lnorm, float* r, int D, float eps) {
    const int col = blockIdx.x, d = threadIdx.x;
    const float v = d < D ? fp8e4m3f(table[(size_t) ids[col] * D + d]) : 0.0f;
    const float k = rsqrtf(block_sum(v * v) / D + eps);
    if (d < D) r[(size_t) col * D + d] = v * k * lnorm[(size_t) col * D + d];
}
// Engram mix: kv = [key_0..key_{hc-1} | value] (+ bias already); delta_i = rms(value)*gamma * (gate_i + hcs_i);
// gate_i = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))), dot = sum_d rms(X_i)_d rms(key_i)_d gw_i,d / sqrt(E).  One block.
__global__ void k_engram_mix(float* X, const float* kv, const float* gw, const float* hcs, const float* gamma,
                             float* delta, int E, int hc, float eps) {
    const float* val = kv + (size_t) hc * E;
    float s = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x) s += val[d] * val[d];
    const float kv_ = rsqrtf(block_sum(s) / E + eps);
    for (int i = 0; i < hc; ++i) {
        const float* x = X + (size_t) i * E;
        const float* k = kv + (size_t) i * E;
        float sx = 0.0f, sk = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) sx += x[d] * x[d], sk += k[d] * k[d];
        const float ix = rsqrtf(block_sum(sx) / E + eps), ik = rsqrtf(block_sum(sk) / E + eps);
        float dt = 0.0f;
        for (int d = threadIdx.x; d < E; d += blockDim.x) dt += x[d] * ix * k[d] * ik * gw[(size_t) i * E + d];
        dt = block_sum(dt) * rsqrtf((float) E);
        const float mag = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        const float gate = sigm(dt < 0.0f ? -mag : mag);
        for (int d = threadIdx.x; d < E; d += blockDim.x) {
            const float dl = val[d] * kv_ * gamma[d] * (gate + hcs[(size_t) i * E + d]);
            delta[(size_t) i * E + d] = dl;
            X[(size_t) i * E + d] += dl;
        }
        __syncthreads();
    }
}

int nb(int n, int t = 256) { return (n + t - 1) / t; }

// ------------------------------------------------------------------------------------------------ model state
struct DW {   // a device weight
    int type = -1;
    int64_t ne[3] = {1, 1, 1};
    void* d = nullptr;
    size_t rb = 0;   // row bytes
};

struct Part {
    int dev = 0, l0 = 0, l1 = 0;
    cudaStream_t st = nullptr;
    std::map<std::string, DW> w;
    std::map<int, float*> kv;                    // layer -> F32 [max_ctx][kvl + rope]
    std::map<int, float*> iv, ig, pooled;        // DSA layer -> indexer state
    void* q8 = nullptr;                          // Q8_1 scratch for MMVQ
    float *X = nullptr, *R = nullptr, *cur = nullptr, *x = nullptr, *y = nullptr, *post = nullptr;
    float *qa = nullptr, *qr = nullptr, *q = nullptr, *kvrow = nullptr, *qabs = nullptr, *o = nullptr, *a = nullptr,
          *gate = nullptr, *ao = nullptr, *iq = nullptr, *ik = nullptr, *ikn = nullptr, *iw = nullptr, *score = nullptr,
          *sc = nullptr;
    int* cells = nullptr;
    float *logits_r = nullptr, *ew = nullptr, *xl = nullptr, *ga = nullptr, *up = nullptr, *dn = nullptr, *moe = nullptr,
          *mu = nullptr, *sh = nullptr, *ffn = nullptr;
    int* eids = nullptr;
    float *eg_r = nullptr, *eg_kv = nullptr, *eg_delta = nullptr;
    int64_t* eg_ids = nullptr;
    float *head_cur = nullptr, *head_x = nullptr, *logits = nullptr, *emb = nullptr;
};

}  // namespace

struct Model::Impl {
    Options o;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    // geometry
    int E = 0, hc = 0, L = 0, V = 0, q_lora = 0, kvl = 0, nope = 0, rope = 0, vh = 0, Hd = 0, Hs = 0, win = 0, IH = 0,
        IK = 0, top_k = 0, kpool = 0, tail = 0, n_exp = 0, n_used = 0, n_ffx = 0, latent = 0, dense_lead = 0, n_ff = 0;
    float eps = 1e-5f, hc_eps = 1e-6f, hc_mag = 2.0f, w_scale = 1.0f, clampv = 10.0f;
    float theta = 1e7f, theta_swa = 1e4f;
    std::vector<int> swa, eg_layers;
    std::vector<uint64_t> eg_mult, eg_off, eg_vsz;
    std::vector<int64_t> eg_map;
    int eg_ngram = 3, eg_heads = 8, eg_D = 96, eg_pad = 0;
    uint8_t* eg_table = nullptr;   // pinned, mapped
    std::vector<Part> parts;
    std::vector<int32_t> hist;     // tokens so far (Engram history)
    int max_sel = 0;

    const strata::TensorInfo* find(const std::string& n, const strata::GgufFile** f) const {
        for (const auto& g : files)
            if (const strata::TensorInfo* t = g->find(n)) {
                *f = g.get();
                return t;
            }
        return nullptr;
    }
    uint64_t meta_u(const std::string& k) const {
        const strata::MetaValue* v = files[0]->get("hy_v4_flash." + k);
        if (!v) throw std::runtime_error("hy4f: missing key " + k);
        return v->type == strata::MetaType::F32 || v->type == strata::MetaType::F64 ? (uint64_t) v->f : v->u;
    }
    double meta_f(const std::string& k) const {
        const strata::MetaValue* v = files[0]->get("hy_v4_flash." + k);
        if (!v) throw std::runtime_error("hy4f: missing key " + k);
        return v->num();
    }
    template <class T> std::vector<T> meta_arr(const std::string& k) const {
        strata::MetaValue v;
        if (!files[0]->get_full("hy_v4_flash." + k, v)) throw std::runtime_error("hy4f: missing key " + k);
        std::vector<T> out;
        for (const auto& it : v.items) out.push_back((T) (it.type == strata::MetaType::F32 ? it.f : (double) it.u));
        return out;
    }
    template <class T> std::vector<T> meta_arr_u(const std::string& k) const {   // integers exact (uint64)
        strata::MetaValue v;
        if (!files[0]->get_full("hy_v4_flash." + k, v)) throw std::runtime_error("hy4f: missing key " + k);
        std::vector<T> out;
        for (const auto& it : v.items) out.push_back((T) it.u);
        return out;
    }

    void geometry() {
        E = (int) meta_u("embedding_length");
        hc = (int) meta_u("hyper_connection.count");
        L = (int) meta_u("block_count") - (int) meta_u("nextn_predict_layers");
        V = (int) meta_u("vocab_size");
        q_lora = (int) meta_u("attention.q_lora_rank");
        kvl = (int) meta_u("attention.kv_lora_rank");
        rope = (int) meta_u("rope.dimension_count");
        nope = (int) meta_u("attention.key_length_mla") - rope;
        vh = (int) meta_u("attention.value_length_mla");
        win = (int) meta_u("attention.sliding_window");
        IH = (int) meta_u("attention.indexer.head_count");
        IK = (int) meta_u("attention.indexer.key_length");
        top_k = (int) meta_u("attention.indexer.top_k");
        kpool = (int) meta_u("attention.indexer.kpool");
        tail = (int) meta_u("attention.indexer.kpool_select_tail");
        n_exp = (int) meta_u("expert_count");
        n_used = (int) meta_u("expert_used_count");
        n_ffx = (int) meta_u("expert_feed_forward_length");
        latent = (int) meta_u("expert_latent_length");
        dense_lead = (int) meta_u("leading_dense_block_count");
        n_ff = (int) meta_u("feed_forward_length");
        eps = (float) meta_f("attention.layer_norm_rms_epsilon");
        hc_eps = (float) meta_f("hyper_connection.epsilon");
        hc_mag = (float) meta_f("hyper_connection.magnitude");
        w_scale = (float) meta_f("expert_weights_scale");
        theta = (float) meta_f("rope.freq_base");
        theta_swa = (float) meta_f("rope.freq_base_swa");
        swa = meta_arr<int>("attention.sliding_window_pattern");
        const std::vector<int> heads = meta_arr<int>("attention.head_count");
        clampv = meta_arr<float>("swiglu_clamp_exp")[0];
        for (int il = 0; il < L; ++il) (swa[il] ? Hs : Hd) = heads[il];
        eg_layers = meta_arr<int>("ple.layers");
        eg_mult = meta_arr_u<uint64_t>("ple.layer_multipliers");
        eg_off = meta_arr_u<uint64_t>("ple.head_offsets");
        eg_vsz = meta_arr_u<uint64_t>("ple.head_vocab_sizes");
        eg_map = meta_arr_u<int64_t>("engram.token_map");
        eg_ngram = (int) meta_u("ple.ngram_size");
        eg_heads = (int) meta_u("ple.heads_per_ngram");
        eg_D = (int) meta_u("embedding_length_per_layer_input");
        eg_pad = (int) meta_u("engram.pad_token_id");
        max_sel = std::max(win, top_k + kpool);
    }

    DW upload(Part& p, const std::string& name) {
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = find(name, &f);
        if (!t) throw std::runtime_error("hy4f: missing tensor " + name);
        DW w;
        w.type = (int) t->type;
        for (size_t i = 0; i < 3 && i < t->shape.size(); ++i) w.ne[i] = (int64_t) t->shape[i];
        w.rb = row_bytes(w.type, w.ne[0]);
        const size_t bytes = w.rb * (size_t) (w.ne[1] * w.ne[2]);
        HY_CK(cudaMalloc(&w.d, bytes));
        HY_CK(cudaMemcpy(w.d, f->tensor_data(*t), bytes, cudaMemcpyHostToDevice));
        p.w[name] = w;
        return w;
    }
    size_t layer_bytes(int il) const {
        size_t s = 0;
        const std::string P = "blk." + std::to_string(il) + ".";
        for (const auto& g : files)
            for (const auto& t : g->tensors())
                if (t.name.rfind(P, 0) == 0) {
                    int64_t n = 1;
                    for (size_t i = 1; i < t.shape.size(); ++i) n *= (int64_t) t.shape[i];
                    s += row_bytes((int) t.type, (int64_t) t.shape[0]) * (size_t) n;
                }
        return s;
    }

    float* dalloc(size_t n) {
        float* p = nullptr;
        HY_CK(cudaMalloc(&p, n * sizeof(float)));
        HY_CK(cudaMemset(p, 0, n * sizeof(float)));
        return p;
    }

    void load_engram() {
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = find("per_layer_token_embd.weight", &f);
        if (!t) throw std::runtime_error("hy4f: no Engram table");
        if ((int) t->type != T_F8) throw std::runtime_error("hy4f: Engram table is not F8_E4M3");
        const size_t bytes = (size_t) t->shape[0] * (size_t) t->shape[1];
        HY_CK(cudaHostAlloc((void**) &eg_table, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        const uint8_t* src = f->tensor_data(*t);
        const size_t chunk = (size_t) 1 << 30;
        for (size_t off = 0; off < bytes; off += chunk) std::memcpy(eg_table + off, src + off, std::min(chunk, bytes - off));
        if (o.verbose) std::fprintf(stderr, "hy4f: Engram table %.1f GiB pinned + mapped\n", bytes / 1073741824.0);
    }

    void build_parts() {
        int ndev = 0;
        HY_CK(cudaGetDeviceCount(&ndev));
        std::vector<int> devs = o.gpus;
        if (devs.empty())
            for (int i = 0; i < ndev; ++i) devs.push_back(i);
        std::vector<int> bounds = o.split;
        if (bounds.empty()) {   // by weight bytes
            std::vector<size_t> lb(L);
            size_t tot = 0;
            for (int il = 0; il < L; ++il) tot += lb[il] = layer_bytes(il);
            size_t acc = 0;
            int k = 1;
            for (int il = 0; il < L && k < (int) devs.size(); ++il) {
                acc += lb[il];
                if (acc >= tot * k / devs.size()) bounds.push_back(il + 1), ++k;
            }
        }
        int l0 = 0;
        for (size_t i = 0; i < devs.size(); ++i) {
            Part p;
            p.dev = devs[i];
            p.l0 = l0;
            p.l1 = i < bounds.size() ? bounds[i] : L;
            l0 = p.l1;
            parts.push_back(std::move(p));
            if (l0 >= L) break;
        }
    }

    void load_part(Part& p, bool first, bool last) {
        HY_CK(cudaSetDevice(p.dev));
        HY_CK(cudaStreamCreateWithFlags(&p.st, cudaStreamNonBlocking));
        if (first) upload(p, "token_embd.weight");
        if (last)
            for (const char* n : {"output.weight", "output_norm.weight", "output_hc_fn.weight", "output_hc_scale.weight",
                                  "output_hc_base.weight"})
                upload(p, n);
        const int64_t C = o.max_context;
        for (int il = p.l0; il < p.l1; ++il) {
            const std::string P = "blk." + std::to_string(il) + ".";
            for (const auto& g : files)
                for (const auto& t : g->tensors())
                    if (t.name.rfind(P, 0) == 0) upload(p, t.name);
            p.kv[il] = dalloc((size_t) C * (kvl + rope));
            if (!swa[il]) {
                p.iv[il] = dalloc((size_t) C * IK);
                p.ig[il] = dalloc((size_t) C * IK);
                p.pooled[il] = dalloc((size_t) (C / kpool + 1) * IK);
            }
        }
        const int H = std::max(Hd, Hs);
        HY_CK(cudaMalloc(&p.q8, kn::native_q8_1_bytes(std::max({16384, n_ff, H * vh, E * (hc + 1)}), 1)));
        p.X = dalloc((size_t) hc * E);
        p.R = dalloc((size_t) hc * E);
        p.cur = dalloc(E);
        p.x = dalloc(E);
        p.y = dalloc(E);
        p.post = dalloc(8);
        p.qa = dalloc(q_lora);
        p.qr = dalloc(q_lora);
        p.q = dalloc((size_t) H * (nope + rope));
        p.kvrow = dalloc(kvl + rope);
        p.qabs = dalloc((size_t) H * kvl);
        p.o = dalloc((size_t) H * kvl);
        p.a = dalloc((size_t) H * vh);
        p.gate = dalloc((size_t) H * vh);
        p.ao = dalloc(E);
        p.iq = dalloc((size_t) IH * IK);
        p.ik = dalloc(IK);
        p.ikn = dalloc(IK);
        p.iw = dalloc(IH);
        p.score = dalloc((size_t) (C / kpool + 1));
        p.sc = dalloc((size_t) H * max_sel);
        HY_CK(cudaMalloc(&p.cells, sizeof(int) * max_sel));
        p.logits_r = dalloc(n_exp);
        p.ew = dalloc(n_used);
        HY_CK(cudaMalloc(&p.eids, sizeof(int) * n_used));
        p.xl = dalloc(latent);
        p.ga = dalloc(std::max(n_ff, n_ffx));
        p.up = dalloc(std::max(n_ff, n_ffx));
        p.dn = dalloc(std::max(latent, E));
        p.moe = dalloc(latent);
        p.mu = dalloc(E);
        p.sh = dalloc(E);
        p.ffn = dalloc(E);
        p.eg_r = dalloc((size_t) eg_D * (eg_ngram - 1) * eg_heads);
        p.eg_kv = dalloc((size_t) E * (hc + 1));
        p.eg_delta = dalloc((size_t) E * hc);
        HY_CK(cudaMalloc(&p.eg_ids, sizeof(int64_t) * (eg_ngram - 1) * eg_heads));
        if (first) p.emb = dalloc(E);
        if (last) {
            p.head_cur = dalloc(E);
            p.head_x = dalloc(E);
            p.logits = dalloc(V);
        }
    }

    // y = W . x for W [n_in -> n_out] at byte offset `off` (an expert slice)
    void gemv(Part& p, const DW& w, size_t off, const float* x, float* y, int n_in, int n_out) {
        const uint8_t* d = static_cast<const uint8_t*>(w.d) + off;
        switch (w.type) {
        case T_Q8_0: kn::native_q8_0_f32(d, x, p.q8, y, n_in, n_out, 1, p.st); break;
        case T_Q3_K: kn::native_q3_k_f32(d, x, p.q8, y, n_in, n_out, 1, p.st); break;
        case T_Q4_K: kn::native_q4_k_f32(d, x, p.q8, y, n_in, n_out, 1, p.st); break;
        case T_Q6_K: kn::native_q6_k_f32(d, x, p.q8, y, n_in, n_out, 1, p.st); break;
        case T_Q4_0: kn::native_q4_0_f32(d, x, p.q8, y, n_in, n_out, 1, p.st); break;
        case T_BF16: kn::bf16_gemv_fp32_mmvf(x, reinterpret_cast<const uint16_t*>(d), y, n_in, n_out, p.st); break;
        case T_F32: k_f32_gemv<<<nb(n_out, 8), 256, 0, p.st>>>(reinterpret_cast<const float*>(d), x, y, n_in, n_out); break;
        default: throw std::runtime_error("hy4f: gemv type " + std::to_string(w.type));
        }
    }
    void gemv(Part& p, const std::string& name, const float* x, float* y) {
        const DW& w = p.w.at(name);
        gemv(p, w, 0, x, y, (int) w.ne[0], (int) w.ne[1]);
    }
    const float* f32(Part& p, const std::string& name) {
        const DW& w = p.w.at(name);
        if (w.type != T_F32) throw std::runtime_error("hy4f: " + name + " is not F32");
        return static_cast<const float*>(w.d);
    }

    // ---- seams (GLM_CB-style names, appended per token)
    void dump(Part& p, const std::string& name, const float* dptr, size_t n) {
        if (o.dump_dir.empty()) return;
        std::vector<float> h(n);
        HY_CK(cudaMemcpyAsync(h.data(), dptr, n * 4, cudaMemcpyDeviceToHost, p.st));
        HY_CK(cudaStreamSynchronize(p.st));
        std::FILE* f = std::fopen((o.dump_dir + "/" + name + ".f32").c_str(), "ab");
        if (f) std::fwrite(h.data(), 4, n, f), std::fclose(f);
    }

    void engram(Part& p, int e, int il, int64_t pos) {
        const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
        auto cid = [&](int64_t t) -> int64_t { return t < 0 || t >= (int64_t) eg_map.size() ? -1 : eg_map[t]; };
        const int64_t pad = eg_map[eg_pad];
        std::vector<uint64_t> tk(ng);
        int64_t c = cid(hist[pos]);
        tk[0] = c < 0 ? pad : c;
        for (int s = 1; s < ng; ++s) {
            c = pos - s < 0 ? -1 : cid(hist[pos - s]);
            tk[s] = c < 0 ? pad : c;
        }
        const uint64_t* mult = eg_mult.data() + (size_t) e * ng;
        std::vector<int64_t> ids(ncol);
        uint64_t rolling = tk[0] * mult[0];
        for (int oo = 1; oo < ng; ++oo) {
            rolling ^= tk[oo] * mult[oo];
            for (int h = 0; h < nh; ++h) {
                const int col = (oo - 1) * nh + h;
                ids[col] = (int64_t) (rolling % eg_vsz[(size_t) e * ncol + col] + eg_off[(size_t) e * ncol + col]);
            }
        }
        const std::string P = "blk." + std::to_string(il) + ".";
        HY_CK(cudaMemcpyAsync(p.eg_ids, ids.data(), sizeof(int64_t) * ncol, cudaMemcpyHostToDevice, p.st));
        k_engram_rows<<<ncol, 128, 0, p.st>>>(eg_table, p.eg_ids, f32(p, P + "engram_lookup_norm.weight"), p.eg_r, eg_D,
                                              1.1920929e-07f);
        gemv(p, P + "engram_kv.weight", p.eg_r, p.eg_kv);
        k_add<<<nb(E * (hc + 1)), 256, 0, p.st>>>(p.eg_kv, f32(p, P + "engram_kv_b.weight"), p.eg_kv, E * (hc + 1));
        k_engram_mix<<<1, 1024, 0, p.st>>>(p.X, p.eg_kv, f32(p, P + "engram_gate.weight"),
                                           f32(p, P + "engram_hc_scale.weight"), f32(p, P + "engram_pre_gamma.weight"),
                                           p.eg_delta, E, hc, 1.1920929e-07f);
        dump(p, "engram_delta-" + std::to_string(il), p.eg_delta, (size_t) E * hc);
    }

    // the k-pool selection for a DSA layer at position pos; returns n_sel (cells in p.cells, -1 none) or 0 for "all"
    int select_kpool(Part& p, int il, int64_t pos) {
        if (pos + 1 <= top_k) return 0;
        const int64_t gcur = pos / kpool;           // the current token's group, forced through the tail
        const int n_pool = (int) gcur;              // candidate groups: completed and not the current one
        kn::glm_dsa_score(p.iq, p.pooled[il], p.iw, IK, IH, 1, n_pool, p.score, p.st);
        std::vector<float> s(n_pool);
        HY_CK(cudaMemcpyAsync(s.data(), p.score, 4 * (size_t) n_pool, cudaMemcpyDeviceToHost, p.st));
        HY_CK(cudaStreamSynchronize(p.st));
        std::vector<int> idx(n_pool);
        std::iota(idx.begin(), idx.end(), 0);
        const int n_top = std::min(n_pool, top_k / kpool - tail);
        std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(),
                          [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
        std::vector<int> cells;
        for (int i = 0; i < n_top; ++i)
            for (int k = 0; k < kpool; ++k) cells.push_back(idx[i] * kpool + k);
        for (int64_t c = gcur * kpool; c <= pos; ++c) cells.push_back((int) c);
        HY_CK(cudaMemcpyAsync(p.cells, cells.data(), sizeof(int) * cells.size(), cudaMemcpyHostToDevice, p.st));
        return (int) cells.size();
    }

    void attention(Part& p, int il, int64_t pos) {
        const std::string P = "blk." + std::to_string(il) + ".", S = "-" + std::to_string(il);
        const bool sw = swa[il] != 0;
        const int H = sw ? Hs : Hd, QH = nope + rope, KR = kvl + rope;
        const float th = sw ? theta_swa : theta;
        gemv(p, P + "attn_q_a.weight", p.x, p.qa);
        k_rms<<<1, 1024, 0, p.st>>>(p.qa, f32(p, P + "attn_q_a_norm.weight"), p.qr, q_lora, eps);
        gemv(p, P + "attn_q_b.weight", p.qr, p.q);
        float* kvc = p.kv[il] + (size_t) pos * KR;
        gemv(p, P + "attn_kv_a_mqa.weight", p.x, p.kvrow);
        k_rope<<<nb(H * rope / 2), 256, 0, p.st>>>(p.q, H, QH, nope, rope, (float) pos, th);
        k_rope<<<1, 64, 0, p.st>>>(p.kvrow, 1, KR, kvl, rope, (float) pos, th);
        k_rms<<<1, 512, 0, p.st>>>(p.kvrow, f32(p, P + "attn_kv_a_norm.weight"), p.kvrow, kvl, eps);
        HY_CK(cudaMemcpyAsync(kvc, p.kvrow, sizeof(float) * KR, cudaMemcpyDeviceToDevice, p.st));
        dump(p, "kv_cmpr" + S, p.kvrow, kvl);

        int n_sel = 0, lo = 0;
        const int* cells = nullptr;
        if (sw) {
            lo = (int) std::max<int64_t>(0, pos - (win - 1));
            n_sel = (int) (pos - lo + 1);
        } else {
            gemv(p, P + "indexer.attn_q_b.weight", p.qr, p.iq);
            k_rope<<<nb(IH * rope / 2), 256, 0, p.st>>>(p.iq, IH, IK, IK - rope, rope, (float) pos, th);
            gemv(p, P + "indexer.attn_k.weight", p.x, p.ik);
            k_rms<<<1, 128, 0, p.st>>>(p.ik, f32(p, P + "indexer.k_norm.weight"), p.ikn, IK, eps);
            gemv(p, P + "indexer_compressor_value.weight", p.ikn, p.iv[il] + (size_t) pos * IK);
            gemv(p, P + "indexer_compressor_gate.weight", p.ikn, p.ig[il] + (size_t) pos * IK);
            if ((pos + 1) % kpool == 0) {
                const int64_t g = pos / kpool, g0 = g * kpool;
                float* pk = p.pooled[il] + (size_t) g * IK;
                kn::glm_dsa_pool(p.iv[il] + (size_t) g0 * IK, p.ig[il] + (size_t) g0 * IK,
                                 f32(p, P + "indexer_compressor_ape.weight"), IK, kpool, 1, pk, p.st);
                k_pool_finish<<<1, 128, 0, p.st>>>(pk, f32(p, P + "indexer_compressor_norm.weight"), IK, eps, rope,
                                                   (float) g0, th);
            }
            gemv(p, P + "indexer.proj.weight", p.x, p.iw);
            k_scale<<<1, 32, 0, p.st>>>(p.iw, IH, 1.0f / std::sqrt((float) (IK * IH)));
            n_sel = select_kpool(p, il, pos);
            if (n_sel == 0) n_sel = (int) pos + 1, lo = 0;
            else cells = p.cells;
        }

        const DW& kb = p.w.at(P + "attn_k_b.weight");
        const DW& vb = p.w.at(P + "attn_v_b.weight");
        if (kb.type == T_Q8_0)
            k_absorb_q8<<<nb(H * kvl), 256, 0, p.st>>>((const uint8_t*) kb.d, p.q, QH, p.qabs, nope, kvl, H);
        else
            k_absorb_bf16<<<nb(H * kvl), 256, 0, p.st>>>((const uint16_t*) kb.d, p.q, QH, p.qabs, nope, kvl, H);
        k_attn<<<H, 256, 0, p.st>>>(p.qabs, p.q, QH, nope, p.kv[il], kvl, rope, cells, lo, n_sel,
                                     f32(p, P + "attn_sinks.weight"), 1.0f / std::sqrt((float) QH), p.o, p.sc, max_sel);
        if (vb.type == T_Q8_0)
            k_vb_q8<<<nb(H * vh), 256, 0, p.st>>>((const uint8_t*) vb.d, p.o, p.a, kvl, vh, H);
        else
            k_vb_bf16<<<nb(H * vh), 256, 0, p.st>>>((const uint16_t*) vb.d, p.o, p.a, kvl, vh, H);
        gemv(p, P + "attn_gate.weight", p.x, p.gate);
        k_sigmul<<<nb(H * vh), 256, 0, p.st>>>(p.a, p.gate, H * vh);
        gemv(p, P + "attn_output.weight", p.a, p.ao);
        k_rms<<<1, 1024, 0, p.st>>>(p.ao, f32(p, P + "post_attention_norm.weight"), p.y, E, eps);
    }

    void ffn(Part& p, int il) {
        const std::string P = "blk." + std::to_string(il) + ".", S = "-" + std::to_string(il);
        if (il < dense_lead) {
            gemv(p, P + "ffn_gate.weight", p.x, p.ga);
            gemv(p, P + "ffn_up.weight", p.x, p.up);
            k_swiglu<<<nb(n_ff), 256, 0, p.st>>>(p.ga, p.up, n_ff, INFINITY);
            gemv(p, P + "ffn_down.weight", p.ga, p.ffn);
        } else {
            gemv(p, P + "ffn_gate_inp.weight", p.x, p.logits_r);
            kn::router_sigmoid(p.logits_r, f32(p, P + "exp_probs_b.bias"), 1, n_exp, n_used, w_scale, true, p.eids, p.ew,
                               p.st);
            std::vector<int> ids(n_used);
            HY_CK(cudaMemcpyAsync(ids.data(), p.eids, sizeof(int) * n_used, cudaMemcpyDeviceToHost, p.st));
            gemv(p, P + "ffn_routed_down.weight", p.x, p.xl);
            HY_CK(cudaMemsetAsync(p.moe, 0, sizeof(float) * latent, p.st));
            HY_CK(cudaStreamSynchronize(p.st));
            const DW& G = p.w.at(P + "ffn_gate_exps.weight");
            const DW& U = p.w.at(P + "ffn_up_exps.weight");
            const DW& D = p.w.at(P + "ffn_down_exps.weight");
            for (int k = 0; k < n_used; ++k) {
                const size_t e = (size_t) ids[k];
                gemv(p, G, e * G.rb * G.ne[1], p.xl, p.ga, latent, n_ffx);
                gemv(p, U, e * U.rb * U.ne[1], p.xl, p.up, latent, n_ffx);
                k_swiglu<<<nb(n_ffx), 256, 0, p.st>>>(p.ga, p.up, n_ffx, clampv);
                gemv(p, D, e * D.rb * D.ne[1], p.ga, p.dn, n_ffx, latent);
                k_axpy_k<<<nb(latent), 256, 0, p.st>>>(p.moe, p.dn, latent, p.ew, k);
            }
            dump(p, "ffn_moe_out" + S, p.moe, latent);
            gemv(p, P + "ffn_routed_up.weight", p.moe, p.mu);
            gemv(p, P + "ffn_gate_shexp.weight", p.x, p.ga);
            gemv(p, P + "ffn_up_shexp.weight", p.x, p.up);
            k_swiglu<<<nb(n_ffx), 256, 0, p.st>>>(p.ga, p.up, n_ffx, INFINITY);
            gemv(p, P + "ffn_down_shexp.weight", p.ga, p.sh);
            dump(p, "ffn_shexp" + S, p.sh, E);
            k_add<<<nb(E), 256, 0, p.st>>>(p.mu, p.sh, p.ffn, E);
        }
        k_rms<<<1, 1024, 0, p.st>>>(p.ffn, f32(p, P + "post_ffw_norm.weight"), p.y, E, eps);
    }

    void layer(Part& p, int il, int64_t pos) {
        const std::string P = "blk." + std::to_string(il) + ".", S = "-" + std::to_string(il);
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) engram(p, (int) e, il, pos);
        const size_t XS = sizeof(float) * hc * E;
        HY_CK(cudaMemcpyAsync(p.R, p.X, XS, cudaMemcpyDeviceToDevice, p.st));
        k_hc_pre<<<1, 1024, 0, p.st>>>(p.X, f32(p, P + "hc_attn_fn.weight"), f32(p, P + "hc_attn_scale.weight"),
                                        f32(p, P + "hc_attn_base.weight"), E, hc, 2 * hc, eps, hc_eps, hc_mag, p.cur,
                                        p.post);
        k_rms<<<1, 1024, 0, p.st>>>(p.cur, f32(p, P + "attn_norm.weight"), p.x, E, eps);
        dump(p, "attn_norm" + S, p.x, E);
        attention(p, il, pos);
        dump(p, "attn_out" + S, p.y, E);
        k_hc_post<<<nb(hc * E), 256, 0, p.st>>>(p.X, p.R, p.y, p.post, E, hc);
        HY_CK(cudaMemcpyAsync(p.R, p.X, XS, cudaMemcpyDeviceToDevice, p.st));
        k_hc_pre<<<1, 1024, 0, p.st>>>(p.X, f32(p, P + "hc_ffn_fn.weight"), f32(p, P + "hc_ffn_scale.weight"),
                                        f32(p, P + "hc_ffn_base.weight"), E, hc, 2 * hc, eps, hc_eps, hc_mag, p.cur,
                                        p.post);
        k_rms<<<1, 1024, 0, p.st>>>(p.cur, f32(p, P + "ffn_norm.weight"), p.x, E, eps);
        ffn(p, il);
        dump(p, "ffn_out" + S, p.y, E);
        k_hc_post<<<nb(hc * E), 256, 0, p.st>>>(p.X, p.R, p.y, p.post, E, hc);
    }
};

Model::Model() : d_(new Impl) {}
Model::~Model() = default;
int Model::vocab() const { return d_->V; }

std::string Model::summary() const {
    std::string s = "hy4f: " + std::to_string(d_->L) + " layers on " + std::to_string(d_->parts.size()) + " GPUs:";
    for (const auto& p : d_->parts)
        s += " [gpu" + std::to_string(p.dev) + " " + std::to_string(p.l0) + ".." + std::to_string(p.l1 - 1) + "]";
    return s;
}

bool Model::load(const Options& o, std::string& err) {
    try {
        Impl& d = *d_;
        d.o = o;
        for (const auto& s : o.shards) d.files.emplace_back(new strata::GgufFile(s));
        const strata::MetaValue* a = d.files[0]->get("general.architecture");
        if (!a || a->s != "hy_v4_flash") throw std::runtime_error("hy4f: not a hy_v4_flash GGUF");
        d.geometry();
        d.build_parts();
        d.load_engram();
        for (size_t i = 0; i < d.parts.size(); ++i) d.load_part(d.parts[i], i == 0, i + 1 == d.parts.size());
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Model::step(int32_t token, int64_t pos, float* logits, std::string& err) {
    try {
        Impl& d = *d_;
        if (pos != (int64_t) d.hist.size()) throw std::runtime_error("hy4f: step out of order");
        if (pos >= d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hist.push_back(token);
        const int E = d.E, hc = d.hc;
        for (size_t i = 0; i < d.parts.size(); ++i) {
            Part& p = d.parts[i];
            HY_CK(cudaSetDevice(p.dev));
            if (i == 0) {
                const DW& emb = p.w.at("token_embd.weight");
                k_q8_0_row<<<1, 256, 0, p.st>>>((const uint8_t*) emb.d + (size_t) token * emb.rb, p.emb, E);
                k_rep<<<nb(E * hc), 256, 0, p.st>>>(p.emb, p.X, E, hc);
            } else {
                Part& prev = d.parts[i - 1];
                HY_CK(cudaStreamSynchronize(prev.st));
                HY_CK(cudaMemcpyPeerAsync(p.X, p.dev, prev.X, prev.dev, sizeof(float) * hc * E, p.st));
            }
            for (int il = p.l0; il < p.l1; ++il) d.layer(p, il, pos);
            if (i + 1 == d.parts.size()) {
                k_hc_pre<<<1, 1024, 0, p.st>>>(p.X, d.f32(p, "output_hc_fn.weight"), d.f32(p, "output_hc_scale.weight"),
                                                d.f32(p, "output_hc_base.weight"), E, hc, hc, d.eps, d.hc_eps, 0.0f,
                                                p.head_cur, p.post);
                k_rms<<<1, 1024, 0, p.st>>>(p.head_cur, d.f32(p, "output_norm.weight"), p.head_x, E, d.eps);
                d.dump(p, "result_norm", p.head_x, E);
                d.gemv(p, "output.weight", p.head_x, p.logits);
                if (logits) {
                    HY_CK(cudaMemcpyAsync(logits, p.logits, sizeof(float) * d.V, cudaMemcpyDeviceToHost, p.st));
                    HY_CK(cudaStreamSynchronize(p.st));
                    if (!d.o.dump_dir.empty()) {
                        std::FILE* f = std::fopen((d.o.dump_dir + "/logits.f32").c_str(), "ab");
                        if (f) std::fwrite(logits, 4, d.V, f), std::fclose(f);
                    }
                }
            }
        }
        HY_CK(cudaGetLastError());
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::hy4f
