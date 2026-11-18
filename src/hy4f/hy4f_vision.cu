// src/hy4f/hy4f_vision.cu - maya-hy4f vision: the Hy4 flash ViT (27 layers) + rmsnorm/conv projector on one GPU.
// Math and the splicing contract: hy4f_vision.hpp.  Precision: the ViT linears run on tensor cores with F16 operands
// (the BF16 weights converted to F16 at load, exact for |w| >= 2^-14) and F32 accumulation, the residual stream, norms,
// softmax and the whole projector (F32 weights in the GGUF) in F32.  Attention = cuBLAS strided-batched GEMMs over
// the 16 heads per query chunk (S in F32, P in F16), so the scratch stays bounded at 4 MP (16384 patches).
#include "hy4f_vision.hpp"
#include "hy4f_vision_pre.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::hy4f::vision {

namespace {

#define HV_CK(x)                                                                                                        \
    do {                                                                                                                \
        cudaError_t e_ = (x);                                                                                           \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)
#define HV_CB(x)                                                                                                        \
    do {                                                                                                                \
        cublasStatus_t s_ = (x);                                                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x) + ": cublas " + std::to_string((int) s_)); \
    } while (0)

enum : uint32_t { G_F32 = 0, G_F16 = 1, G_BF16 = 30 };

// ---------------------------------------------------------------- kernels

template <class T> __device__ __forceinline__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum / max (blockDim multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : 0.f;
    return warp_sum(r);
}
__device__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : -INFINITY;
    return warp_max(r);
}

__global__ void k_bf16_to_f16(const uint16_t* s, __half* d, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint32_t u = (uint32_t) s[i] << 16;
        d[i] = __float2half_rn(__uint_as_float(u));
    }
}
__global__ void k_f32_to_f16(const float* s, __half* d, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        d[i] = __float2half_rn(s[i]);
}

// im2col of the 16x16 patches + normalize: patches[p][c*P*P + kh*P + kw], p = py*pw + px
__global__ void k_patchify(const uint8_t* img, int W, int pw, int P, float* patches, float m0, float m1, float m2,
                           float s0, float s1, float s2) {
    const int p = blockIdx.x;
    const int py = p / pw, px = p % pw;
    const int K = 3 * P * P;
    for (int e = threadIdx.x; e < K; e += blockDim.x) {
        const int c = e / (P * P), r = e % (P * P), kh = r / P, kw = r % P;
        const int y = py * P + kh, x = px * P + kw;
        float v = (float) img[((size_t) y * W + x) * 3 + c] / 255.0f;
        const float m = c == 0 ? m0 : c == 1 ? m1 : m2;
        const float s = c == 0 ? s0 : c == 1 ? s1 : s2;
        v = (v - m) / s;
        patches[(size_t) p * K + e] = v;
    }
}

// x[p] = (x[p] + patch_bias) + bilinear(pos_grid) at (py, px); the llama.cpp CPU resample (sf = (n + 0.1) / grid)
__global__ void k_patch_bias_pos(float* x, const float* pb, const float* grid, int D, int pw, int ph, int ng) {
    const int p = blockIdx.x;
    const int y = p / pw, xx = p % pw;
    const float sx = (float) (pw + 0.1f) / (float) ng;
    const float sy = (float) (ph + 0.1f) / (float) ng;
    const float fy = ((float) y + 0.5f) / sy - 0.5f;
    int y0 = (int) floorf(fy), y1 = y0 + 1;
    y0 = min(max(y0, 0), ng - 1);
    y1 = min(max(y1, 0), ng - 1);
    const float wy1 = fminf(fmaxf(fy - (float) y0, 0.0f), 1.0f), wy0 = 1.0f - wy1;
    const float fx = ((float) xx + 0.5f) / sx - 0.5f;
    int x0 = (int) floorf(fx), x1 = x0 + 1;
    x0 = min(max(x0, 0), ng - 1);
    x1 = min(max(x1, 0), ng - 1);
    const float wx1 = fminf(fmaxf(fx - (float) x0, 0.0f), 1.0f), wx0 = 1.0f - wx1;
    const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
    const float* s00 = grid + ((size_t) y0 * ng + x0) * D;
    const float* s01 = grid + ((size_t) y0 * ng + x1) * D;
    const float* s10 = grid + ((size_t) y1 * ng + x0) * D;
    const float* s11 = grid + ((size_t) y1 * ng + x1) * D;
    float* r = x + (size_t) p * D;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const float pos = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
        r[c] = (r[c] + pb[c]) + pos;
    }
}

// x += pending bias (optional, written back); out = LayerNorm(x) * w + b  (F16).  One block per row, D <= 4 * blockDim.
__global__ void k_layernorm(float* x, const float* add, const float* w, const float* b, __half* out, int D, float eps) {
    __shared__ float sh[32];
    float* r = x + (size_t) blockIdx.x * D;
    float v[4];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        v[i] = 0.f;
        if (c < D) {
            float t = r[c];
            if (add) {
                t += add[c];
                r[c] = t;
            }
            v[i] = t;
            s += t;
        }
    }
    const float mean = block_sum(s, sh) / D;
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < D) {
            const float d = v[i] - mean;
            q += d * d;
        }
    }
    const float var = block_sum(q, sh) / D;
    const float scale = 1.0f / sqrtf(var + eps);
    __half* o = out + (size_t) blockIdx.x * D;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < D) o[c] = __float2half_rn((v[i] - mean) * scale * w[c] + b[c]);
    }
}

// x += pending bias; out = x / sqrt(mean(x^2) + eps) * w  (F32)
__global__ void k_rmsnorm(const float* x, const float* add, const float* w, float* out, int D, float eps) {
    __shared__ float sh[32];
    const float* r = x + (size_t) blockIdx.x * D;
    float v[4];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        v[i] = 0.f;
        if (c < D) {
            v[i] = add ? r[c] + add[c] : r[c];
            s += v[i] * v[i];
        }
    }
    const float scale = 1.0f / sqrtf(block_sum(s, sh) / D + eps);
    float* o = out + (size_t) blockIdx.x * D;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * blockDim.x;
        if (c < D) o[c] = v[i] * scale * w[c];
    }
}

// qkv [N][3D] F32 + bias -> q, k, v [N][D] F16
__global__ void k_qkv_split(const float* qkv, const float* b, __half* q, __half* k, __half* v, int N, int D) {
    const size_t n = (size_t) N * 3 * D;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int col = (int) (i % (3 * D));
        const size_t row = i / (3 * D);
        const __half h = __float2half_rn(qkv[i] + b[col]);
        const int part = col / D, c = col % D;
        __half* dst = part == 0 ? q : part == 1 ? k : v;
        dst[row * D + c] = h;
    }
}

// row softmax(s * scale) -> P (F16); one block per row of n keys
__global__ void k_softmax(const float* S, __half* P, int n, float scale) {
    __shared__ float sh[32];
    const float* s = S + (size_t) blockIdx.x * n;
    __half* p = P + (size_t) blockIdx.x * n;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, s[i] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(s[i] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = __float2half_rn(expf(s[i] * scale - mx) * inv);
}

// ot [head][query][dh] F32 -> o [query][head*dh] F16
__global__ void k_merge_heads(const float* ot, __half* o, int nq, int nh, int dh) {
    const size_t n = (size_t) nq * nh * dh;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int d = (int) (i % dh);
        const size_t r = i / dh;
        const int h = (int) (r % nh);
        const size_t q = r / nh;
        o[i] = __float2half_rn(ot[((size_t) h * nq + q) * dh + d]);
    }
}

// one-pass variant: the row (n <= NT * 16) is read once into registers (S is the attention's memory bottleneck)
template <int NT> __global__ void __launch_bounds__(NT) k_softmax_reg(const float* S, __half* P, int n, float scale) {
    __shared__ float sh[32];
    const float* s = S + (size_t) blockIdx.x * n;
    __half* p = P + (size_t) blockIdx.x * n;
    float v[16];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int i = threadIdx.x + j * NT;
        v[j] = i < n ? s[i] * scale : -INFINITY;
        mx = fmaxf(mx, v[j]);
    }
    mx = block_max(mx, sh);
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        v[j] = expf(v[j] - mx);  // exp(-inf) = 0 for the padding
        sum += v[j];
    }
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int i = threadIdx.x + j * NT;
        if (i < n) p[i] = __float2half_rn(v[j] * inv);
    }
}

// h = gelu_tanh(t + b) as F16 (ggml's GELU: 0.5 x (1 + tanh(sqrt(2/pi) x (1 + 0.044715 x^2))))
__global__ void k_bias_gelu_tanh(const float* t, const float* b, __half* h, size_t n, int F) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float x = t[i] + b[i % F];
        const float y = 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
        h[i] = __float2half_rn(y);
    }
}

// in place: t = gelu_erf(t + b)
__global__ void k_bias_gelu_erf(float* t, const float* b, size_t n, int F) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float x = t[i] + b[i % F];
        t[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440084436210485f));
    }
}

// 2x2 stride-2 conv im2col: g[m][c*4 + kh*2 + kw] = x[(2oy+kh)*pw + 2ox+kw][c], m = oy*ow + ox
__global__ void k_gather2x2(const float* x, float* g, int D, int pw, int ow) {
    const int m = blockIdx.x;
    const int oy = m / ow, ox = m % ow;
    float* dst = g + (size_t) m * D * 4;
    for (int e = threadIdx.x; e < D * 4; e += blockDim.x) {
        const int c = e >> 2, kh = (e >> 1) & 1, kw = e & 1;
        dst[e] = x[((size_t) (2 * oy + kh) * pw + 2 * ox + kw) * D + c];
    }
}

// out[oy*(ow+1) + ox] = ox < ow ? m2[oy*ow + ox] + b : newline
__global__ void k_emit(const float* m2, const float* b, const float* nl, float* out, int O, int ow) {
    const int t = blockIdx.x;
    const int oy = t / (ow + 1), ox = t % (ow + 1);
    float* o = out + (size_t) t * O;
    if (ox == ow) {
        for (int c = threadIdx.x; c < O; c += blockDim.x) o[c] = nl[c];
    } else {
        const float* s = m2 + ((size_t) oy * ow + ox) * O;
        for (int c = threadIdx.x; c < O; c += blockDim.x) o[c] = s[c] + b[c];
    }
}

int grid_for(size_t n, int bs = 256) { return (int) std::min<size_t>((n + bs - 1) / bs, 65535 * 4); }

template <class T> struct DevBuf {
    T* p = nullptr;
    size_t n = 0;
    void need(size_t m) {
        if (m <= n) return;
        if (p) cudaFree(p);
        p = nullptr;
        HV_CK(cudaMalloc(&p, m * sizeof(T)));
        n = m;
    }
    ~DevBuf() {
        if (p) cudaFree(p);
    }
};

}  // namespace

struct Encoder::Impl {
    Options opt;
    int D = 0, NH = 0, DH = 0, F = 0, L = 0, P = 0, merge = 2, O = 0, NG = 0;
    int min_pixels = 0, max_pixels = 0;
    float eps = 1e-5f, mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};

    cudaStream_t st = nullptr;
    cublasHandle_t cb = nullptr;
    std::vector<void*> allocs;

    float *patch_w = nullptr, *patch_b = nullptr, *pos_grid = nullptr;
    struct Layer {
        __half *wqkv, *wo, *wup, *wdown;
        float *bqkv, *bo, *bup, *bdown, *ln1w, *ln1b, *ln2w, *ln2b;
    };
    std::vector<Layer> layers;
    float *pre_norm = nullptr, *mm0w = nullptr, *mm0b = nullptr, *mm2w = nullptr, *mm2b = nullptr, *newline = nullptr;

    DevBuf<uint8_t> d_img;
    DevBuf<float> d_patches, d_x, d_tmp, d_S, d_ot, d_xr, d_g, d_m0, d_m2, d_out;
    DevBuf<__half> d_h, d_q, d_k, d_v, d_o, d_P;

    ~Impl() {
        if (opt.gpu >= 0) cudaSetDevice(opt.gpu);
        for (void* p : allocs) cudaFree(p);
        if (cb) cublasDestroy(cb);
        if (st) cudaStreamDestroy(st);
    }

    template <class T> T* dalloc(size_t n) {
        void* p = nullptr;
        HV_CK(cudaMalloc(&p, n * sizeof(T)));
        allocs.push_back(p);
        return (T*) p;
    }

    const strata::TensorInfo& T(const strata::GgufFile& g, const std::string& n, uint64_t elems) {
        const strata::TensorInfo* t = g.find(n);
        if (!t) throw std::runtime_error("hy4f_vision: missing tensor " + n);
        if (elems && t->elements() != elems)
            throw std::runtime_error("hy4f_vision: " + n + " has " + std::to_string(t->elements()) + " elements, want " +
                                     std::to_string(elems));
        if (t->type != G_F32 && t->type != G_F16 && t->type != G_BF16)
            throw std::runtime_error("hy4f_vision: " + n + " has unsupported type " + std::to_string(t->type));
        return *t;
    }

    // a tensor as F32 on the device (into dst when given)
    float* up_f32(const strata::GgufFile& g, const std::string& n, uint64_t elems, float* dst = nullptr) {
        const auto& t = T(g, n, elems);
        const size_t ne = t.elements();
        if (!dst) dst = dalloc<float>(ne);
        const uint8_t* src = g.tensor_data(t);
        if (t.type == G_F32) {
            HV_CK(cudaMemcpy(dst, src, ne * 4, cudaMemcpyHostToDevice));
        } else {
            std::vector<float> h(ne);
            for (size_t i = 0; i < ne; ++i) {
                uint16_t u;
                std::memcpy(&u, src + 2 * i, 2);
                if (t.type == G_BF16) {
                    const uint32_t b = (uint32_t) u << 16;
                    std::memcpy(&h[i], &b, 4);
                } else {
                    __half_raw hr;
                    hr.x = u;
                    h[i] = __half2float(__half(hr));
                }
            }
            HV_CK(cudaMemcpy(dst, h.data(), ne * 4, cudaMemcpyHostToDevice));
        }
        return dst;
    }

    // a tensor as F16 on the device at dst (staged through `stage`)
    void up_f16(const strata::GgufFile& g, const std::string& n, uint64_t elems, __half* dst, DevBuf<uint8_t>& stage) {
        const auto& t = T(g, n, elems);
        const size_t ne = t.elements();
        const uint8_t* src = g.tensor_data(t);
        if (t.type == G_F16) {
            HV_CK(cudaMemcpyAsync(dst, src, ne * 2, cudaMemcpyHostToDevice, st));
            HV_CK(cudaStreamSynchronize(st));
            return;
        }
        const size_t bytes = ne * (t.type == G_F32 ? 4 : 2);
        stage.need(bytes);
        // same stream as the conversion: a pageable cudaMemcpy may return before its DMA lands, and st is non-blocking
        HV_CK(cudaMemcpyAsync(stage.p, src, bytes, cudaMemcpyHostToDevice, st));
        if (t.type == G_BF16)
            k_bf16_to_f16<<<grid_for(ne), 256, 0, st>>>((const uint16_t*) stage.p, dst, ne);
        else
            k_f32_to_f16<<<grid_for(ne), 256, 0, st>>>((const float*) stage.p, dst, ne);
        HV_CK(cudaGetLastError());
        HV_CK(cudaStreamSynchronize(st));
    }

    uint64_t meta_u(const strata::GgufFile& g, const std::string& k) {
        const strata::MetaValue* v = g.get(k);
        if (!v || !v->is_num()) throw std::runtime_error("hy4f_vision: missing key " + k);
        return (uint64_t) v->num();
    }

    void load(const std::string& path) {
        HV_CK(cudaSetDevice(opt.gpu));
        HV_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        HV_CB(cublasCreate(&cb));
        HV_CB(cublasSetStream(cb, st));
        HV_CB(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));  // no TF32 for the F32 projector GEMMs

        strata::GgufFile g(path);
        const strata::MetaValue* pt = g.get("clip.projector_type");
        if (!pt || pt->s != "hy_v4_flash") throw std::runtime_error("hy4f_vision: projector_type is not hy_v4_flash");
        D = (int) meta_u(g, "clip.vision.embedding_length");
        NH = (int) meta_u(g, "clip.vision.attention.head_count");
        F = (int) meta_u(g, "clip.vision.feed_forward_length");
        L = (int) meta_u(g, "clip.vision.block_count");
        P = (int) meta_u(g, "clip.vision.patch_size");
        O = (int) meta_u(g, "clip.vision.projection_dim");
        if (g.get("clip.vision.spatial_merge_size")) merge = (int) meta_u(g, "clip.vision.spatial_merge_size");
        min_pixels = (int) meta_u(g, "clip.vision.image_min_pixels");
        max_pixels = (int) meta_u(g, "clip.vision.image_max_pixels");
        if (const strata::MetaValue* e = g.get("clip.vision.attention.layer_norm_epsilon")) eps = (float) e->num();
        if (const strata::MetaValue* m = g.get("clip.vision.image_mean"); m && m->items.size() == 3)
            for (int i = 0; i < 3; ++i) mean[i] = (float) m->items[i].num();
        if (const strata::MetaValue* s = g.get("clip.vision.image_std"); s && s->items.size() == 3)
            for (int i = 0; i < 3; ++i) stdv[i] = (float) s->items[i].num();
        DH = D / NH;
        if (merge != 2 || D % NH || D > 4 * 512 || (int64_t) D * NH == 0)
            throw std::runtime_error("hy4f_vision: unexpected geometry");
        const strata::TensorInfo* tp = g.find("v.position_embd.weight");
        if (!tp) throw std::runtime_error("hy4f_vision: missing v.position_embd.weight");
        const int64_t npos = (int64_t) (tp->elements() / D);
        NG = (int) std::lround(std::sqrt((double) npos));
        if ((int64_t) NG * NG != npos) throw std::runtime_error("hy4f_vision: position grid is not square");

        const int K = 3 * P * P;
        patch_w = up_f32(g, "v.patch_embd.weight", (uint64_t) K * D);
        patch_b = up_f32(g, "v.patch_embd.bias", D);
        pos_grid = up_f32(g, "v.position_embd.weight", (uint64_t) npos * D);

        DevBuf<uint8_t> stage;
        layers.resize(L);
        for (int il = 0; il < L; ++il) {
            const std::string b = "v.blk." + std::to_string(il) + ".";
            Layer& y = layers[il];
            y.wqkv = dalloc<__half>((size_t) 3 * D * D);
            up_f16(g, b + "attn_q.weight", (uint64_t) D * D, y.wqkv, stage);
            up_f16(g, b + "attn_k.weight", (uint64_t) D * D, y.wqkv + (size_t) D * D, stage);
            up_f16(g, b + "attn_v.weight", (uint64_t) D * D, y.wqkv + (size_t) 2 * D * D, stage);
            y.bqkv = dalloc<float>((size_t) 3 * D);
            up_f32(g, b + "attn_q.bias", D, y.bqkv);
            up_f32(g, b + "attn_k.bias", D, y.bqkv + D);
            up_f32(g, b + "attn_v.bias", D, y.bqkv + 2 * D);
            y.wo = dalloc<__half>((size_t) D * D);
            up_f16(g, b + "attn_out.weight", (uint64_t) D * D, y.wo, stage);
            y.bo = up_f32(g, b + "attn_out.bias", D);
            y.wup = dalloc<__half>((size_t) F * D);
            up_f16(g, b + "ffn_up.weight", (uint64_t) F * D, y.wup, stage);
            y.bup = up_f32(g, b + "ffn_up.bias", F);
            y.wdown = dalloc<__half>((size_t) D * F);
            up_f16(g, b + "ffn_down.weight", (uint64_t) D * F, y.wdown, stage);
            y.bdown = up_f32(g, b + "ffn_down.bias", D);
            y.ln1w = up_f32(g, b + "ln1.weight", D);
            y.ln1b = up_f32(g, b + "ln1.bias", D);
            y.ln2w = up_f32(g, b + "ln2.weight", D);
            y.ln2b = up_f32(g, b + "ln2.bias", D);
        }
        pre_norm = up_f32(g, "mm.pre_norm.weight", D);
        mm0w = up_f32(g, "mm.0.weight", (uint64_t) 4 * D * 4 * D);
        mm0b = up_f32(g, "mm.0.bias", 4 * D);
        mm2w = up_f32(g, "mm.2.weight", (uint64_t) 4 * D * O);
        mm2b = up_f32(g, "mm.2.bias", O);
        newline = up_f32(g, "v.image_newline", O);
        if (g.find("v.pre_ln.weight") || g.find("v.post_ln.weight"))
            throw std::runtime_error("hy4f_vision: unexpected pre/post LN in the encoder");
        HV_CK(cudaDeviceSynchronize());
        if (opt.verbose)
            std::fprintf(stderr, "hy4f_vision: D %d heads %d ffn %d layers %d patch %d proj %d grid %d px [%d, %d] eps %g\n",
                         D, NH, F, L, P, O, NG, min_pixels, max_pixels, eps);
    }

    // Y[N][M] (F32, ldy) = X[N][K] (F16) . W[M][K]^T (F16), Y = alpha XW^T + beta Y
    void gemm_h(const __half* X, const __half* W, float* Y, int N, int M, int K, float beta) {
        const float alpha = 1.f;
        HV_CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &beta, Y,
                           CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void gemm_f(const float* X, const float* W, float* Y, int N, int M, int K) {
        const float alpha = 1.f, beta = 0.f;
        HV_CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, K, X, K, &beta, Y, M));
    }

    // debug: HY4F_VISION_DUMP=<dir> saves seams as <dir>/t_<name>.f32 (llama.cpp cb names), + bias when given
    const char* dump_dir = std::getenv("HY4F_VISION_DUMP");
    template <class TT> void dump(const char* name, const TT* p, size_t n, const float* bias = nullptr, int cols = 0) {
        if (!dump_dir) return;
        HV_CK(cudaStreamSynchronize(st));
        std::vector<TT> h(n);
        HV_CK(cudaMemcpy(h.data(), p, n * sizeof(TT), cudaMemcpyDeviceToHost));
        std::vector<float> f(n), b;
        if (bias) {
            b.resize(cols);
            HV_CK(cudaMemcpy(b.data(), bias, cols * sizeof(float), cudaMemcpyDeviceToHost));
        }
        for (size_t i = 0; i < n; ++i) f[i] = (float) h[i] + (bias ? b[i % cols] : 0.f);
        std::FILE* fp = std::fopen((std::string(dump_dir) + "/t_" + name + ".f32").c_str(), "wb");
        if (fp) {
            std::fwrite(f.data(), 4, n, fp);
            std::fclose(fp);
        }
    }

    void run(const Image& im, Output& out) {
        const int W = im.w, H = im.h;
        const int pw = W / P, ph = H / P, N = pw * ph;
        const int ow = pw / merge, oh = ph / merge, M = ow * oh, NT = (ow + 1) * oh;
        const int K = 3 * P * P;

        d_img.need((size_t) W * H * 3);
        d_patches.need((size_t) N * K);
        d_x.need((size_t) N * D);
        d_tmp.need((size_t) N * std::max(3 * D, F));
        d_h.need((size_t) N * std::max(D, F));
        d_q.need((size_t) N * D);
        d_k.need((size_t) N * D);
        d_v.need((size_t) N * D);
        d_o.need((size_t) N * D);
        // query chunk: NH * qc * N * (4 + 2) bytes <= scratch
        int qc = (int) std::max<size_t>(1, opt.attn_scratch_bytes / ((size_t) NH * N * 6));
        qc = std::min(qc, N);
        if (qc > 64) qc = qc / 64 * 64;
        d_S.need((size_t) NH * qc * N);
        d_P.need((size_t) NH * qc * N);
        d_ot.need((size_t) NH * qc * DH);
        d_xr.need((size_t) N * D);
        d_g.need((size_t) M * 4 * D);
        d_m0.need((size_t) M * 4 * D);
        d_m2.need((size_t) M * O);
        d_out.need((size_t) NT * O);

        HV_CK(cudaMemcpyAsync(d_img.p, im.rgb.data(), (size_t) W * H * 3, cudaMemcpyHostToDevice, st));
        k_patchify<<<N, 256, 0, st>>>(d_img.p, W, pw, P, d_patches.p, mean[0], mean[1], mean[2], stdv[0], stdv[1], stdv[2]);
        gemm_f(d_patches.p, patch_w, d_x.p, N, D, K);
        k_patch_bias_pos<<<N, 256, 0, st>>>(d_x.p, patch_b, pos_grid, D, pw, ph, NG);
        dump("pos_embed", d_x.p, (size_t) N * D);

        const int ln_threads = std::max(32, ((D + 3) / 4 + 31) / 32 * 32);
        const float kq_scale = 1.0f / sqrtf((float) DH);
        const float one = 1.f, zero = 0.f;
        const float* pending = nullptr;
        for (int il = 0; il < L; ++il) {
            const Layer& y = layers[il];
            k_layernorm<<<N, ln_threads, 0, st>>>(d_x.p, pending, y.ln1w, y.ln1b, d_h.p, D, eps);
            if (il == 0) dump("layer_inp_normed-0", d_h.p, (size_t) N * D);
            gemm_h(d_h.p, y.wqkv, d_tmp.p, N, 3 * D, D, 0.f);
            k_qkv_split<<<grid_for((size_t) N * 3 * D), 256, 0, st>>>(d_tmp.p, y.bqkv, d_q.p, d_k.p, d_v.p, N, D);
            for (int q0 = 0; q0 < N; q0 += qc) {
                const int nq = std::min(qc, N - q0);
                // S_h (keys x queries, col-major) = K_h^T Q_h
                HV_CB(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, DH, &one, d_k.p, CUDA_R_16F, D,
                                                 DH, d_q.p + (size_t) q0 * D, CUDA_R_16F, D, DH, &zero, d_S.p,
                                                 CUDA_R_32F, N, (long long) N * nq, NH, CUBLAS_COMPUTE_32F,
                                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                if (N <= 256 * 16)
                    k_softmax_reg<256><<<NH * nq, 256, 0, st>>>(d_S.p, d_P.p, N, kq_scale);
                else if (N <= 512 * 16)
                    k_softmax_reg<512><<<NH * nq, 512, 0, st>>>(d_S.p, d_P.p, N, kq_scale);
                else if (N <= 1024 * 16)
                    k_softmax_reg<1024><<<NH * nq, 1024, 0, st>>>(d_S.p, d_P.p, N, kq_scale);
                else
                    k_softmax<<<NH * nq, 256, 0, st>>>(d_S.p, d_P.p, N, kq_scale);
                // O_h (dh x queries, F32 [head][query][dh]) = V_h (dh x keys) P_h (keys x queries); the heads' outputs
                // are kept apart (strided C must not interleave) and merged into o [N][D] F16 below
                HV_CB(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, DH, nq, N, &one, d_v.p, CUDA_R_16F, D,
                                                 DH, d_P.p, CUDA_R_16F, N, (long long) N * nq, &zero, d_ot.p,
                                                 CUDA_R_32F, DH, (long long) DH * nq, NH, CUBLAS_COMPUTE_32F,
                                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                k_merge_heads<<<grid_for((size_t) nq * D), 256, 0, st>>>(d_ot.p, d_o.p + (size_t) q0 * D, nq, NH, DH);
            }
            if (il == 0) {
                dump("Qcur-0", d_q.p, (size_t) N * D);
                dump("Kcur-0", d_k.p, (size_t) N * D);
                dump("Vcur-0", d_v.p, (size_t) N * D);
                dump("kqv_out-0", d_o.p, (size_t) N * D);
            }
            gemm_h(d_o.p, y.wo, d_x.p, N, D, D, 1.f);  // residual += attn_out (bias added by the next norm)
            if (il == 0) dump("ffn_inp-0", d_x.p, (size_t) N * D, y.bo, D);
            k_layernorm<<<N, ln_threads, 0, st>>>(d_x.p, y.bo, y.ln2w, y.ln2b, d_h.p, D, eps);
            gemm_h(d_h.p, y.wup, d_tmp.p, N, F, D, 0.f);
            k_bias_gelu_tanh<<<grid_for((size_t) N * F), 256, 0, st>>>(d_tmp.p, y.bup, d_h.p, (size_t) N * F, F);
            gemm_h(d_h.p, y.wdown, d_x.p, N, D, F, 1.f);  // residual += ffn (bias pending)
            pending = y.bdown;
            if (il == 0 || il == L - 1) {
                char nm[32];
                std::snprintf(nm, sizeof nm, "layer_out-%d", il);
                dump(nm, d_x.p, (size_t) N * D, pending, D);
            }
        }
        k_rmsnorm<<<N, ln_threads, 0, st>>>(d_x.p, pending, pre_norm, d_xr.p, D, eps);
        dump("mm_rms", d_xr.p, (size_t) N * D);
        k_gather2x2<<<M, 256, 0, st>>>(d_xr.p, d_g.p, D, pw, ow);
        gemm_f(d_g.p, mm0w, d_m0.p, M, 4 * D, 4 * D);
        k_bias_gelu_erf<<<grid_for((size_t) M * 4 * D), 256, 0, st>>>(d_m0.p, mm0b, (size_t) M * 4 * D, 4 * D);
        gemm_f(d_m0.p, mm2w, d_m2.p, M, O, 4 * D);
        k_emit<<<NT, 256, 0, st>>>(d_m2.p, mm2b, newline, d_out.p, O, ow);
        HV_CK(cudaGetLastError());
        HV_CK(cudaStreamSynchronize(st));

        out.n_tokens = NT;
        out.n_embd = O;
        out.ow = ow;
        out.oh = oh;
        out.d_embd = d_out.p;
    }
};

void image_positions(int ow, int oh, PosMode mode, std::vector<Pos>& pos, int& n_pos) {
    const int nx = ow + 1, n = nx * oh;
    pos.resize(n);
    if (mode == PosMode::Sequential) {
        for (int i = 0; i < n; ++i) pos[i] = {i, i, i};
        n_pos = n;
    } else {
        for (int i = 0; i < n; ++i) pos[i] = {0, i / nx, i % nx};
        n_pos = std::max(nx, oh);
    }
}

Encoder::Encoder() : d_(new Impl) {}
Encoder::~Encoder() = default;
int Encoder::n_embd() const { return d_->O; }

bool Encoder::load(const std::string& mmproj_path, const Options& o, std::string& err) {
    try {
        d_->opt = o;
        d_->load(mmproj_path);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void Encoder::target_size(int w, int h, int& tw, int& th) const {
    detail::smart_size(w, h, d_->P * d_->merge, d_->min_pixels, d_->max_pixels, tw, th);
}

bool Encoder::encode(const Image& img, Output& out, std::string& err, bool to_host, PosMode mode) {
    try {
        if (img.w <= 0 || img.h <= 0 || img.rgb.size() != (size_t) img.w * img.h * 3)
            throw std::runtime_error("hy4f_vision: bad image");
        HV_CK(cudaSetDevice(d_->opt.gpu));
        const auto t0 = std::chrono::steady_clock::now();
        int tw = 0, th = 0;
        target_size(img.w, img.h, tw, th);
        Image rs;
        detail::resize_lanczos(img, tw, th, rs);
        const auto t1 = std::chrono::steady_clock::now();
        d_->run(rs, out);
        const auto t2 = std::chrono::steady_clock::now();
        out.img_w = tw;
        out.img_h = th;
        out.ms_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out.ms_gpu = std::chrono::duration<double, std::milli>(t2 - t1).count();
        image_positions(out.ow, out.oh, mode, out.pos, out.n_pos);
        if (to_host) {
            out.embd.resize((size_t) out.n_tokens * out.n_embd);
            HV_CK(cudaMemcpy(out.embd.data(), out.d_embd, out.embd.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::hy4f::vision
