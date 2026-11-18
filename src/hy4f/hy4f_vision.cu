// src/hy4f/hy4f_vision.cu - maya-hy4f vision: the Hy4 flash ViT (27 layers) + rmsnorm/conv projector on one GPU,
// inside a caller-provided device arena.  Math and the splicing contract: hy4f_vision.hpp.
//
// Memory: the weights live in pinned host memory; per encode they are streamed through two slots of the arena
// (items: the patch-embedding conv, the 27 ViT layers, then the projector convs in row chunks), the copy of item i+2 waiting for the compute of
// item i (events), so H2D overlaps compute.  The 128x128 position grid is read in place (mapped pinned memory).
// Arena layout (256-byte aligned regions), N = patches, D = 1152:
//   slot0 | slot1                 2 x 29.1 MiB (one ViT layer: q|k|v|o|up|down F16 + biases / LN F32)
//   WS  4 MiB                     cuBLAS workspace
//   X   [N][D] F32                residual stream; later the RMS output, then the projector's mm.2 output
//   R   N * 8D bytes              u8 image + im2col patches -> qkv [N][3D] F16 + LN out / attention out [N][D] F16;
//                                 MLP token chunks (up F32 + gelu F16) reuse the qkv part; projector: g | m0 (F32)
//   SCR rest                      only for the non-flash fallback (HY4F_VISION_ATTN=gemm or head dim > 80):
//                                 S [hg][qc][N] F32 + P F16 + per-head out for hg heads x qc queries (linear in N)
// Attention: k_flash_attn (WMMA, two passes over the key tiles, shared-memory scratch only) - the arena needs no
// attention scratch: arena = 2 slots + 4 MiB + 13.5 KiB per patch.
// Precision: ViT linears on tensor cores with F16 operands (BF16 weights -> F16 at load, exact for |w| >= 2^-14) and
// F32 accumulation; residual stream, norms, softmax and the projector (F32 weights) in F32.
#include "hy4f_vision.hpp"
#include "hy4f_vision_pre.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>

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

constexpr size_t AL = 256;
constexpr size_t WS_BYTES = size_t(4) << 20;  // cuBLAS workspace, inside the arena
inline size_t al(size_t x) { return (x + AL - 1) / AL * AL; }

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

// ---- flash attention (WMMA F16 -> F32, sm_80+): one block = 128 queries (8 warps x 16 rows) of one head; keys in
// tiles of 32; head dim padded to 80.  Two passes over the keys: (1) row max / sum of exp, (2) P = exp(s - m) / l
// (F16, as the GEMM path) and O += P V in WMMA accumulators.  No score matrix in memory: the scratch is shared memory.
constexpr int FA_DP = 80, FA_LD = 88, FA_BK = 32, FA_WARPS = 8, FA_BQ = FA_WARPS * 16, FA_SLD = 36, FA_PLD = 40;
constexpr int FA_SMEM = 40960;  // max(Q staging 128x88 F16, K|V|S|P tiles, O staging 8x16x80 F32)

__global__ void __launch_bounds__(FA_WARPS * 32) k_flash_attn(const __half* qkv, __half* out, int N, int D, int dh,
                                                              float scale_log2) {
    using namespace nvcuda;
    __shared__ __align__(128) unsigned char sm[FA_SMEM];
    const int h = blockIdx.y, q0 = blockIdx.x * FA_BQ;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t ld = (size_t) 3 * D;
    const __half* Qg = qkv + (size_t) h * dh;
    const __half* Kg = qkv + D + (size_t) h * dh;
    const __half* Vg = qkv + 2 * (size_t) D + (size_t) h * dh;
    constexpr int CH = FA_DP / 8;  // 16-byte chunks per padded row
    const uint4 z = make_uint4(0, 0, 0, 0);

    __half* Qs = (__half*) sm;
    for (int i = threadIdx.x; i < FA_BQ * CH; i += blockDim.x) {
        const int r = i / CH, c = (i % CH) * 8;
        uint4 v = z;
        if (q0 + r < N && c < dh) v = *(const uint4*) (Qg + (size_t) (q0 + r) * ld + c);
        *(uint4*) (Qs + r * FA_LD + c) = v;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[FA_DP / 16];
#pragma unroll
    for (int k = 0; k < FA_DP / 16; ++k) wmma::load_matrix_sync(qf[k], Qs + warp * 16 * FA_LD + k * 16, FA_LD);
    __syncthreads();

    __half* Ks = (__half*) sm;
    __half* Vs = Ks + FA_BK * FA_LD;
    float* Ss = (float*) (sm + 2 * FA_BK * FA_LD * 2) + warp * 16 * FA_SLD;
    __half* Ps = (__half*) (sm + 2 * FA_BK * FA_LD * 2 + FA_WARPS * 16 * FA_SLD * 4) + warp * 16 * FA_PLD;
    auto load_tile = [&](const __half* G, __half* T_, int k0) {
        for (int i = threadIdx.x; i < FA_BK * CH; i += blockDim.x) {
            const int r = i / CH, c = (i % CH) * 8;
            uint4 v = z;
            if (k0 + r < N && c < dh) v = *(const uint4*) (G + (size_t) (k0 + r) * ld + c);
            *(uint4*) (T_ + r * FA_LD + c) = v;
        }
    };
    auto scores = [&]() {  // Ss[16][32] = Q_w K_tile^T (raw)
#pragma unroll
        for (int j = 0; j < FA_BK / 16; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
            wmma::fill_fragment(sf, 0.f);
#pragma unroll
            for (int k = 0; k < FA_DP / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Ks + j * 16 * FA_LD + k * 16, FA_LD);
                wmma::mma_sync(sf, qf[k], kf, sf);
            }
            wmma::store_matrix_sync(Ss + j * 16, sf, FA_SLD, wmma::mem_row_major);
        }
        __syncwarp();
    };
    const int r = lane >> 1, c0 = (lane & 1) * 16;

    float m = -INFINITY, l = 0.f;
    for (int k0 = 0; k0 < N; k0 += FA_BK) {
        __syncthreads();
        load_tile(Kg, Ks, k0);
        __syncthreads();
        scores();
        float v[16], tm = -INFINITY;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = k0 + c0 + j < N ? Ss[r * FA_SLD + c0 + j] * scale_log2 : -INFINITY;
            tm = fmaxf(tm, v[j]);
        }
        tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
        const float mn = fmaxf(m, tm);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) s += exp2f(v[j] - mn);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        l = l * exp2f(m - mn) + s;
        m = mn;
        __syncwarp();
    }
    const float inv_l = 1.0f / l;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[FA_DP / 16];
#pragma unroll
    for (int n = 0; n < FA_DP / 16; ++n) wmma::fill_fragment(of[n], 0.f);
    for (int k0 = 0; k0 < N; k0 += FA_BK) {
        __syncthreads();
        load_tile(Kg, Ks, k0);
        load_tile(Vg, Vs, k0);
        __syncthreads();
        scores();
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float p = k0 + c0 + j < N ? exp2f(Ss[r * FA_SLD + c0 + j] * scale_log2 - m) * inv_l : 0.f;
            Ps[r * FA_PLD + c0 + j] = __float2half_rn(p);
        }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < FA_BK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, Ps + kk * 16, FA_PLD);
#pragma unroll
            for (int n = 0; n < FA_DP / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, Vs + kk * 16 * FA_LD + n * 16, FA_LD);
                wmma::mma_sync(of[n], pf, vf, of[n]);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    float* Os = (float*) sm + warp * 16 * FA_DP;
#pragma unroll
    for (int n = 0; n < FA_DP / 16; ++n) wmma::store_matrix_sync(Os + n * 16, of[n], FA_DP, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * dh; i += 32) {
        const int rr = i / dh, cc = i % dh, q = q0 + warp * 16 + rr;
        if (q < N) out[(size_t) q * D + (size_t) h * dh + cc] = __float2half_rn(Os[rr * FA_DP + cc]);
    }
}

// y (F16, [rows][cols]) += b  (in place)
__global__ void k_add_bias_f16(__half* y, const float* b, size_t n, int cols) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        y[i] = __float2half_rn(__half2float(y[i]) + b[i % cols]);
}

// x (F32, [rows][cols]) += b
__global__ void k_add_bias_f32(float* x, const float* b, size_t n, int cols) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        x[i] += b[i % cols];
}

// ot [hg][nq][dh] F32 -> o[q][(h0 + h) * dh + d] F16 (o row stride ldo)
__global__ void k_merge_heads(const float* ot, __half* o, int nq, int hg, int dh, int h0, int ldo) {
    const size_t n = (size_t) nq * hg * dh;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int d = (int) (i % dh);
        const size_t r = i / dh;
        const int h = (int) (r % hg);
        const size_t q = r / hg;
        o[q * ldo + (size_t) (h0 + h) * dh + d] = __float2half_rn(ot[((size_t) h * nq + q) * dh + d]);
    }
}

int grid_for(size_t n, int bs = 256) { return (int) std::min<size_t>((n + bs - 1) / bs, 65535 * 4); }

struct PinnedBuf {
    void* p = nullptr;
    size_t n = 0;
    void alloc(size_t bytes) {
        HV_CK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
        n = bytes;
    }
    ~PinnedBuf() {
        if (p) cudaFreeHost(p);
    }
};

// host float -> F16 bits (round to nearest even), the same as __float2half_rn
inline uint16_t f2h(float f) {
    __half h = __float2half_rn(f);
    __half_raw r = h;
    return r.x;
}

}  // namespace

int image_tokens(int w, int h, int* ow, int* oh, int* tw, int* th, int min_pixels, int max_pixels, int patch, int merge) {
    int a = 0, b = 0;
    detail::smart_size(w, h, patch * merge, min_pixels, max_pixels, a, b);
    const int gw = a / (patch * merge), gh = b / (patch * merge);
    if (ow) *ow = gw;
    if (oh) *oh = gh;
    if (tw) *tw = a;
    if (th) *th = b;
    return (gw + 1) * gh;
}

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

struct Encoder::Impl {
    Options opt;
    int D = 0, NH = 0, DH = 0, F = 0, L = 0, P = 0, merge = 2, O = 0, NG = 0;
    int min_pixels = 0, max_pixels = 0;
    bool flash = true;  // WMMA flash attention (head dim <= 80, multiple of 8); else the tiled cuBLAS path
    float eps = 1e-5f, mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};

    // pinned host weights
    struct LayerOff {
        size_t wqkv, wo, wup, wdown, bqkv, bo, bup, bdown, ln1w, ln1b, ln2w, ln2b, bytes;
    } lo{};
    PinnedBuf layers;     // L x lo.bytes
    PinnedBuf proj;       // mm.0 [4D][4D] F32 then mm.2 [O][4D] F32
    PinnedBuf grid;       // position grid [NG*NG][D] F32, read in place by the GPU (mapped)
    PinnedBuf patchw;     // patch conv [D][3*P*P] F32 (streamed as the first item)
    const float* grid_dev = nullptr;
    size_t slot_bytes = 0;
    int rows_per_chunk = 0;  // projector rows per streamed item
    // small host params, uploaded once per GPU
    std::vector<float> patch_b, pre_norm, mm0b, mm2b, newline;

    struct Gpu {
        int dev = -1;
        cublasHandle_t cb = nullptr;
        cudaStream_t cs = nullptr;  // copy stream
        cudaEvent_t ready[2] = {}, freed[2] = {}, start = nullptr;
        void* fixed = nullptr;
        float *patch_b = nullptr, *pre_norm = nullptr, *mm0b = nullptr, *mm2b = nullptr,
              *newline = nullptr;
        size_t fixed_bytes = 0;
    };
    std::map<int, Gpu> gpus;

    ~Impl() {
        for (auto& [dev, g] : gpus) {
            cudaSetDevice(dev);
            if (g.cb) cublasDestroy(g.cb);
            for (int i = 0; i < 2; ++i) {
                if (g.ready[i]) cudaEventDestroy(g.ready[i]);
                if (g.freed[i]) cudaEventDestroy(g.freed[i]);
            }
            if (g.start) cudaEventDestroy(g.start);
            if (g.cs) cudaStreamDestroy(g.cs);
            if (g.fixed) cudaFree(g.fixed);
        }
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
    static float elem(const strata::TensorInfo& t, const uint8_t* src, size_t i) {
        if (t.type == G_F32) {
            float f;
            std::memcpy(&f, src + 4 * i, 4);
            return f;
        }
        uint16_t u;
        std::memcpy(&u, src + 2 * i, 2);
        if (t.type == G_BF16) {
            const uint32_t b = (uint32_t) u << 16;
            float f;
            std::memcpy(&f, &b, 4);
            return f;
        }
        __half_raw hr;
        hr.x = u;
        return __half2float(__half(hr));
    }
    void to_f32(const strata::GgufFile& g, const std::string& n, uint64_t elems, float* dst) {
        const auto& t = T(g, n, elems);
        const uint8_t* src = g.tensor_data(t);
        const size_t ne = t.elements();
        if (t.type == G_F32) std::memcpy(dst, src, ne * 4);
        else
            for (size_t i = 0; i < ne; ++i) dst[i] = elem(t, src, i);
    }
    std::vector<float> vec_f32(const strata::GgufFile& g, const std::string& n, uint64_t elems) {
        std::vector<float> v(elems);
        to_f32(g, n, elems, v.data());
        return v;
    }
    void to_f16(const strata::GgufFile& g, const std::string& n, uint64_t elems, uint16_t* dst) {
        const auto& t = T(g, n, elems);
        const uint8_t* src = g.tensor_data(t);
        const size_t ne = t.elements();
        if (t.type == G_F16) std::memcpy(dst, src, ne * 2);
        else
            for (size_t i = 0; i < ne; ++i) dst[i] = f2h(elem(t, src, i));
    }
    uint64_t meta_u(const strata::GgufFile& g, const std::string& k) {
        const strata::MetaValue* v = g.get(k);
        if (!v || !v->is_num()) throw std::runtime_error("hy4f_vision: missing key " + k);
        return (uint64_t) v->num();
    }

    void load(const std::string& path) {
        int ndev = 0;
        HV_CK(cudaGetDeviceCount(&ndev));  // a context for cudaHostAlloc
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
        DH = NH ? D / NH : 0;
        flash = DH <= FA_DP && DH % 8 == 0 && D % 8 == 0;
        if (const char* e = std::getenv("HY4F_VISION_ATTN"); e && std::string(e) == "gemm") flash = false;
        // the arena layout relies on: 2x2 merge, im2col patches and the u8 image fit R, the mm.2 output fits X
        if (merge != 2 || D % NH || D > 4 * 1024 || 3 * P * P * 4 + P * P * 3 > 8 * D || O > 4 * D)
            throw std::runtime_error("hy4f_vision: unexpected geometry");
        const strata::TensorInfo* tp = g.find("v.position_embd.weight");
        if (!tp) throw std::runtime_error("hy4f_vision: missing v.position_embd.weight");
        const int64_t npos = (int64_t) (tp->elements() / D);
        NG = (int) std::lround(std::sqrt((double) npos));
        if ((int64_t) NG * NG != npos) throw std::runtime_error("hy4f_vision: position grid is not square");
        if (g.find("v.pre_ln.weight") || g.find("v.post_ln.weight"))
            throw std::runtime_error("hy4f_vision: unexpected pre/post LN in the encoder");

        // one layer blob: F16 q|k|v|o|up|down, then F32 biases and LN
        size_t o = 0;
        auto put = [&](size_t bytes) { size_t r = o; o += al(bytes); return r; };
        lo.wqkv = put((size_t) 3 * D * D * 2);
        lo.wo = put((size_t) D * D * 2);
        lo.wup = put((size_t) F * D * 2);
        lo.wdown = put((size_t) D * F * 2);
        lo.bqkv = put((size_t) 3 * D * 4);
        lo.bo = put((size_t) D * 4);
        lo.bup = put((size_t) F * 4);
        lo.bdown = put((size_t) D * 4);
        lo.ln1w = put((size_t) D * 4);
        lo.ln1b = put((size_t) D * 4);
        lo.ln2w = put((size_t) D * 4);
        lo.ln2b = put((size_t) D * 4);
        lo.bytes = o;
        slot_bytes = lo.bytes;
        rows_per_chunk = (int) (slot_bytes / ((size_t) 4 * D * 4));

        layers.alloc((size_t) L * lo.bytes);
        std::vector<std::string> errs(L);
        auto fill = [&](int il) {
            try {
                uint8_t* b = (uint8_t*) layers.p + (size_t) il * lo.bytes;
                const std::string n = "v.blk." + std::to_string(il) + ".";
                const uint64_t DD = (uint64_t) D * D;
                to_f16(g, n + "attn_q.weight", DD, (uint16_t*) (b + lo.wqkv));
                to_f16(g, n + "attn_k.weight", DD, (uint16_t*) (b + lo.wqkv) + DD);
                to_f16(g, n + "attn_v.weight", DD, (uint16_t*) (b + lo.wqkv) + 2 * DD);
                to_f16(g, n + "attn_out.weight", DD, (uint16_t*) (b + lo.wo));
                to_f16(g, n + "ffn_up.weight", (uint64_t) F * D, (uint16_t*) (b + lo.wup));
                to_f16(g, n + "ffn_down.weight", (uint64_t) D * F, (uint16_t*) (b + lo.wdown));
                to_f32(g, n + "attn_q.bias", D, (float*) (b + lo.bqkv));
                to_f32(g, n + "attn_k.bias", D, (float*) (b + lo.bqkv) + D);
                to_f32(g, n + "attn_v.bias", D, (float*) (b + lo.bqkv) + 2 * D);
                to_f32(g, n + "attn_out.bias", D, (float*) (b + lo.bo));
                to_f32(g, n + "ffn_up.bias", F, (float*) (b + lo.bup));
                to_f32(g, n + "ffn_down.bias", D, (float*) (b + lo.bdown));
                to_f32(g, n + "ln1.weight", D, (float*) (b + lo.ln1w));
                to_f32(g, n + "ln1.bias", D, (float*) (b + lo.ln1b));
                to_f32(g, n + "ln2.weight", D, (float*) (b + lo.ln2w));
                to_f32(g, n + "ln2.bias", D, (float*) (b + lo.ln2b));
            } catch (const std::exception& e) {
                errs[il] = e.what();
            }
        };
        {
            std::vector<std::thread> th;
            const int nt = std::max(1, std::min<int>(L, 16));
            for (int t = 0; t < nt; ++t)
                th.emplace_back([&, t] {
                    for (int il = t; il < L; il += nt) fill(il);
                });
            for (auto& x : th) x.join();
        }
        for (auto& e : errs)
            if (!e.empty()) throw std::runtime_error(e);

        const size_t K4 = (size_t) 4 * D;
        proj.alloc((K4 * K4 + (size_t) O * K4) * 4);
        to_f32(g, "mm.0.weight", K4 * K4, (float*) proj.p);
        to_f32(g, "mm.2.weight", (uint64_t) O * K4, (float*) proj.p + K4 * K4);
        grid.alloc((size_t) npos * D * 4);
        to_f32(g, "v.position_embd.weight", (uint64_t) npos * D, (float*) grid.p);
        void* gd = nullptr;
        HV_CK(cudaHostGetDevicePointer(&gd, grid.p, 0));
        grid_dev = (const float*) gd;

        patchw.alloc((size_t) 3 * P * P * D * 4);
        to_f32(g, "v.patch_embd.weight", (uint64_t) 3 * P * P * D, (float*) patchw.p);
        if (patchw.n > slot_bytes) throw std::runtime_error("hy4f_vision: patch conv larger than a slot");
        patch_b = vec_f32(g, "v.patch_embd.bias", D);
        pre_norm = vec_f32(g, "mm.pre_norm.weight", D);
        mm0b = vec_f32(g, "mm.0.bias", K4);
        mm2b = vec_f32(g, "mm.2.bias", O);
        newline = vec_f32(g, "v.image_newline", O);
        if (opt.verbose)
            std::fprintf(stderr,
                         "hy4f_vision: D %d heads %d ffn %d layers %d patch %d proj %d grid %d px [%d, %d] eps %g; pinned "
                         "%.0f MiB, slot %.1f MiB\n",
                         D, NH, F, L, P, O, NG, min_pixels, max_pixels, eps,
                         (layers.n + proj.n + grid.n + patchw.n) / 1048576.0, slot_bytes / 1048576.0);
    }

    Gpu& gpu(int dev) {
        auto it = gpus.find(dev);
        if (it != gpus.end()) return it->second;
        Gpu& g = gpus[dev];
        g.dev = dev;
        HV_CK(cudaSetDevice(dev));

        HV_CK(cudaStreamCreateWithFlags(&g.cs, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) {
            HV_CK(cudaEventCreateWithFlags(&g.ready[i], cudaEventDisableTiming));
            HV_CK(cudaEventCreateWithFlags(&g.freed[i], cudaEventDisableTiming));
        }
        HV_CK(cudaEventCreateWithFlags(&g.start, cudaEventDisableTiming));
        const size_t K4 = (size_t) 4 * D;
        size_t o = 0;
        auto put = [&](size_t bytes) { size_t r = o; o += al(bytes); return r; };
        const size_t opb = put(patch_b.size() * 4), opn = put(pre_norm.size() * 4), om0 = put(K4 * 4),
                     om2 = put((size_t) O * 4), onl = put((size_t) O * 4);
        g.fixed_bytes = o;
        HV_CK(cudaMalloc(&g.fixed, o));
        uint8_t* b = (uint8_t*) g.fixed;
        g.patch_b = (float*) (b + opb);
        g.pre_norm = (float*) (b + opn);
        g.mm0b = (float*) (b + om0);
        g.mm2b = (float*) (b + om2);
        g.newline = (float*) (b + onl);
        HV_CK(cudaMemcpy(g.patch_b, patch_b.data(), patch_b.size() * 4, cudaMemcpyHostToDevice));
        HV_CK(cudaMemcpy(g.pre_norm, pre_norm.data(), pre_norm.size() * 4, cudaMemcpyHostToDevice));
        HV_CK(cudaMemcpy(g.mm0b, mm0b.data(), mm0b.size() * 4, cudaMemcpyHostToDevice));
        HV_CK(cudaMemcpy(g.mm2b, mm2b.data(), mm2b.size() * 4, cudaMemcpyHostToDevice));
        HV_CK(cudaMemcpy(g.newline, newline.data(), newline.size() * 4, cudaMemcpyHostToDevice));
        if (opt.verbose) std::fprintf(stderr, "hy4f_vision: GPU %d fixed allocation %.1f KiB\n", dev, o / 1024.0);
        return g;
    }

    // arena plan for N patches
    struct Plan {
        size_t slot[2], ws, x, r, scr, scr_bytes, total;
    };
    Plan plan(int N, size_t arena_bytes) const {
        Plan p{};
        size_t o = 0;
        p.slot[0] = o, o += al(slot_bytes);
        p.slot[1] = o, o += al(slot_bytes);
        p.ws = o, o += WS_BYTES;
        p.x = o, o += al((size_t) N * D * 4);
        p.r = o, o += al((size_t) N * 8 * D);
        p.scr = o;
        p.scr_bytes = arena_bytes > o ? arena_bytes - o : 0;
        p.total = o;
        return p;
    }
    size_t attn_bytes(int N, int hg, int qc) const {
        return al((size_t) hg * qc * N * 4) + al((size_t) hg * qc * N * 2) + al((size_t) hg * qc * DH * 4);
    }
    // the attention tile for a scratch size: all heads if possible, queries in multiples of 16 (64 above 64)
    bool attn_tile(int N, size_t scr, int& hg, int& qc) const {
        const int qmin = std::min(N, 16);
        for (hg = NH; hg >= 1; --hg) {
            if (NH % hg) continue;
            if (attn_bytes(N, hg, qmin) <= scr) break;
        }
        if (hg < 1) return false;
        int lo_q = qmin, hi_q = N;
        while (lo_q < hi_q) {  // largest qc that fits
            const int mid = (lo_q + hi_q + 1) / 2;
            if (attn_bytes(N, hg, mid) <= scr) lo_q = mid;
            else hi_q = mid - 1;
        }
        qc = lo_q;
        if (qc < N) qc = qc > 64 ? qc / 64 * 64 : std::max(qmin, qc / 16 * 16);
        return true;
    }

    // Y[N][M] (ldy) = X[N][K] (F16, ldx) . W[M][K]^T (F16), C type ct
    void gemm_h(cublasHandle_t cb, const __half* X, int ldx, const __half* W, void* Y, cudaDataType ct, int ldy, int N,
                int M, int K, float beta) {
        const float alpha = 1.f;
        HV_CB(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, ldx, &beta, Y,
                           ct, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void gemm_f(cublasHandle_t cb, const float* X, const float* W, float* Y, int ldy, int N, int M, int K) {
        const float alpha = 1.f, beta = 0.f;
        HV_CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, K, X, K, &beta, Y, ldy));
    }

    void run(const Image& im, Output& out, const Arena& ar, float* dst) {
        const int W = im.w, H = im.h;
        const int pw = W / P, ph = H / P, N = pw * ph;
        const int ow = pw / merge, oh = ph / merge, M = ow * oh, NT = (ow + 1) * oh;
        const int K = 3 * P * P, K4 = 4 * D;
        const Plan pl = plan(N, ar.bytes);
        int hg = NH, qc = FA_BQ;
        if (ar.bytes < pl.total || (!flash && !attn_tile(N, pl.scr_bytes, hg, qc)))
            throw std::runtime_error("hy4f_vision: arena too small: " + std::to_string(ar.bytes) + " bytes, need >= " +
                                     std::to_string(pl.total + (flash ? 0 : attn_bytes(N, 1, std::min(N, 16)))));
        Gpu& g = gpu(ar.gpu);
        HV_CK(cudaSetDevice(ar.gpu));
        if (!ar.cublas && !g.cb) {
            size_t f0 = 0, f1 = 0, tot = 0;
            cudaMemGetInfo(&f0, &tot);
            HV_CB(cublasCreate(&g.cb));
            HV_CB(cublasSetMathMode(g.cb, CUBLAS_DEFAULT_MATH));  // no TF32 for the F32 projector GEMMs
            cudaMemGetInfo(&f1, &tot);
            if (opt.verbose)
                std::fprintf(stderr, "hy4f_vision: GPU %d own cublasCreate %.1f MiB\n", ar.gpu, ((double) f0 - f1) / 1048576.0);
        }
        cudaStream_t st = ar.stream;
        uint8_t* A = (uint8_t*) ar.dev_ptr;
        cublasHandle_t cb = ar.cublas ? (cublasHandle_t) ar.cublas : g.cb;
        cudaStream_t caller_cb_stream = nullptr;
        if (ar.cublas) HV_CB(cublasGetStream(cb, &caller_cb_stream));
        HV_CB(cublasSetStream(cb, st));
        HV_CB(cublasSetWorkspace(cb, A + pl.ws, WS_BYTES));  // after SetStream, which resets it
        struct Restore {  // a caller's handle gets its stream back (and with it cuBLAS's default workspace)
            cublasHandle_t cb;
            bool own;
            cudaStream_t s;
            ~Restore() {
                if (!own) cublasSetStream(cb, s);
            }
        } restore{cb, ar.cublas == nullptr, caller_cb_stream};

        uint8_t* slot[2] = {A + pl.slot[0], A + pl.slot[1]};
        float* X = (float*) (A + pl.x);
        uint8_t* R = A + pl.r;
        __half* QKV = (__half*) R;                           // [N][3D]
        __half* Hh = (__half*) (R + (size_t) N * 3 * D * 2);  // [N][D]: LN output, attention output
        uint8_t* SCR = A + pl.scr;
        float* Sbuf = (float*) SCR;
        __half* Pbuf = (__half*) (SCR + al((size_t) hg * qc * N * 4));
        float* OT = (float*) (SCR + al((size_t) hg * qc * N * 4) + al((size_t) hg * qc * N * 2));

        // the streamed items: L layers, then mm.0 / mm.2 in row chunks
        struct Item {
            const void* host;
            size_t bytes;
        };
        std::vector<Item> items;
        items.push_back({patchw.p, patchw.n});
        for (int il = 0; il < L; ++il) items.push_back({(const uint8_t*) layers.p + (size_t) il * lo.bytes, lo.bytes});
        const float* mm0h = (const float*) proj.p;
        const float* mm2h = mm0h + (size_t) K4 * K4;
        const int n_mm0 = (K4 + rows_per_chunk - 1) / rows_per_chunk, n_mm2 = (O + rows_per_chunk - 1) / rows_per_chunk;
        for (int c = 0; c < n_mm0; ++c) {
            const int r0 = c * rows_per_chunk, r1 = std::min(K4, r0 + rows_per_chunk);
            items.push_back({mm0h + (size_t) r0 * K4, (size_t) (r1 - r0) * K4 * 4});
        }
        for (int c = 0; c < n_mm2; ++c) {
            const int r0 = c * rows_per_chunk, r1 = std::min(O, r0 + rows_per_chunk);
            items.push_back({mm2h + (size_t) r0 * K4, (size_t) (r1 - r0) * K4 * 4});
        }
        const int n_items = (int) items.size();
        // slots are free once the caller's prior work on the arena is done
        HV_CK(cudaEventRecord(g.freed[0], st));
        HV_CK(cudaEventRecord(g.freed[1], st));
        auto issue = [&](int i) {
            const int s = i & 1;
            HV_CK(cudaStreamWaitEvent(g.cs, g.freed[s], 0));
            HV_CK(cudaMemcpyAsync(slot[s], items[i].host, items[i].bytes, cudaMemcpyHostToDevice, g.cs));
            HV_CK(cudaEventRecord(g.ready[s], g.cs));
        };
        auto acquire = [&](int i) -> uint8_t* {
            HV_CK(cudaStreamWaitEvent(st, g.ready[i & 1], 0));
            return slot[i & 1];
        };
        auto release = [&](int i) {
            HV_CK(cudaEventRecord(g.freed[i & 1], st));
            if (i + 2 < n_items) issue(i + 2);
        };
        issue(0);
        if (n_items > 1) issue(1);

        // patch embedding + position grid (in R: im2col F32 at 0, the u8 image behind it)
        uint8_t* img_d = R + (size_t) N * K * 4;
        HV_CK(cudaMemcpyAsync(img_d, im.rgb.data(), (size_t) W * H * 3, cudaMemcpyHostToDevice, st));
        float* patches = (float*) R;
        k_patchify<<<N, 256, 0, st>>>(img_d, W, pw, P, patches, mean[0], mean[1], mean[2], stdv[0], stdv[1], stdv[2]);
        gemm_f(cb, patches, (const float*) acquire(0), X, D, N, D, K);
        release(0);
        k_patch_bias_pos<<<N, 256, 0, st>>>(X, g.patch_b, grid_dev, D, pw, ph, NG);

        const int ln_threads = std::max(32, ((D + 3) / 4 + 31) / 32 * 32);
        const float kq_scale = 1.0f / sqrtf((float) DH);
        const float one = 1.f, zero = 0.f;
        const int T = (int) std::max<size_t>(1, std::min<size_t>(N, (size_t) N * 3 * D * 2 / ((size_t) F * 6 + 512)));
        for (int il = 0; il < L; ++il) {
            uint8_t* b = acquire(1 + il);
            const __half* wqkv = (const __half*) (b + lo.wqkv);
            const __half* wo = (const __half*) (b + lo.wo);
            const __half* wup = (const __half*) (b + lo.wup);
            const __half* wdown = (const __half*) (b + lo.wdown);
            const float *bqkv = (const float*) (b + lo.bqkv), *bo = (const float*) (b + lo.bo),
                        *bup = (const float*) (b + lo.bup), *bdown = (const float*) (b + lo.bdown);
            k_layernorm<<<N, ln_threads, 0, st>>>(X, nullptr, (const float*) (b + lo.ln1w), (const float*) (b + lo.ln1b),
                                                  Hh, D, eps);
            gemm_h(cb, Hh, D, wqkv, QKV, CUDA_R_16F, 3 * D, N, 3 * D, D, 0.f);
            k_add_bias_f16<<<grid_for((size_t) N * 3 * D), 256, 0, st>>>(QKV, bqkv, (size_t) N * 3 * D, 3 * D);
            const __half *Q = QKV, *Kp = QKV + D, *V = QKV + 2 * D;
            if (flash) {
                k_flash_attn<<<dim3((N + FA_BQ - 1) / FA_BQ, NH), FA_WARPS * 32, 0, st>>>(
                    QKV, Hh, N, D, DH, kq_scale * 1.4426950408889634f);
            } else {
                for (int h0 = 0; h0 < NH; h0 += hg)
                    for (int q0 = 0; q0 < N; q0 += qc) {
                        const int nq = std::min(qc, N - q0);
                        // S_h (keys x queries, col-major) = K_h^T Q_h
                        HV_CB(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, DH, &one, Kp + h0 * DH,
                                                         CUDA_R_16F, 3 * D, DH, Q + (size_t) q0 * 3 * D + h0 * DH, CUDA_R_16F,
                                                         3 * D, DH, &zero, Sbuf, CUDA_R_32F, N, (long long) N * nq, hg,
                                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        const int rows = hg * nq;
                        if (N <= 256 * 16) k_softmax_reg<256><<<rows, 256, 0, st>>>(Sbuf, Pbuf, N, kq_scale);
                        else if (N <= 512 * 16) k_softmax_reg<512><<<rows, 512, 0, st>>>(Sbuf, Pbuf, N, kq_scale);
                        else if (N <= 1024 * 16) k_softmax_reg<1024><<<rows, 1024, 0, st>>>(Sbuf, Pbuf, N, kq_scale);
                        else k_softmax<<<rows, 256, 0, st>>>(Sbuf, Pbuf, N, kq_scale);
                        // O_h (dh x queries) = V_h (dh x keys) P_h (keys x queries), heads kept apart in OT
                        HV_CB(cublasGemmStridedBatchedEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, DH, nq, N, &one, V + h0 * DH,
                                                         CUDA_R_16F, 3 * D, DH, Pbuf, CUDA_R_16F, N, (long long) N * nq, &zero,
                                                         OT, CUDA_R_32F, DH, (long long) DH * nq, hg, CUBLAS_COMPUTE_32F,
                                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        k_merge_heads<<<grid_for((size_t) nq * hg * DH), 256, 0, st>>>(OT, Hh + (size_t) q0 * D, nq, hg, DH,
                                                                                     h0, D);
                    }
            }
            gemm_h(cb, Hh, D, wo, X, CUDA_R_32F, D, N, D, D, 1.f);  // residual += attn_out (bias in the next norm)
            k_layernorm<<<N, ln_threads, 0, st>>>(X, bo, (const float*) (b + lo.ln2w), (const float*) (b + lo.ln2b), Hh, D,
                                                  eps);
            float* up = (float*) R;  // the qkv part of R is free now
            __half* uph = (__half*) (R + al((size_t) T * F * 4));
            for (int t0 = 0; t0 < N; t0 += T) {
                const int nt = std::min(T, N - t0);
                gemm_h(cb, Hh + (size_t) t0 * D, D, wup, up, CUDA_R_32F, F, nt, F, D, 0.f);
                k_bias_gelu_tanh<<<grid_for((size_t) nt * F), 256, 0, st>>>(up, bup, uph, (size_t) nt * F, F);
                gemm_h(cb, uph, F, wdown, X + (size_t) t0 * D, CUDA_R_32F, D, nt, D, F, 1.f);
            }
            k_add_bias_f32<<<grid_for((size_t) N * D), 256, 0, st>>>(X, bdown, (size_t) N * D, D);
            release(1 + il);
        }

        // projector: RMS in place, 2x2 gather into R, mm.0 (row chunks) -> m0 in R, gelu, mm.2 (row chunks) -> X
        k_rmsnorm<<<N, ln_threads, 0, st>>>(X, nullptr, g.pre_norm, X, D, eps);
        float* G = (float*) R;
        float* M0 = G + (size_t) M * K4;
        k_gather2x2<<<M, 256, 0, st>>>(X, G, D, pw, ow);
        int it = 1 + L;
        for (int c = 0; c < n_mm0; ++c, ++it) {
            const int r0 = c * rows_per_chunk, r1 = std::min(K4, r0 + rows_per_chunk);
            const float* w = (const float*) acquire(it);
            gemm_f(cb, G, w, M0 + r0, K4, M, r1 - r0, K4);
            release(it);
        }
        k_bias_gelu_erf<<<grid_for((size_t) M * K4), 256, 0, st>>>(M0, g.mm0b, (size_t) M * K4, K4);
        float* M2 = X;
        for (int c = 0; c < n_mm2; ++c, ++it) {
            const int r0 = c * rows_per_chunk, r1 = std::min(O, r0 + rows_per_chunk);
            const float* w = (const float*) acquire(it);
            gemm_f(cb, M0, w, M2 + r0, O, M, r1 - r0, K4);
            release(it);
        }
        float* outp = dst ? dst : (float*) R;  // host output staged in R (g / m0 are consumed)
        k_emit<<<NT, 256, 0, st>>>(M2, g.mm2b, g.newline, outp, O, ow);
        HV_CK(cudaGetLastError());
        if (!dst) {
            out.embd.resize((size_t) NT * O);
            HV_CK(cudaMemcpyAsync(out.embd.data(), outp, out.embd.size() * 4, cudaMemcpyDeviceToHost, st));
        }
        HV_CK(cudaStreamSynchronize(st));
        HV_CK(cudaStreamSynchronize(g.cs));
        out.n_tokens = NT;
        out.n_embd = O;
        out.ow = ow;
        out.oh = oh;
        out.attn_heads = hg;
        out.attn_queries = qc;
    }
};

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

int Encoder::n_tokens(int w, int h, int* ow, int* oh) const {
    return image_tokens(w, h, ow, oh, nullptr, nullptr, d_->min_pixels, d_->max_pixels, d_->P, d_->merge);
}

size_t Encoder::arena_bytes(int w, int h, bool fast) const {
    int tw = 0, th = 0;
    target_size(w, h, tw, th);
    const int N = (tw / d_->P) * (th / d_->P);
    const auto pl = d_->plan(N, 0);
    if (d_->flash) return pl.total;  // flash attention: no scratch in the arena
    const int hg = fast ? d_->NH : 1, qc = std::min(N, fast ? 64 : 16);
    return pl.total + d_->attn_bytes(N, hg, qc);
}

bool Encoder::encode(const Image& img, Output& out, std::string& err, const Arena& arena, float* dst, PosMode mode) {
    try {
        if (img.w <= 0 || img.h <= 0 || img.rgb.size() != (size_t) img.w * img.h * 3)
            throw std::runtime_error("hy4f_vision: bad image");
        if (!arena.dev_ptr) throw std::runtime_error("hy4f_vision: no arena");
        const auto t0 = std::chrono::steady_clock::now();
        int tw = 0, th = 0;
        target_size(img.w, img.h, tw, th);
        Image rs;
        detail::resize_lanczos(img, tw, th, rs);
        const auto t1 = std::chrono::steady_clock::now();
        d_->run(rs, out, arena, dst);
        const auto t2 = std::chrono::steady_clock::now();
        out.img_w = tw;
        out.img_h = th;
        out.ms_preprocess = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out.ms_gpu = std::chrono::duration<double, std::milli>(t2 - t1).count();
        image_positions(out.ow, out.oh, mode, out.pos, out.n_pos);
        if (dst) out.embd.clear();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

}  // namespace strata::hy4f::vision
