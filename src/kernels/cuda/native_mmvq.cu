// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* __restrict__ x,
                                           Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = fminf(amax / 127.0f, 65504.0f);   // finite in FP16 (Strata #1448); below that nothing changes
    const int8_t q = amax == 0.0f ? 0 : fmaxf(-127.0f, fminf(127.0f, roundf(xi / d)));
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = __dp4a(u, qx, sumi);
        sumi = __dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = __dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = __dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = __dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = __dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = __dp4a(v.x, q8[i], sumi);
        sumi = __dp4a(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = __dp4a(q8[j * 2], r.qx[j], sumi);
            sumi = __dp4a(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = __dp4a(r.v[j].x, u0, sumi);
            sumi = __dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// The four 32-element formats: `load` keeps the block pointer (their decode is a few integer ops) and `apply` is
// the unchanged small_q8_dot. Their per-column cost is small; the win above is for the K and IQ formats.
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    __device__ static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    __device__ static W load(const Block* __restrict__ w, int) { return W{w}; }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int k) { return small_q8_dot(r.w, x, k); }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

template<typename F, int NCOLS, int NW, int ROWS>
__launch_bounds__(NW * WARP, 1)
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        native_mmvq_multi_kernel<F, NCOLS, NW, 2><<<blocks, dim3(WARP, NW), 0, s>>>(w, x, y, n_in, n_out);
        return;
    }
    const dim3 threads(WARP, WARPS);
    if (n_in / F::DIV < F::BPI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_mmvq_multi_kernel<F, NCOLS, WARPS, WARPS><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

} // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                 static_cast<cudaStream_t>(stream)>>>(x, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

// A physical warp holds the four partial sums of the canonical four-warp
// Q8_0 reduction. Each lane walks the same blocks as before. Adding the
// partials locally avoids the shared-memory exchange for every output row.
namespace {
template<int T>
__global__ void __launch_bounds__(128) hy4_q80_rows(const Q80Block* __restrict__ w,
                                                   const Q81Block* __restrict__ x, float* __restrict__ y,
                                                   int nb, int rows) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const Q80Block* wr = w + size_t(row) * nb;
    float acc[T][4] = {};
    const int q = (lane & 3) * 2;
    for (int i = lane / 4; i < nb; i += 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = i + j * 8;
            if (b >= nb) continue;
            const int v0 = load_int_b2(wr[b].qs, q), v1 = load_int_b2(wr[b].qs, q + 1);
            const float dw = wr[b].d;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const auto& a = x[t * nb + b];
                const int* in = reinterpret_cast<const int*>(a.qs) + q;
                int si = __dp4a(v1, in[1], __dp4a(v0, in[0], 0));
                acc[t][j] += dw * __low2float(a.ds) * float(si);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = ((acc[t][0] + acc[t][1]) + acc[t][2]) + acc[t][3];
        v = warp_sum(v);
        if (lane == 0) y[t * rows + row] = v;
    }
}
} // namespace
// Keep lane assignments, K traversal and four-warp accumulation identical
// while amortizing activation loads over more output rows in each block.
namespace {
// The release52 4-warp decomposition, specialized for fixed multiples of 1024.
// The eight signed weights stay at their original GGUF block offsets.
template<bool Wide>
__device__ __forceinline__ int2 q80_octet(const signed char* q, int b) {
    if constexpr (!Wide) return make_int2(load_int_b2(q,0),load_int_b2(q,1));
    // A GGUF Q8_0 block is 34 bytes; every second qs payload is two bytes
    // off a 32-bit boundary. Neighbor words cover all eight original bytes.
    const auto* u=reinterpret_cast<const unsigned*>((uintptr_t)q & ~uintptr_t(3));
    const unsigned a=u[0],c=u[1];
    if (b%2) return make_int2(a,c);
    const unsigned d=u[2];
    return make_int2(__byte_perm(a,c,0x5432),__byte_perm(c,d,0x5432));
}
template<int K,int T,int R,bool Wide>
__global__ __launch_bounds__(128) void hy4_q80_static(const Q80Block* __restrict__ w,const Q81Block* __restrict__ x,
                                                float* __restrict__ y,int rows) {
    constexpr int Nb=K/32;
    const int tid=threadIdx.x, lane=tid%32, warp=tid/32, row0=blockIdx.x*R;
    const int octet=tid%4;
    float acc[R][T]={};
#pragma unroll
    for(int seg=0;seg<Nb/32;++seg) {
        const int b=tid/4+32*seg;
        int2 input[T];float scale[T];
#pragma unroll
        for(int t=0;t<T;++t) {
            const Q81Block& v=x[t*Nb+b];
            input[t]=make_int2(reinterpret_cast<const int*>(v.qs)[octet*2],reinterpret_cast<const int*>(v.qs)[octet*2+1]);
            scale[t]=__low2float(v.ds);
        }
#pragma unroll
        for(int rr=0;rr<R;++rr) {
            const Q80Block& wr=w[(row0+rr)*Nb+b];
            const int2 v=q80_octet<Wide>(wr.qs+octet*8,b);
            const float d=__half2float(wr.d);
#pragma unroll
            for(int t=0;t<T;++t) {
                int dot=__dp4a(v.y,input[t].y,__dp4a(v.x,input[t].x,0));
                acc[rr][t]=fmaf(d*scale[t],float(dot),acc[rr][t]);
            }
        }
    }
    __shared__ float partial[3][R][T][32];
    if(warp)
#pragma unroll
        for(int rr=0;rr<R;++rr)
#pragma unroll
            for(int t=0;t<T;++t)partial[warp-1][rr][t][lane]=acc[rr][t];
    __syncthreads();
    if(warp)return;
#pragma unroll
    for(int rr=0;rr<R;++rr)
#pragma unroll
        for(int t=0;t<T;++t) {
            float sum=acc[rr][t];
#pragma unroll
            for(int j=0;j<3;++j)sum+=partial[j][rr][t][lane];
#pragma unroll
            for(int skip=16;skip;skip/=2)sum+=__shfl_xor_sync(~0u,sum,skip);
            if(lane==rr)y[t*rows+row0+rr]=sum;
        }
}

// Each shape uses the best geometry measured on release52 for its exact four-warp reduction.
template<int T> bool q80_static_shape(const Q80Block* w,const Q81Block* x,float* y,int k,int rows,cudaStream_t st) {
    if (k == 1024 && (rows == 4096 || rows == 8192)) {
        hy4_q80_static<1024,T,2,true><<<rows/2,128,0,st>>>(w,x,y,rows);
    } else if (k == 4096 && rows == 3072) {
        hy4_q80_static<4096,T,1,T==3><<<rows,128,0,st>>>(w,x,y,rows);
    } else if (k == 3072 && rows == 2624) {
        constexpr int r=T==3?2:1;
        hy4_q80_static<3072,T,r,true><<<rows/r,128,0,st>>>(w,x,y,rows);
    } else return false;
    return true;
}

template<int T> void q8_tiled(int tile, const Q80Block* w, const Q81Block* x, float* y, int n, int rows, cudaStream_t st) {
    using F = SmallTraits<Q80Block, 8>;
    const dim3 threads(32, 4);
    switch (tile) {
    case 1: native_mmvq_multi_kernel<F,T,4,1><<<rows,threads,0,st>>>(w,x,y,n,rows); break;
    case 2: native_mmvq_multi_kernel<F,T,4,2><<<(rows+1)/2,threads,0,st>>>(w,x,y,n,rows); break;
    case 4: native_mmvq_multi_kernel<F,T,4,4><<<(rows+3)/4,threads,0,st>>>(w,x,y,n,rows); break;
    case 8: native_mmvq_multi_kernel<F,T,4,8><<<(rows+7)/8,threads,0,st>>>(w,x,y,n,rows); break;
    }
}
}
void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    const char* fixed = std::getenv("HY4F_Q8_STATIC");
    if (fixed && fixed[0]=='1' && !fixed[1] && (ncols==2 || ncols==3) && !(reinterpret_cast<uintptr_t>(weights)&3u)
        && !std::getenv("HY4F_Q8_TILE") && !std::getenv("HY4F_Q80_ROWS")) {
        validate_pointer(weights);
        validate_pointer(x_q8_1);
        validate_pointer(y);
        validate_stream(stream);
        const auto w=static_cast<const Q80Block*>(weights);
        const auto x=static_cast<const Q81Block*>(x_q8_1);
        const auto st=static_cast<cudaStream_t>(stream);
        if (ncols==2 ? q80_static_shape<2>(w,x,y,n_in,n_out,st) : q80_static_shape<3>(w,x,y,n_in,n_out,st)) {
            launch_check();
            return;
        }
    }
    const bool automatic = std::getenv("HY4F_Q8_AUTO") != nullptr;
    const char* requested = std::getenv("HY4F_Q8_TILE");
    if (!requested && automatic && n_in == 3072 && (n_out == 2304 || n_out == 5696) && (ncols == 2 || ncols == 3))
        requested = "1";
    const bool single_warp = std::getenv("HY4F_Q80_ROWS") || (automatic && n_in == 384 && n_out == 3072);
    const bool eligible = ncols >= 1 && ncols <= 4 && n_in > 0 && n_in % 32 == 0 && n_out > 0;
    if (eligible && (requested || single_warp)) {
        validate_pointer(weights);
        validate_pointer(x_q8_1);
        validate_pointer(y);
        validate_stream(stream);
    }
    if (requested && eligible) {
        int tile = std::atoi(requested);
        if (tile == 1 || tile == 2 || tile == 4 || tile == 8) {
            const auto* w = static_cast<const Q80Block*>(weights);
            const auto* x = static_cast<const Q81Block*>(x_q8_1);
            const auto st = static_cast<cudaStream_t>(stream);
            switch (ncols) {
            case 1: q8_tiled<1>(tile,w,x,y,n_in,n_out,st); break;
            case 2: q8_tiled<2>(tile,w,x,y,n_in,n_out,st); break;
            case 3: q8_tiled<3>(tile,w,x,y,n_in,n_out,st); break;
            case 4: q8_tiled<4>(tile,w,x,y,n_in,n_out,st); break;
            }
            launch_check();
            return;
        }
    }
    if (single_warp && eligible) {
        const unsigned blocks = (n_out + 3) / 4;
        auto st = static_cast<cudaStream_t>(stream);
        auto w = static_cast<const Q80Block*>(weights);
        auto x = static_cast<const Q81Block*>(x_q8_1);
        switch (ncols) {
        case 1: hy4_q80_rows<1><<<blocks, 128, 0, st>>>(w, x, y, n_in / 32, n_out); break;
        case 2: hy4_q80_rows<2><<<blocks, 128, 0, st>>>(w, x, y, n_in / 32, n_out); break;
        case 3: hy4_q80_rows<3><<<blocks, 128, 0, st>>>(w, x, y, n_in / 32, n_out); break;
        case 4: hy4_q80_rows<4><<<blocks, 128, 0, st>>>(w, x, y, n_in / 32, n_out); break;
        }
        launch_check(); return;
    }
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

} // namespace strata::kernels

// ---------------------------------------------------------------------------------------------------------------
// hy4f: routed-expert MMVQ with the experts picked on the device.  One launch covers n_ids experts of one 3-D
// expert tensor {n_in, n_out, n_expert}: blockIdx.y = k reads expert ids[k], activation column k (or the one shared
// column) and writes output column k.  The per-row math is the ncols=1 small-K kernels' (four rows per block,
// eight weight blocks per iteration, the same dot and the same reduction order), so every column is bitwise the
// single-expert native_*_mmvq call on that expert's slice.
namespace strata::kernels {
namespace {

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_kq_mmvq_ids_kernel(const Block* __restrict__ w, std::size_t expert_blocks,
                                          const int* __restrict__ ids, const Q81Block* __restrict__ x,
                                          std::size_t x_col_blocks, float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = WARPS;
    constexpr int BLOCKS_PER_ITER = 8;
    const int k = int(blockIdx.y);
    const int id = ids[k];
    if (id < 0) {   // not this shard's expert (expert parallel): its column is zero
        const int r = WARPS * int(blockIdx.x);
        if (threadIdx.y == 0 && int(threadIdx.x) < WARPS && r + int(threadIdx.x) < n_out)
            y[std::size_t(k) * n_out + r + threadIdx.x] = 0.0f;
        return;
    }
    const Block* __restrict__ wk = w + std::size_t(id) * expert_blocks;
    const Q81Block* __restrict__ xk = x + std::size_t(k) * x_col_blocks;
    float* __restrict__ yk = y + std::size_t(k) * n_out;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = KQS_MUL * (tid % 16);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += Dot(wk + block, xk + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) yk[row0 + i] = tmp[i];
    }
}

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
void launch_ids(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids, const void* x_q8_1,
                bool x_shared, float* y, int n_in, int n_out, void* stream) {
    if (n_in <= 0 || n_in % QK != 0 || n_in / QK >= 8) throw std::invalid_argument("native MMVQ ids: n_in");
    if (n_out <= 0 || n_ids <= 0 || n_ids > 65535) throw std::invalid_argument("native MMVQ ids: n_out / n_ids");
    if (expert_bytes % sizeof(Block) != 0) throw std::invalid_argument("native MMVQ ids: expert_bytes");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const dim3 threads(WARP, WARPS);
    const dim3 blocks(unsigned((n_out + WARPS - 1) / WARPS), unsigned(n_ids));
    native_kq_mmvq_ids_kernel<Block, Dot, KQS_MUL><<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Block*>(weights), expert_bytes / sizeof(Block), ids, static_cast<const Q81Block*>(x_q8_1),
        x_shared ? 0 : std::size_t(n_in / Q8K), y, n_in, n_out);
    launch_check();
}

}  // namespace

void native_q3_k_mmvq_ids(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids,
                          const void* x_q8_1, bool x_shared, float* y, int n_in, int n_out, void* stream) {
    launch_ids<Q3KBlock, q3_q8_dot, 1>(weights, expert_bytes, ids, n_ids, x_q8_1, x_shared, y, n_in, n_out, stream);
}
void native_q4_k_mmvq_ids(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids,
                          const void* x_q8_1, bool x_shared, float* y, int n_in, int n_out, void* stream) {
    launch_ids<Q4KBlock, q4_q8_dot, VDR>(weights, expert_bytes, ids, n_ids, x_q8_1, x_shared, y, n_in, n_out, stream);
}

}  // namespace strata::kernels

// ---------------------------------------------------------------------------------------------------------------
// hy4f: the routed-expert MMVQ with the slot loop inside the block: one block per ROWS output rows walks all n_ids
// slots and skips the other shards' experts (id < 0) without a launch.  The per-slot math is native_kq_mmvq_ids_kernel's
// (bitwise the same rows); the grid is n_out / ROWS blocks instead of n_out / ROWS x n_ids (that launch was mostly
// blocks that exited at once: with 4 shards 3 slots in 4 are another shard's).  Other shards' columns are NOT written.
namespace strata::kernels {
namespace {

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_kq_mmvq_ids_loop_kernel(const Block* __restrict__ w, std::size_t expert_blocks,
                                               const int* __restrict__ ids, int n_ids, const Q81Block* __restrict__ x,
                                               std::size_t x_col_blocks, float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = WARPS;
    constexpr int BLOCKS_PER_ITER = 8;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    for (int k = 0; k < n_ids; ++k) {
        const int id = ids[k];
        if (id < 0) continue;
        const Block* __restrict__ wk = w + std::size_t(id) * expert_blocks;
        const Q81Block* __restrict__ xk = x + std::size_t(k) * x_col_blocks;
        float tmp[ROWS] = {};
        for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
            const int kby = kbx * (QK / Q8K);
            const int kqs = KQS_MUL * (tid % 16);
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
                    const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                    tmp[i] += Dot(wk + block, xk + kby, kqs);
                }
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
                tmp[i] = warp_sum(tmp[i]);
                if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(k) * n_out + row0 + i] = tmp[i];
            }
        }
        __syncthreads();
    }
}

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
void launch_ids_loop(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids, const void* x_q8_1,
                     bool x_shared, float* y, int n_in, int n_out, void* stream) {
    if (n_in <= 0 || n_in % QK != 0 || n_in / QK >= 8) throw std::invalid_argument("native MMVQ ids: n_in");
    if (n_out <= 0 || n_ids <= 0) throw std::invalid_argument("native MMVQ ids: n_out / n_ids");
    if (expert_bytes % sizeof(Block) != 0) throw std::invalid_argument("native MMVQ ids: expert_bytes");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const dim3 threads(WARP, WARPS);
    const dim3 blocks(unsigned((n_out + WARPS - 1) / WARPS), 1u);
    native_kq_mmvq_ids_loop_kernel<Block, Dot, KQS_MUL><<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Block*>(weights), expert_bytes / sizeof(Block), ids, n_ids, static_cast<const Q81Block*>(x_q8_1),
        x_shared ? 0 : std::size_t(n_in / Q8K), y, n_in, n_out);
    launch_check();
}

}  // namespace

void native_q3_k_mmvq_ids_loop(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids,
                               const void* x_q8_1, bool x_shared, float* y, int n_in, int n_out, void* stream) {
    launch_ids_loop<Q3KBlock, q3_q8_dot, 1>(weights, expert_bytes, ids, n_ids, x_q8_1, x_shared, y, n_in, n_out, stream);
}
void native_q4_k_mmvq_ids_loop(const void* weights, std::size_t expert_bytes, const int* ids, int n_ids,
                               const void* x_q8_1, bool x_shared, float* y, int n_in, int n_out, void* stream) {
    launch_ids_loop<Q4KBlock, q4_q8_dot, VDR>(weights, expert_bytes, ids, n_ids, x_q8_1, x_shared, y, n_in, n_out, stream);
}

}  // namespace strata::kernels

// ---------------------------------------------------------------------------------------------------------------
// hy4f: routed experts for a few tokens with the weights read once per (row, expert): groups = the distinct local
// experts of a step with the slots that picked each (at most one slot per token).  A warp takes work items
// (row, group) from a balanced grid-stride list (persistent warps, no blocks that exit at once); two 16-lane halves
// walk the row's 256-value blocks, each block's dot reused for every slot of the group.  Same dot as the ncols=1 path.
namespace strata::kernels {
namespace {

// Hy4's latent width is 1536: the two halves of the warp always process
// exactly three 256-value blocks each. Compile the singleton and shared
// cases separately so weight decoding and activation address arithmetic
// are invariant across the unrolled inner loop.
template<class F, int Slots>
__device__ __forceinline__ void grouped_row1536(const typename F::Block* w, const Q81Block* x,
                                                const int* slots, float* y, int row, int rows) {
    const int lane = threadIdx.x & 31;
    const int kqs = F::kqs(lane & 15);
    int slot[Slots];
    float acc[Slots] = {};
#pragma unroll
    for (int j = 0; j < Slots; ++j) slot[j] = slots[j];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int kb = lane / 16 + i * 2;
        const auto v = F::load(w + kb, kqs);
#pragma unroll
        for (int j = 0; j < Slots; ++j) acc[j] += F::apply(v, x + slot[j] * 48 + kb * 8, kqs);
    }
#pragma unroll
    for (int j = 0; j < Slots; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) y[slot[j] * rows + row] = v;
    }
}

template<class F, int Rows>
__global__ void __launch_bounds__(128) native_hy4_grouped1536(
    const typename F::Block* __restrict__ w, size_t expert_blocks, const int* __restrict__ ng,
    const int* __restrict__ gexp, const int* __restrict__ goff, const int* __restrict__ gslot,
    const Q81Block* __restrict__ x, float* __restrict__ y) {
    const int items = *ng * Rows;
    for (int it = blockIdx.x * 4 + (threadIdx.x >> 5); it < items; it += gridDim.x * 4) {
        const int g = it / Rows, row = it % Rows;
        const auto* wr = w + size_t(gexp[g]) * expert_blocks + row * 6;
        const int off = goff[g], ns = goff[g + 1] - off;
        switch (ns) {
            case 1: grouped_row1536<F, 1>(wr, x, gslot + off, y, row, Rows); break;
            case 2: grouped_row1536<F, 2>(wr, x, gslot + off, y, row, Rows); break;
            case 3: grouped_row1536<F, 3>(wr, x, gslot + off, y, row, Rows); break;
            case 4: grouped_row1536<F, 4>(wr, x, gslot + off, y, row, Rows); break;
        }
    }
}
// Keep the same expert's activation blocks in registers across neighboring
// output rows. Each warp loads 12 DP4A operands and scales per slot once.
template<int Slots, int Rows, int Reuse>
__device__ __forceinline__ void q3_rows_cache(const Q3KBlock* weights, const Q81Block* x, const int* slots,
                                             int start_row, float* y) {
    const int lane = threadIdx.x & 31, iq = lane & 15;
    const int qoff = 4 * (iq / 8), soff = (iq / 8) * 8 + (iq % 8) / 4;
    int id[Slots], u[Slots][3][4]; float ds[Slots][3][4];
#pragma unroll
    for (int t = 0; t < Slots; ++t) {
        id[t] = slots[t];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const Q81Block* a = x + id[t] * 48 + ((lane >> 4) + 2 * k) * 8 + qoff;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                u[t][k][j] = reinterpret_cast<const int*>(a[j].qs)[iq % 8];
                ds[t][k][j] = __low2float(a[j].ds);
            }
        }
    }
#pragma unroll 1
    for (int row = start_row; row < start_row + 4 * Reuse; row += 4) {
        float acc[Slots] = {};
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const auto& w = weights[row * 6 + (lane >> 4) + 2 * k];
            const unsigned l = load_int_b2(w.qs, iq), h = ~unsigned(load_int_b2(w.hmask, iq % 8)) >> qoff;
            float sub[Slots] = {};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int si = soff + 2 * j;
                const int lo = (w.scales[si % 8] >> ((si / 8) * 4)) & 15;
                const int hi = ((w.scales[8 + si % 4] >> ((si / 4) * 2)) & 3) * 16;
                const unsigned val = (l >> (2 * j)) & 0x03030303u;
                const unsigned sign = ((h >> j) << 2) & 0x04040404u;
                const int vi = int(val | (sign * 63u)); // each byte's high six bits equal the sign
#pragma unroll
                for (int t = 0; t < Slots; ++t)
                    sub[t] += ds[t][k][j] * float(__dp4a(vi, u[t][k][j], 0) * ((lo | hi) - 32));
            }
#pragma unroll
            for (int t = 0; t < Slots; ++t) acc[t] += __half2float(w.d) * sub[t];
        }
#pragma unroll
        for (int t = 0; t < Slots; ++t) {
            const float v = warp_sum(acc[t]);
            if (!lane) y[id[t] * Rows + row] = v;
        }
    }
}
// x's 8-element sums provide Q4_K's minimum correction. The four
// dot-of-ones instructions and four x loads are shared by Reuse rows.
template<int Slots, int Rows, int Reuse>
__device__ __forceinline__ void q4_rows_cache(const Q4KBlock* weights, const Q81Block* x, const int* slots,
                                             int start_row, float* y) {
    const int lane = threadIdx.x & 31, iq = lane & 15;
    const int qoff = 2 * (iq / 4), qi = iq % 4;
    int id[Slots], u[Slots][3][4], usum[Slots][3][2]; float ds[Slots][3][2];
#pragma unroll
    for (int t = 0; t < Slots; ++t) {
        id[t] = slots[t];
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const Q81Block* a = x + id[t] * 48 + ((lane >> 4) + 2 * k) * 8 + qoff;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int* q = reinterpret_cast<const int*>(a[j].qs) + qi;
                u[t][k][2*j] = q[0]; u[t][k][2*j + 1] = q[4];
                usum[t][k][j] = __dp4a(0x01010101,q[4],__dp4a(0x01010101,q[0],0));
                ds[t][k][j] = __low2float(a[j].ds);
            }
        }
    }
#pragma unroll 1
    for (int row = start_row; row < start_row + 4 * Reuse; row += 4) {
        float acc[Slots] = {};
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const auto wt = Q4KTraits::load(weights + row * 6 + (lane >> 4) + 2*k, iq * 2);
            const auto* sc = reinterpret_cast<const uint8_t*>(wt.aux);
            const auto dm = __half22float2(wt.dm);
            float sub[Slots] = {}, mins[Slots] = {};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int v0 = (wt.v[0] >> (4*j)) & 0x0f0f0f0f, v1 = (wt.v[1] >> (4*j)) & 0x0f0f0f0f;
#pragma unroll
                for (int t = 0; t < Slots; ++t) {
                    int dp = __dp4a(v1,u[t][k][2*j+1], __dp4a(v0,u[t][k][2*j],0));
                    sub[t] += ds[t][k][j] * float(dp * sc[j]);
                    mins[t] += ds[t][k][j] * float(usum[t][k][j] * sc[2+j]);
                }
            }
#pragma unroll
            for (int t = 0; t < Slots; ++t) acc[t] += dm.x * sub[t] - dm.y * mins[t];
        }
#pragma unroll
        for (int t = 0; t < Slots; ++t) {
            const float v = warp_sum(acc[t]);
            if (!lane) y[id[t] * Rows + row] = v;
        }
    }
}
template<int Rows, int Reuse>
__global__ void __launch_bounds__(128) hy4_q3_cache(const Q3KBlock* w, size_t eb, const int* ng, const int* gexp,
                                                    const int* goff, const int* gslot, const Q81Block* x, float* y) {
    const int tiles = Rows / (4 * Reuse), total = *ng * tiles;
    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int g = tile / tiles, r = (tile % tiles) * (4 * Reuse) + (threadIdx.x >> 5);
        const Q3KBlock* wg = w + size_t(gexp[g]) * eb;
        const int off = goff[g], ns = goff[g + 1] - off;
        const int* slots = gslot + off;
        if (ns == 1) q3_rows_cache<1, Rows, Reuse>(wg, x, slots, r, y);
        else if (ns == 2) q3_rows_cache<2, Rows, Reuse>(wg, x, slots, r, y);
        else {
            for (int i = r; i < r + 4 * Reuse; i += 4)
                if (ns == 3) grouped_row1536<Q3KTraits, 3>(wg + i * 6, x, slots, y, i, Rows);
                else grouped_row1536<Q3KTraits, 4>(wg + i * 6, x, slots, y, i, Rows);
        }
    }
}
template<int Rows, int Reuse>
__global__ void __launch_bounds__(128) hy4_q4_cache(const Q4KBlock* w, size_t eb, const int* ng, const int* gexp,
                                                    const int* goff, const int* gslot, const Q81Block* x, float* y) {
    const int tiles = Rows / (4 * Reuse), total = *ng * tiles;
    for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        const int g = tile / tiles, r = (tile % tiles) * (4 * Reuse) + (threadIdx.x >> 5);
        const Q4KBlock* wg = w + size_t(gexp[g]) * eb;
        const int off = goff[g], ns = goff[g + 1] - off;
        const int* slots = gslot + off;
        if (ns == 1) q4_rows_cache<1, Rows, Reuse>(wg, x, slots, r, y);
        else if (ns == 2) q4_rows_cache<2, Rows, Reuse>(wg, x, slots, r, y);
        else {
            for (int i = r; i < r + 4 * Reuse; i += 4)
                if (ns == 3) grouped_row1536<Q4KTraits, 3>(wg + i * 6, x, slots, y, i, Rows);
                else grouped_row1536<Q4KTraits, 4>(wg + i * 6, x, slots, y, i, Rows);
        }
    }
}
// Each half warp owns one original 144-byte Q4_K block. Repeated slots
// share its weights, and two neighboring output rows share the Q8 operands.
template<int Slots, int Rows>
__device__ __forceinline__ void q4_repeated_pair(const Q4KBlock* expert, const Q81Block* x,
                                                const int* slots, float* y, int row) {
    const int lane=threadIdx.x&31;
    const int group=(lane&15)/4, part=lane&3;
    int id[Slots];
    float accum[2][Slots]{};
#pragma unroll
    for (int s=0;s<Slots;++s) id[s]=slots[s];
#pragma unroll
    for (int k=0;k<3;++k) {
        const int block=2*k+(lane>>4);
        int low[Slots][2], high[Slots][2], sums[Slots][2];
        float scales[Slots][2];
#pragma unroll
        for (int s=0;s<Slots;++s) {
            const auto* act=x+48*id[s]+8*block+2*group;
#pragma unroll
            for (int i=0;i<2;++i) {
                const auto* qs=reinterpret_cast<const int*>(act[i].qs);
                low[s][i]=qs[part];
                high[s][i]=qs[part+4];
                sums[s][i]=__dp4a(0x01010101,high[s][i],__dp4a(0x01010101,low[s][i],0));
                scales[s][i]=__low2float(act[i].ds);
            }
        }
#pragma unroll
        for (int r=0;r<2;++r) {
            const auto& w=expert[(row+4*r)*6+block];
            uint4 h;
            asm("ld.global.v4.u32 {%0,%1,%2,%3},[%4];"
                : "=r"(h.x),"=r"(h.y),"=r"(h.z),"=r"(h.w) : "l"(&w));
            const int shift=16*(group&1);
            const unsigned s0=h.y>>shift,m0=h.z>>shift,top=h.w>>shift;
            const unsigned sc=group<2 ? s0&0x3f3f : (top&0x0f0f)|((s0&0xc0c0)>>2);
            const unsigned mn=group<2 ? m0&0x3f3f : ((top>>4)&0x0f0f)|((m0&0xc0c0)>>2);
            const float dw=__half2float(__ushort_as_half(h.x));
            const float dm=__half2float(__ushort_as_half(h.x>>16));
            const auto* qs=reinterpret_cast<const int*>(w.qs)+8*group+part;
            const int a=qs[0], b=qs[4];
#pragma unroll
            for (int s=0;s<Slots;++s) {
                float d=0.f,m=0.f;
#pragma unroll
                for (int i=0;i<2;++i) {
                    const int v0=(a>>(4*i))&0x0f0f0f0f, v1=(b>>(4*i))&0x0f0f0f0f;
                    const int dot=__dp4a(v1,high[s][i],__dp4a(v0,low[s][i],0));
                    d+=scales[s][i]*float(dot*int((sc>>(8*i))&255));
                    m+=scales[s][i]*float(sums[s][i]*int((mn>>(8*i))&255));
                }
                accum[r][s]+=dw*d-dm*m;
            }
        }
    }
#pragma unroll
    for (int r=0;r<2;++r) {
#pragma unroll
        for (int s=0;s<Slots;++s) {
            const float v=warp_sum(accum[r][s]);
            if (!lane) y[id[s]*Rows+row+4*r]=v;
        }
    }
}

template<int Rows>
__global__ __launch_bounds__(128) void hy4_q4_repeated(const Q4KBlock* w,size_t eb,const int* ng,
        const int* gexp,const int* goff,const int* gslot,const Q81Block* x,float* y) {
    const int tiles=Rows/4, limit=(*ng)*tiles;
    for (int tile=blockIdx.x;tile<limit;tile+=gridDim.x) {
        const int g=tile/tiles, local=tile%tiles;
        const int row=4*local+(threadIdx.x>>5);
        const auto* expert=w+size_t(gexp[g])*eb;
        const int first=goff[g], count=goff[g+1]-first;
        const int* slots=gslot+first;
        if (count==1) {
            grouped_row1536<Q4KTraits,1>(expert+6*row,x,slots,y,row,Rows);
        } else if (!(local&1)) {
            if (count==2) q4_repeated_pair<2,Rows>(expert,x,slots,y,row);
            else if (count==3) q4_repeated_pair<3,Rows>(expert,x,slots,y,row);
            else if (count==4) q4_repeated_pair<4,Rows>(expert,x,slots,y,row);
        }
    }
}
template<class F> bool try_grouped1536(const void* weights, size_t expert_bytes, const int* ng, const int* gexp,
                                       const int* goff, const int* gslot, const void* x, float* y,
                                       int n_in, int n_out, int blocks, void* stream) {
    if (std::getenv("HY4F_GROUP_REFERENCE") || n_in != 1536 || (n_out != 1536 && n_out != 3072)) return false;
    if (expert_bytes % sizeof(typename F::Block) != 0) throw std::invalid_argument("native MMVQ grouped: expert_bytes");
    const auto* w = static_cast<const typename F::Block*>(weights);
    const size_t eb = expert_bytes / sizeof(typename F::Block);
    const auto* q = static_cast<const Q81Block*>(x);
    const auto st = static_cast<cudaStream_t>(stream);
    if constexpr (std::is_same_v<F, Q3KTraits>) {
        if (std::getenv("HY4F_Q3_CACHE")) {
            if (n_out == 3072) hy4_q3_cache<3072, 4><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
            else hy4_q3_cache<1536, 4><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
            launch_check();
            return true;
        }
    }
    if constexpr (std::is_same_v<F, Q4KTraits>) {
        if (std::getenv("HY4F_Q4_CACHE")) {
            if (n_out == 3072) hy4_q4_cache<3072, 4><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
            else hy4_q4_cache<1536, 4><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
            launch_check();
            return true;
        }
    }
    if (n_out == 3072) native_hy4_grouped1536<F, 3072><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
    else native_hy4_grouped1536<F, 1536><<<blocks * 2, 128, 0, st>>>(w, eb, ng, gexp, goff, gslot, q, y);
    launch_check();
    return true;
}
template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
__global__ void __launch_bounds__(256) native_kq_mmvq_grouped_kernel(
    const Block* __restrict__ w, std::size_t expert_blocks, const int* __restrict__ ng_p, const int* __restrict__ gexp,
    const int* __restrict__ goff, const int* __restrict__ gslot, const Q81Block* __restrict__ x, std::size_t x_col_blocks,
    float* __restrict__ y, int n_in, int n_out) {
    const int ng = *ng_p, lane = threadIdx.x & 31, half = lane >> 4;
    const int bpr = n_in / QK;
    const long long items = (long long) ng * n_out;
    const long long nwarps = (long long) gridDim.x * (blockDim.x >> 5);
    for (long long it = (long long) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int gi = (int) (it / n_out), r = (int) (it % n_out);
        const Block* __restrict__ wr = w + std::size_t(gexp[gi]) * expert_blocks + std::size_t(r) * bpr;
        const int s0 = goff[gi], ns = goff[gi + 1] - s0;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        const int kqs = KQS_MUL * (lane & 15);
        for (int kbx = half; kbx < bpr; kbx += 2) {
            const int kby = kbx * (QK / Q8K);
#pragma unroll
            for (int s = 0; s < 4; ++s)
                if (s < ns) acc[s] += Dot(wr + kbx, x + std::size_t(gslot[s0 + s]) * x_col_blocks + kby, kqs);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s < ns) {
                const float v = warp_sum(acc[s]);
                if (lane == 0) y[std::size_t(gslot[s0 + s]) * n_out + r] = v;
            }
        }
    }
}

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
__global__ void __launch_bounds__(256) native_kq_mmvq_grouped_mp_kernel(
    const Block* __restrict__ w, std::size_t expert_blocks, const int* __restrict__ ng_p, const int* __restrict__ gexp,
    const int* __restrict__ goff, const int* __restrict__ gslot, const Q81Block* __restrict__ x, std::size_t x_col_blocks,
    float* __restrict__ y, int n_in, int n_out) {
    const int ng = *ng_p, lane = threadIdx.x & 31, half = lane >> 4;
    const int bpr = n_in / QK;
    const long long items = (long long) ng * n_out;
    const long long nwarps = (long long) gridDim.x * (blockDim.x >> 5);
    for (long long it = (long long) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += nwarps) {
        const int gi = (int) (it / n_out), r = (int) (it % n_out);
        const Block* __restrict__ wr = w + std::size_t(gexp[gi]) * expert_blocks + std::size_t(r) * bpr;
        const int kqs = KQS_MUL * (lane & 15);
        // (slots in passes of 4: a step of more than 4 tokens may give one expert more slots; the row is read again)
        for (int s0 = goff[gi], se = goff[gi + 1]; s0 < se; s0 += 4) {
        const int ns = min(4, se - s0);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int kbx = half; kbx < bpr; kbx += 2) {
            const int kby = kbx * (QK / Q8K);
#pragma unroll
            for (int s = 0; s < 4; ++s)
                if (s < ns) acc[s] += Dot(wr + kbx, x + std::size_t(gslot[s0 + s]) * x_col_blocks + kby, kqs);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (s < ns) {
                const float v = warp_sum(acc[s]);
                if (lane == 0) y[std::size_t(gslot[s0 + s]) * n_out + r] = v;
            }
        }
        }
    }
}

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
void launch_grouped(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp, const int* goff,
                    const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out, int blocks, void* stream) {
    if (n_in <= 0 || n_in % QK != 0) throw std::invalid_argument("native MMVQ grouped: n_in");
    if (expert_bytes % sizeof(Block) != 0) throw std::invalid_argument("native MMVQ grouped: expert_bytes");
    native_kq_mmvq_grouped_kernel<Block, Dot, KQS_MUL><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Block*>(weights), expert_bytes / sizeof(Block), ng, gexp, goff, gslot,
        static_cast<const Q81Block*>(x_q8_1), std::size_t(n_in / Q8K), y, n_in, n_out);
    launch_check();
}

template <class Block, float (*Dot)(const Block* __restrict__, const Q81Block* __restrict__, int), int KQS_MUL>
void launch_grouped_mp(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp, const int* goff,
                    const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out, int blocks, void* stream) {
    if (n_in <= 0 || n_in % QK != 0) throw std::invalid_argument("native MMVQ grouped: n_in");
    if (expert_bytes % sizeof(Block) != 0) throw std::invalid_argument("native MMVQ grouped: expert_bytes");
    native_kq_mmvq_grouped_mp_kernel<Block, Dot, KQS_MUL><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Block*>(weights), expert_bytes / sizeof(Block), ng, gexp, goff, gslot,
        static_cast<const Q81Block*>(x_q8_1), std::size_t(n_in / Q8K), y, n_in, n_out);
    launch_check();
}

}  // namespace

void native_q3_k_mmvq_grouped(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp,
                              const int* goff, const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out,
                              int blocks, void* stream) {
    if (try_grouped1536<Q3KTraits>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream)) return;
    launch_grouped<Q3KBlock, q3_q8_dot, 1>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream);
}
void native_q4_k_mmvq_grouped(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp,
                              const int* goff, const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out,
                              int blocks, void* stream) {
    if (try_grouped1536<Q4KTraits>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream)) return;
    launch_grouped<Q4KBlock, q4_q8_dot, VDR>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream);
}

void native_q4_k_mmvq_grouped_repeated(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp,
                                       const int* goff, const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out,
                                       int blocks, void* stream) {
    if (n_in!=1536 || (n_out!=1536 && n_out!=3072) || (reinterpret_cast<uintptr_t>(weights)&15u) ||
        expert_bytes%sizeof(Q4KBlock) || std::getenv("HY4F_GROUP_REFERENCE")) {
        native_q4_k_mmvq_grouped(weights,expert_bytes,ng,gexp,goff,gslot,x_q8_1,y,n_in,n_out,blocks,stream);
        return;
    }
    const auto* w=static_cast<const Q4KBlock*>(weights);
    const size_t eb=expert_bytes/sizeof(Q4KBlock);
    const auto* q=static_cast<const Q81Block*>(x_q8_1);
    const auto st=static_cast<cudaStream_t>(stream);
    if (n_out==3072) hy4_q4_repeated<3072><<<blocks*2,128,0,st>>>(w,eb,ng,gexp,goff,gslot,q,y);
    else hy4_q4_repeated<1536><<<blocks*2,128,0,st>>>(w,eb,ng,gexp,goff,gslot,q,y);
    launch_check();
}

void native_q3_k_mmvq_grouped_mp(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp,
                              const int* goff, const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out,
                              int blocks, void* stream) {
    launch_grouped_mp<Q3KBlock, q3_q8_dot, 1>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream);
}
void native_q4_k_mmvq_grouped_mp(const void* weights, std::size_t expert_bytes, const int* ng, const int* gexp,
                              const int* goff, const int* gslot, const void* x_q8_1, float* y, int n_in, int n_out,
                              int blocks, void* stream) {
    launch_grouped_mp<Q4KBlock, q4_q8_dot, VDR>(weights, expert_bytes, ng, gexp, goff, gslot, x_q8_1, y, n_in, n_out, blocks, stream);
}

}  // namespace strata::kernels
