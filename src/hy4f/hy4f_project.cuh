// Q8_0 K/V projection weights: four successive blocks per vector load from
// each lane, followed by the original half scales in row/block order.
#pragma once
#include "hy4f_dkern.cuh"
#include <algorithm>
#include <vector>

namespace strata::hy4f::kern {

inline std::vector<uint8_t> pack_projection(const uint8_t* src, int cols, int rows) {
    const int blocks = cols / 32;
    std::vector<uint8_t> packed((size_t)rows * blocks * 34);
    uint8_t* scales = packed.data() + (size_t)rows * cols;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks; ++b) {
            const size_t block = (size_t)r * blocks + b;
            const uint8_t* in = src + block * 34;
            scales[2 * block] = in[0];
            scales[2 * block + 1] = in[1];
            const int group = b / 4, count = std::min(4, blocks - group * 4);
            uint8_t* out = packed.data() + (size_t)r * cols + group * 128 + b % 4;
            for (int lane = 0; lane < 32; ++lane) out[lane * count] = in[2 + lane];
        }
    }
    return packed;
}

template<int Count>
__device__ __forceinline__ void projection_weights(const uint8_t* q, const __half* scales, float (&weights)[Count]) {
    const auto* ds = reinterpret_cast<const __half2*>(scales);
    const float2 lo = __half22float2(ds[0]);
    if constexpr (Count == 4) {
        const char4 v = *reinterpret_cast<const char4*>(q);
        const float2 hi = __half22float2(ds[1]);
        weights[0] = lo.x * float(v.x); weights[1] = lo.y * float(v.y);
        weights[2] = hi.x * float(v.z); weights[3] = hi.y * float(v.w);
    } else {
        const char2 v = *reinterpret_cast<const char2*>(q);
        weights[0] = lo.x * float(v.x); weights[1] = lo.y * float(v.y);
    }
}

// Two or four output channels share a warp. Virtual lanes keep independent
// accumulators until all of K has been processed, then fold the original
// 32-lane XOR tree in the same order.
struct ProjectionArgs {
    const uint8_t* w;
    const float* x;
    int H, ld, head_stride;
    float* output = nullptr;
    __half* query = nullptr;
    const int* pos = nullptr;
    float theta = 0, scale = 1;
    const float* gate = nullptr;
    int gate_stride = 0;
    Q81* qout = nullptr;
};

template<int K, int T, int Groups, int Base>
__device__ __forceinline__ void projection_dot(const ProjectionArgs& p, const __half* scales, int head, int row,
                                                int lane, float (&acc)[Groups][T]) {
    constexpr int Count = K / 32 - Base < 4 ? K / 32 - Base : 4;
#pragma unroll
    for (int v = 0; v < Groups; ++v) {
        const int vl = lane + v * (32 / Groups);
        float weight[Count];
        projection_weights<Count>(p.w + (size_t)row * K + Base * 32 + vl * Count,
                                  scales + (size_t)row * (K / 32) + Base, weight);
#pragma unroll
        for (int j = 0; j < Count; ++j)
#pragma unroll
            for (int t = 0; t < T; ++t)
                acc[v][t] += weight[j] * p.x[(size_t)t * p.ld + head * p.head_stride + vl + (Base + j) * 32];
    }
    if constexpr (Base + Count < K / 32)
        projection_dot<K, T, Groups, Base + Count>(p, scales, head, row, lane, acc);
}

template<int K, int T>
__global__ void kd_project_q8(ProjectionArgs p) {
    constexpr int O = K == 512 ? 256 : 512;
    constexpr int Groups = K == 192 && T > 1 ? 4 : 2;
    constexpr int Lanes = 32 / Groups;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    const int lane = threadIdx.x % Lanes, head = row / O;
    const auto* scales = reinterpret_cast<const __half*>(p.w + (size_t)p.H * O * K);
    float acc[Groups][T] = {};
    projection_dot<K, T, Groups, 0>(p, scales, head, row, lane, acc);
    __shared__ float output_row[K == 512 ? T : 1][32];
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int d = Groups / 2; d; d /= 2)
#pragma unroll
            for (int j = 0; j < d; ++j) acc[j][t] += acc[j + d][t];
        float sum = acc[0][t];
#pragma unroll
        for (int d = Lanes / 2; d; d /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, d);
        if constexpr (K == 512) {
            if (!lane) output_row[t][row % 32] = sum * sigm(p.gate[(size_t)t * p.gate_stride + row]);
        } else if (p.query) {
            __half* q = p.query + ((size_t)t * p.H + head) * 576;
            if (!lane) q[row % O] = __float2half_rn(sum * p.scale);
            if (row % O == 0) {
#pragma unroll
                for (int rot = lane; rot < 32; rot += Lanes) {
                    const float* x = p.x + (size_t)t * p.ld + (size_t)head * p.head_stride + K + 2 * rot;
                    float sn, cs;
                    sincosf((float)(*p.pos + t) * powf(p.theta, -2.f * rot / 64), &sn, &cs);
                    const float even = x[0] * cs - x[1] * sn, odd = x[0] * sn + x[1] * cs;
                    reinterpret_cast<__half2*>(q + 512)[rot] = __floats2half2_rn(even * p.scale, odd * p.scale);
                }
            }
        } else if (!lane) p.output[(size_t)t * p.H * O + row] = sum;
    }
    if constexpr (K == 512) {
        __syncthreads();
        if (threadIdx.x < 32)
#pragma unroll
            for (int t = 0; t < T; ++t)
                q81_store(p.qout + (size_t)t * p.H * (O / 32) + blockIdx.x, output_row[t][threadIdx.x]);
    }
}

template<int K, int T>
void launch_projection(const ProjectionArgs& p, cudaStream_t stream) {
    constexpr int O = K == 512 ? 256 : 512;
    constexpr int groups = K == 192 && T > 1 ? 4 : 2;
    constexpr int threads = K == 512 ? 512 : 256;
    kd_project_q8<K, T><<<p.H * O * (32 / groups) / threads, threads, 0, stream>>>(p);
}

template<int K>
void launch_projection(int n, const ProjectionArgs& p, cudaStream_t stream) {
    switch (n) {
        case 1: launch_projection<K, 1>(p, stream); break;
        case 2: launch_projection<K, 2>(p, stream); break;
        case 3: launch_projection<K, 3>(p, stream); break;
        case 4: launch_projection<K, 4>(p, stream); break;
        case 5: launch_projection<K, 5>(p, stream); break;
        case 6: launch_projection<K, 6>(p, stream); break;
    }
}

template<int K>
__global__ void kb_deq_project(const uint8_t* w, float* out, int rows) {
    constexpr int Groups = (K / 32 + 3) / 4;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const int row = warp / Groups, group = warp % Groups;
    if (row >= rows) return;
    const auto* scales = reinterpret_cast<const __half*>(w + (size_t)rows * K) + (size_t)row * (K / 32) + group * 4;
    const uint8_t* q = w + (size_t)row * K + group * 128;
    float* dst = out + (size_t)row * K + group * 128 + lane;
    if (K == 192 && group == 1) {
        float weight[2];
        projection_weights<2>(q + lane * 2, scales, weight);
#pragma unroll
        for (int j = 0; j < 2; ++j) dst[j * 32] = weight[j];
    } else {
        float weight[4];
        projection_weights<4>(q + lane * 4, scales, weight);
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j * 32] = weight[j];
    }
}

// The original single-token eager path accumulates each entire Q8 block
// before applying its scale. Retain that order when reading packed weights.
template<int K>
__global__ void k_project_eager(const uint8_t* w, const float* x, const float* gate, float* y, int H, int head_stride) {
    constexpr int O = K == 192 ? 512 : 256;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = K == 192 ? tid : tid / 32, lane = tid % 32;
    if (row >= H * O) return;
    const auto* scales = reinterpret_cast<const __half*>(w + (size_t)H * O * K) + row * (K / 32);
    const float* input = x + (size_t)(row / O) * head_stride;
    float sum = 0;
    for (int b = K == 192 ? 0 : lane; b < K / 32; b += K == 192 ? 1 : 32) {
        const int group = b / 4, count = min(4, K / 32 - group * 4);
        const auto* q = reinterpret_cast<const int8_t*>(w + (size_t)row * K + group * 128 + b % 4);
        float v = 0;
#pragma unroll
        for (int i = 0; i < 32; ++i) v += float(q[i * count]) * input[b * 32 + i];
        sum += __half2float(scales[b]) * v;
    }
    if constexpr (K == 192) y[row] = sum;
    else {
        sum = warp_sum(sum);
        if (!lane) y[row] = sum * sigm(gate[row]);
    }
}
} // namespace strata::hy4f::kern
