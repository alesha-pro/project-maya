#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::hy4f::expert_q8 {

struct Weight {
    __half d;
    uint16_t q[16];
};
struct Activation {
    __half2 ds;
    int q[8];
};
static_assert(sizeof(Weight) == 34 && sizeof(Activation) == 36);

// A native Q8_0 row uses four warps and four lanes per 32-element block.
// Compute those same four partial sums in one warp, then add them in the
// native order before the XOR reduction. Slots of the same expert reuse W.
template<int Slots, int FixedBlocks>
__device__ __forceinline__ void dot(const Weight* row, const Activation* input, size_t stride,
                                    const int* slots, int first_slot, float* out, int rows, int r, int dyn_blocks) {
    const int lane = threadIdx.x & 31, word = 2 * (lane & 3);
    const int blocks = FixedBlocks ? FixedBlocks : dyn_blocks;
    int slot[Slots];
    float sum[Slots] = {};
#pragma unroll
    for (int t = 0; t < Slots; ++t) slot[t] = slots ? slots[t] : first_slot;
#pragma unroll
    for (int warp = 0; warp < 4; ++warp) {
        float partial[Slots] = {};
        for (int b = 8 * warp + (lane >> 2); b < blocks; b += 32) {
            const Weight& wb = row[b];
            const auto* q = wb.q + 2 * word;
            const int lo = int(unsigned(q[0]) | (unsigned(q[1]) << 16));
            const int hi = int(unsigned(q[2]) | (unsigned(q[3]) << 16));
            const float dw = __half2float(wb.d);
#pragma unroll
            for (int t = 0; t < Slots; ++t) {
                const auto& a = input[size_t(slot[t]) * stride + b];
                const int v = __dp4a(hi, a.q[word + 1], __dp4a(lo, a.q[word], 0));
                partial[t] += (dw * __low2float(a.ds)) * float(v);
            }
        }
#pragma unroll
        for (int t = 0; t < Slots; ++t) {
            if (warp == 0) sum[t] = partial[t];
            else sum[t] += partial[t];
        }
    }
#pragma unroll
    for (int t = 0; t < Slots; ++t) {
#pragma unroll
        for (int d = 16; d; d >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], d);
        if (!lane) out[size_t(slot[t]) * rows + r] = sum[t];
    }
}

template<int FixedBlocks, bool Grouped>
__global__ void by_expert(const Weight* w, size_t expert_blocks, const int* ids, int entries, const int* ng,
                          const int* offsets, const int* slots, const Activation* x, size_t stride, float* out,
                          int blocks, int rows) {
    const int count = Grouped ? *ng : entries;
    for (int item = blockIdx.x * 4 + threadIdx.x / 32; item < count * rows; item += gridDim.x * 4) {
        const int g = item / rows, r = item % rows, e = ids[g];
        if (e < 0) continue;
        const Weight* row = w + size_t(e) * expert_blocks + size_t(r) * blocks;
        if constexpr (Grouped) {
            int i = offsets[g];
            const int end = offsets[g + 1];
            for (; i + 4 <= end; i += 4) dot<4, FixedBlocks>(row, x, stride, slots + i, 0, out, rows, r, blocks);
            switch (end - i) {
            case 3: dot<3, FixedBlocks>(row, x, stride, slots + i, 0, out, rows, r, blocks); break;
            case 2: dot<2, FixedBlocks>(row, x, stride, slots + i, 0, out, rows, r, blocks); break;
            case 1: dot<1, FixedBlocks>(row, x, stride, slots + i, 0, out, rows, r, blocks); break;
            }
        } else {
            dot<1, FixedBlocks>(row, x, stride, nullptr, g, out, rows, r, blocks);
        }
    }
}

template<bool Grouped>
inline void launch(const void* weights, size_t expert_bytes, const int* ids, int entries, const int* ng,
                   const int* offsets, const int* slots, const void* x, bool shared, float* out, int n_in,
                   int n_out, int grid, cudaStream_t stream) {
    const auto* w = static_cast<const Weight*>(weights);
    const auto* input = static_cast<const Activation*>(x);
    const int blocks = n_in / 32;
    const size_t stride = shared ? 0 : size_t(blocks);
    const size_t expert_blocks = expert_bytes / sizeof(Weight);
    if (blocks == 48) {
        by_expert<48, Grouped><<<grid, 128, 0, stream>>>(w, expert_blocks, ids, entries, ng, offsets, slots, input,
                                                        stride, out, blocks, n_out);
    } else {
        by_expert<0, Grouped><<<grid, 128, 0, stream>>>(w, expert_blocks, ids, entries, ng, offsets, slots, input,
                                                       stride, out, blocks, n_out);
    }
}

} // namespace strata::hy4f::expert_q8
