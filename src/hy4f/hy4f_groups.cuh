#pragma once

#include <cuda_runtime.h>

namespace strata::hy4f::kern {

// A router emits each expert at most once for one token.
__global__ void kd_groups_one(const int* local, int n, int* ng, int* experts, int* off, int* slots) {
    const int lane = threadIdx.x, e = lane < n ? local[lane] : -1;
    const unsigned valid = __ballot_sync(0xffffffffu, e >= 0);
    if (e >= 0) {
        const int group = __popc(valid & ((1u << lane) - 1));
        experts[group] = e;
        slots[group] = lane;
        off[group] = group;
    }
    if (lane == 0) {
        const int groups = __popc(valid);
        *ng = groups;
        off[groups] = groups;
    }
}

// One mask per (warp, local expert) identifies a group's first slot and the
// rank of each occurrence. Scanning leaders preserves both original orders.
__global__ void kd_groups_match(const int* local, int n, int expert_count, int* ng, int* experts, int* off, int* slots) {
    extern __shared__ unsigned hits[];
    __shared__ int warp_sum[4], preceding[128];
    const int slot = threadIdx.x, lane = slot % 32, warp = slot / 32, warps = blockDim.x / 32;
    const int e = slot < n ? local[slot] : -1;
    for (int i = slot; i < expert_count * warps; i += blockDim.x) hits[i] = 0;
    __syncthreads();
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) hits[warp * expert_count + e] = peers;
    __syncthreads();

    int leader = slot, rank = 0, count = 0;
    if (e >= 0) {
        for (int w = 0; w < warps; ++w) {
            const unsigned m = hits[w * expert_count + e];
            if (m) leader = min(leader, w * 32 + __ffs(m) - 1);
            count += __popc(m);
            if (w < warp) rank += __popc(m);
        }
        rank += __popc(peers & ((1u << lane) - 1));
    }
    // Low byte counts groups, high byte counts slots. Each total is <= 128.
    const int own = e >= 0 && leader == slot ? (count << 8) + 1 : 0;
    int prefix = own;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        const int v = __shfl_up_sync(0xffffffffu, prefix, delta);
        if (lane >= delta) prefix += v;
    }
    if (lane == 31) warp_sum[warp] = prefix;
    __syncthreads();
    for (int w = 0; w < warp; ++w) prefix += warp_sum[w];
    preceding[slot] = prefix - own;
    if (own) {
        const int prev = prefix - own;
        experts[prev & 255] = e;
        off[prev & 255] = prev >> 8;
    }
    if (slot == blockDim.x - 1) {
        *ng = prefix & 255;
        off[prefix & 255] = prefix >> 8;
    }
    __syncthreads();
    if (e >= 0) slots[(preceding[leader] >> 8) + rank] = slot;
}

} // namespace strata::hy4f::kern
