// Exact ordered decode selection for at most 65536 pooled keys after kd_score_hist_tile.
#pragma once
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include "hy4f_select.cuh"

namespace strata::hy4f::kern {
constexpr int KD_CANDIDATES = 1024;
struct KdCompactRow {
    unsigned next;
    unsigned candidate_count;
    uint2 candidates[KD_CANDIDATES];
};

__global__ void kd_score_select_init(unsigned* hist, KdCompactRow* rows) {
    const int row = blockIdx.x, tid = threadIdx.x;
    for (int b = tid; b < 512; b += blockDim.x) hist[row * 512 + b] = 0;
    if (tid == 0) rows[row].next = rows[row].candidate_count = 0;
}

// All CTAs see the same published complete histogram after the score kernel.
__global__ void kd_score_compact(const float* scores, int ld, const int* pos0, int pool, int top_k,
                                 int top_pools, const unsigned* hist, KdCompactRow* rows) {
    const int row = blockIdx.y, pos = *pos0 + row, tid = threadIdx.x;
    if (pos + 1 <= top_k) return;
    const int n = pos / pool, k = min(n, top_pools);
    const int hi = 511 - 2 * tid;
    const unsigned a = hist[row * 512 + hi], b = hist[row * 512 + hi - 1];
    using Scan = cub::BlockScan<unsigned, 256>;
    __shared__ Scan::TempStorage scan;
    __shared__ unsigned chosen, count;
    unsigned before;
    Scan(scan).ExclusiveSum(a + b, before);
    if (before < k && k <= before + a + b) {
        const bool first = k <= before + a;
        chosen = first ? hi : hi - 1;
        count = first ? before + a : before + a + b;
    }
    __syncthreads();
    if (blockIdx.x == 0 && tid == 0) rows[row].candidate_count = count;
    if (count > KD_CANDIDATES) return;
    const int lane = tid & 31, step = gridDim.x * blockDim.x;
    for (int base = blockIdx.x * blockDim.x + tid - lane; base < n; base += step) {
        const int i = base + lane;
        const float v = i < n ? scores[(size_t)row * ld + i] : 0.f;
        const unsigned members = __ballot_sync(0xffffffffu, i < n && kd_score_coarse(v) >= chosen);
        if (!members) continue;
        unsigned start = lane == 0 ? atomicAdd(&rows[row].next, __popc(members)) : 0;
        start = __shfl_sync(0xffffffffu, start, 0);
        if (members & (1u << lane)) {
            const unsigned slot = start + __popc(members & ((1u << lane) - 1));
            rows[row].candidates[slot] = make_uint2(hy4f_key(v), i);
        }
    }
}

using KdSort = cub::BlockRadixSort<unsigned, 256, KD_CANDIDATES / 256>;
using KdScan = cub::BlockScan<int, 256>;
union KdSelectTemp {
    KdSort::TempStorage sort;
    KdScan::TempStorage scan;
    unsigned radix[2048];
};

// Oversubscribed boundary bucket. Compute the exact kth bit key as in the reference,
// then write strictly larger group IDs in index order, followed by equal IDs in index order.
__device__ __forceinline__ void kd_score_wide_bucket(const float* score, int n, int k, int pool,
                                                     int* cells, KdSelectTemp& temp) {
    __shared__ unsigned prefix, mask;
    __shared__ int need;
    int tid = threadIdx.x;
    if (!tid) { prefix = mask = 0; need = k; }
    __syncthreads();
    for (int pass = 0; pass < 3; ++pass) {
        const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0;
        const unsigned bins = pass == 2 ? 1024 : 2048;
        for (int i = tid; i < bins; i += 256) temp.radix[i] = 0;
        __syncthreads();
        const unsigned old = prefix, checked = mask;
        for (int i = tid; i < n; i += 256) {
            const unsigned v = hy4f_key(score[i]);
            if ((v & checked) == old) atomicAdd(temp.radix + ((v >> shift) & (bins - 1)), 1u);
        }
        __syncthreads();
        if (!tid) {
            int remaining = need;
            for (int i = (int)bins - 1; i >= 0; --i) {
                const int size = temp.radix[i];
                if (remaining <= size) {
                    prefix = old | ((unsigned)i << shift);
                    mask = checked | ((bins - 1) << shift);
                    need = remaining;
                    break;
                }
                remaining -= size;
            }
        }
        __syncthreads();
    }
    const unsigned cut = prefix;
    const int n_gt = k - need;
    int gt = 0, eq = 0;
    for (int base = 0; base < n; base += 256) {
        const int i = base + tid;
        const unsigned v = i < n ? hy4f_key(score[i]) : 0;
        const int above = i < n && v > cut, at = i < n && v == cut;
        // Both 16-bit fields hold at most 256 flags in this scan.
        int offset, total;
        KdScan(temp.scan).ExclusiveSum(above + (at << 16), offset, total);
        __syncthreads();
        const int rank = eq + (offset >> 16);
        if (above || (at && rank < need)) {
            const int slot = above ? gt + (offset & 65535) : n_gt + rank;
            for (int j = 0; j < pool; ++j) cells[slot * pool + j] = i * pool + j;
        }
        gt += total & 65535;
        eq += total >> 16;
    }
}

__global__ void kd_score_finish(const float* scores, int ld, const int* pos0, int pool, int top_k, int top_pools,
                                int cap, const KdCompactRow* rows, int* all_cells) {
    const int row = blockIdx.x, pos = *pos0 + row, tid = threadIdx.x;
    int* cells = all_cells + (size_t)row * cap;
    if (pos + 1 <= top_k) {
        for (int i = tid; i < cap; i += 256) cells[i] = i <= pos ? i : -1;
        return;
    }
    __shared__ KdSelectTemp temp;
    __shared__ unsigned threshold;
    const int n = pos / pool, k = min(n, top_pools);
    const unsigned count = rows[row].candidate_count;
    if (count <= KD_CANDIDATES) {
        constexpr int items = KD_CANDIDATES / 256;
        unsigned values[items], ids[items], sorted[items];
        // Striped loads, blocked output from the sorts.
        #pragma unroll
        for (int j = 0; j < items; ++j) {
            const int i = tid + j * 256;
            const uint2 kv = i < count ? rows[row].candidates[i] : make_uint2(0, 0xffffffffu);
            values[j] = sorted[j] = kv.x;
            ids[j] = kv.y;
        }
        KdSort(temp.sort).SortDescending(sorted);
        if (tid == (k - 1) / items) threshold = sorted[(k - 1) % items];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < items; ++j) {
            const unsigned id = ids[j];
            sorted[j] = id == 0xffffffffu || values[j] < threshold ? 0x3ffffu
                      : values[j] == threshold ? (0x10000u | id) : id;
        }
        KdSort(temp.sort).Sort(sorted, 0, 18);
        #pragma unroll
        for (int j = 0; j < items; ++j) {
            const int slot = tid * items + j;
            if (slot < k) for (int z = 0; z < pool; ++z)
                cells[slot * pool + z] = (sorted[j] & 65535u) * pool + z;
        }
    } else {
        kd_score_wide_bucket(scores + (size_t)row * ld, n, k, pool, cells, temp);
    }
    for (int i = k * pool + tid; i < cap; i += 256) {
        const int index = n * pool + i - k * pool;
        cells[i] = index <= pos ? index : -1;
    }
}
} // namespace strata::hy4f::kern
