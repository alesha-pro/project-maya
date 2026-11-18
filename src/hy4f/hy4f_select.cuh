// src/hy4f/hy4f_select.cuh - the k-pool selection on the device, graph-ready (the position read from device memory).
//
// For the token at pos (dpos[0]): the candidate groups are the completed groups before the current one,
// n_pool = pos / kpool.  While pos + 1 <= top_k every cell 0..pos is attended.  Otherwise the top_pools groups by
// score (ties: the lower index first, llama.cpp's stable order) give kpool cells each, then the current group's cells
// gcur*kpool .. pos (the forced tail), then -1 up to n_cap.  One block of 1024 threads: a radix select of the
// top_pools-th largest key over n_pool scores (three passes, 11 + 11 + 10 bits), then the strictly larger keys
// (any order) and the ties in index order (a block-wide scan per 1024-chunk).
#pragma once

#include <cub/block/block_scan.cuh>

namespace strata::hy4f {

__device__ __forceinline__ unsigned hy4f_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}

template <int NT>
__global__ void k_select_kpool(const float* __restrict__ score, const int* __restrict__ dpos, int kpool, int top_k,
                               int top_pools, int n_cap, int* __restrict__ cells) {
    const int pos = dpos[0];
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
        const int sh = shifts[pass], nb = 1 << widths[pass];
        for (int i = tid; i < nb; i += NT) hist[i] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n_pool; i += NT) {
            const unsigned key = hy4f_key(score[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> sh) & (nb - 1)], 1u);
        }
        __syncthreads();
        if (tid == 0) {   // walk the bins from the top: the bin holding the kk-th largest
            int kk = s_kk;
            for (int b = nb - 1; b >= 0; --b) {
                const int c = (int) hist[b];
                if (c >= kk) {
                    s_prefix = prefix | ((unsigned) b << sh);
                    s_mask = mask | ((unsigned) (nb - 1) << sh);
                    break;
                }
                kk -= c;
            }
            s_kk = kk;   // ties still to take at the threshold, after this pass's narrowing
        }
        __syncthreads();
    }
    const unsigned thr = s_prefix;
    const int ties = s_kk;   // how many keys == thr to take (lowest index first)
    __shared__ int s_out;
    if (tid == 0) s_out = 0;
    __syncthreads();
    // strictly larger keys: any order
    for (int i = tid; i < n_pool; i += NT)
        if (hy4f_key(score[i]) > thr) {
            const int slot = atomicAdd(&s_out, 1);
            for (int m = 0; m < kpool; ++m) cells[slot * kpool + m] = i * kpool + m;
        }
    __syncthreads();
    // ties in index order
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
    // the forced tail and the padding
    const int n_top = k * kpool;
    for (int i = n_top + tid; i < n_cap; i += NT) {
        const int c = gcur * kpool + (i - n_top);
        cells[i] = c <= pos ? c : -1;
    }
}

}  // namespace strata::hy4f
