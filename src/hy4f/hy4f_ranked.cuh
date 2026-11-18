#pragma once
#include "hy4f.hpp"
#include <cub/block/block_radix_sort.cuh>

namespace strata::hy4f::kern {

template<bool Merge>
__global__ void kd_rank_head(const float* logits, const HeadValue* tiles, HeadValue* result,
                              int n, int blocks, int first_id) {
    constexpr int Threads=256, Items=Merge ? 8 : 4;
    using Sort=cub::BlockRadixSort<float,Threads,Items,int>;
    __shared__ typename Sort::TempStorage sort;
    const int row=blockIdx.y, block=blockIdx.x;
    float keys[Items]; int ids[Items]; bool invalid=false;
#pragma unroll
    for (int j=0; j<Items; ++j) {
        const int i=(block*Threads+threadIdx.x)*Items+j;
        keys[j]=-CUDART_INF_F; ids[j]=-1;
        if constexpr (Merge) {
            if (i<blocks*head_candidates) {
                const auto v=tiles[(size_t)row*blocks*head_candidates+i];
                keys[j]=v.logit; ids[j]=v.id; invalid|=v.id==-2;
            }
        } else if (i<n) {
            keys[j]=logits[(size_t)row*n+i]; ids[j]=first_id+i;
            invalid|=!isfinite(keys[j]);
        }
    }
    const bool bad=__syncthreads_or(invalid);
    Sort(sort).SortDescending(keys,ids);
#pragma unroll
    for (int j=0; j<Items; ++j) {
        const int rank=threadIdx.x*Items+j;
        if (rank<head_candidates) {
            auto& dst=result[((size_t)row*gridDim.x+block)*head_candidates+rank];
            dst.logit=keys[j]; dst.id=bad && !rank ? -2 : ids[j];
        }
    }
}
}  // namespace strata::hy4f::kern
