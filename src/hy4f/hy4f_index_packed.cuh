#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace strata::hy4f::packed_index {

// The prefill arena is exclusively borrowed until the next prefill. Tiles above
// this aligned, immutable prefix still come from the canonical row-major keys.
struct Cache { const uint4* base; int tiles; };

// Eight pooled groups by 32 dimensions occupy exactly four 128-byte sectors per warp load.
// Every lane receives both B operands for two successive K16 instructions.
__global__ void pack_keys(const __half* rows, uint4* packed, int count, Cache* cache) {
    if (!blockIdx.x && !threadIdx.x) *cache = {packed, count/8};
    const int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= ((count + 7) / 8) * 4 * 32) return;
    const int lane = index % 32, chunk = index / 32 % 4, row = index / 128 * 8 + lane / 4;
    const int pair = chunk * 16 + lane % 4;
    const auto* pair_row = reinterpret_cast<const unsigned*>(rows) + size_t(row) * 64;
    packed[index] = row < count ? make_uint4(pair_row[pair], pair_row[pair+4], pair_row[pair+8], pair_row[pair+12])
                                : make_uint4(0,0,0,0);
}

__device__ __forceinline__ unsigned fp16_pair(const float* values) {
    const float2 v = *reinterpret_cast<const float2*>(values);
    const __half2 p = __floats2half2_rn(v.x, v.y);
    const __half2_raw raw=p;
    return (unsigned(raw.y)<<16)|raw.x;
}
__device__ __forceinline__ void multiply(float (&c)[4], const uint4& a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b0),"r"(b1));
}

template<int Tokens, int WarpTokens, bool Histogram>
__device__ __forceinline__ void score_impl(const float* query, int ldq, const __half* raw, const Cache* state,
        const float* weights, const int* pos, int capacity, float* scores, int lds, unsigned* global_hist) {
    constexpr int Warps=4, PerToken=WarpTokens==1?Tokens:1;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int first_token = WarpTokens==1?0:warp%WarpTokens;
    if (first_token >= Tokens) return;
    const int p = *pos, groups=min(capacity,(p+Tokens-1)/4);
    if (p+Tokens <= 2048) return;
    const int first=int(blockIdx.x)*(Warps/WarpTokens)+warp/WarpTokens;
    if (first*8 >= groups) return;
    // PTX m16n8k16 A fragment order spans the two 8-head halves in each 8-K slice.
    // Some token/warp masks return above. Each active warp owns and reduces its histogram independently.
    __shared__ unsigned local_hist[Histogram ? Warps : 1][Histogram ? PerToken : 1][Histogram ? 256 : 1];
    if constexpr (Histogram) {
        for (int t = 0; t < PerToken; ++t)
            for (int i = lane; i < 256; i += 32) local_hist[warp][t][i] = 0;
        __syncwarp();
    }
    uint4 q[PerToken][8];
#pragma unroll
    for(int t=0;t<PerToken;++t) {
        const float* row=query+size_t(first_token+t)*ldq+(lane/4)*128+2*(lane%4);
#pragma unroll
        for(int k=0;k<8;++k) {
            const float* lo=row+16*k;
            q[t][k]=make_uint4(fp16_pair(lo),fp16_pair(lo+8*128),
                               fp16_pair(lo+8),fp16_pair(lo+8*128+8));
        }
    }
    // The stride places eight independent 16-byte score reads in distinct banks.
    __shared__ __align__(16) float dots[Warps][8][20];
    __shared__ __align__(16) float scale[Warps][PerToken][16];
    if(lane<16) {
#pragma unroll
        for(int t=0;t<PerToken;++t) scale[warp][t][lane]=weights[(first_token+t)*16+lane];
    }
    __syncwarp();
    const Cache cache=*state;
    const int stride=gridDim.x*(Warps/WarpTokens);
    auto score_tile = [&](int tile, auto packed) {

        float accum[PerToken][4]={};
#pragma unroll
        for(int k=0;k<4;++k) {
            uint4 b;
            if constexpr(decltype(packed)::value)
                b=cache.base[(size_t(tile)*4+k)*32+lane];
            else {
                const int row=8*tile+lane/4;
                const auto* pairs=reinterpret_cast<const unsigned*>(raw)+size_t(row)*64+k*16+lane%4;
                b=row<groups ? make_uint4(pairs[0],pairs[4],pairs[8],pairs[12]) : make_uint4(0,0,0,0);
            }
#pragma unroll
            for(int t=0;t<PerToken;++t) {
                multiply(accum[t],q[t][2*k],b.x,b.y);
                multiply(accum[t],q[t][2*k+1],b.z,b.w);
            }
        }
#pragma unroll
        for(int t=0;t<PerToken;++t) {
            const int head=lane/4,col=2*(lane%4);
            dots[warp][col][head]=accum[t][0];
            dots[warp][col+1][head]=accum[t][1];
            dots[warp][col][head+8]=accum[t][2];
            dots[warp][col+1][head+8]=accum[t][3];
            __syncwarp();
            const int tt=first_token+t,g=tile*8+lane;
            if(lane<8 && g<(p+tt)/4 && p+tt+1>2048) {
                float sum=0.f;
#pragma unroll
                for(int h=0;h<16;h+=4) {
                    const float4 d=*reinterpret_cast<const float4*>(&dots[warp][lane][h]);
                    const float4 w=*reinterpret_cast<const float4*>(&scale[warp][t][h]);
                    sum+=fmaxf(d.x,0.f)*w.x;
                    sum+=fmaxf(d.y,0.f)*w.y;
                    sum+=fmaxf(d.z,0.f)*w.z;
                    sum+=fmaxf(d.w,0.f)*w.w;
                }
                scores[size_t(tt)*lds+g]=sum;
                if constexpr (Histogram) {
                    const unsigned bin=kern::kd_score_coarse(sum);
                    atomicAdd(&local_hist[warp][t][bin >> 1], 1u << (16 * (bin & 1u)));
                }
            }
            __syncwarp();
        }
    };
    int tile=first;
    for (; tile<min(cache.tiles,(groups+7)/8); tile+=stride) score_tile(tile,std::true_type{});
    for (; tile*8<groups; tile+=stride) score_tile(tile,std::false_type{});
    if constexpr (Histogram) {
        __syncwarp();
        for (int t = 0; t < PerToken; ++t) {
            unsigned* hist = global_hist + (first_token + t) * 512;
            for (int i = lane; i < 256; i += 32) {
                const unsigned counts = local_hist[warp][t][i];
                if (counts & 65535u) atomicAdd(hist + 2*i, counts & 65535u);
                if (counts >> 16) atomicAdd(hist + 2*i + 1, counts >> 16);
            }
        }
    }
}

template<int Tokens, int WarpTokens>
__global__ __launch_bounds__(128) void score(const float* query, int ldq, const __half* raw, const Cache* state,
        const float* weights, const int* pos, int capacity, float* scores, int lds) {
    score_impl<Tokens,WarpTokens,false>(query,ldq,raw,state,weights,pos,capacity,scores,lds,nullptr);
}

template<int Tokens, int WarpTokens>
__global__ __launch_bounds__(128) void score_hist(const float* query, int ldq, const __half* raw, const Cache* state,
        const float* weights, const int* pos, int capacity, float* scores, int lds, unsigned* histogram) {
    score_impl<Tokens,WarpTokens,true>(query,ldq,raw,state,weights,pos,capacity,scores,lds,histogram);
}
}  // namespace strata::hy4f::packed_index

