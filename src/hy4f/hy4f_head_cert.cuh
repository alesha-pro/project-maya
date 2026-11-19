// Certified per-shard BF16 top-1/top-2 using only Q8_0 for candidate generation.
#pragma once
#include "hy4f_dkern.cuh"
#include <cmath>
namespace strata::hy4f::kern {
inline constexpr int cert_tile = 256;
struct HeadCertificate {
    float2* weight = nullptr;  // upper bounds for ||W-Wq||, ||Wq||, per vocabulary row
    float2* input = nullptr;   // upper bounds for ||x||, ||x-xq||, per verify position
    float2* lower = nullptr;   // two largest lower endpoints in each tile
    float* upper = nullptr;
    int* count = nullptr;
    int* ids = nullptr;        // union of surviving rows over the verify positions
};

__global__ void kd_head_input_norm(const float* x, const Q81* q, int n, float2* out) {
    const int t = blockIdx.x;
    double sx = 0, sd = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float val = x[t*n + i];
        const Q81& blk = q[(t*n+i)/32];
        const float scale = __half2float(blk.ds.x);
        if (!isfinite(val) || fabsf(val) > 0x1p30f || !isfinite(scale)) { sx = INFINITY; break; }
        const double decoded = double(scale) * double(blk.qs[i%32]);
        // Conversion of a float subnormal may flush under --use_fast_math.
        // The absolute allowance in kd_head_bounds covers that contribution.
        const double z = double(val), e = fabs(z-decoded);
        sx = __dadd_ru(sx, __dmul_ru(z,z));
        const double eu = __dmul_ru(e, 1.0 + 0x1p-51);
        sd = __dadd_ru(sd, __dmul_ru(eu,eu));
    }
    for (int off=16; off; off/=2) {
        sx=__dadd_ru(sx,__shfl_down_sync(~0u,sx,off));
        sd=__dadd_ru(sd,__shfl_down_sync(~0u,sd,off));
    }
    __shared__ double a[8], b[8];
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    if (!lane) { a[warp]=sx; b[warp]=sd; }
    __syncthreads();
    if (!warp) {
        sx=lane<8?a[lane]:0.; sd=lane<8?b[lane]:0.;
        for (int off=4;off;off/=2) {
            sx=__dadd_ru(sx,__shfl_down_sync(~0u,sx,off));
            sd=__dadd_ru(sd,__shfl_down_sync(~0u,sd,off));
        }
        if (!lane) out[t]=make_float2(__double2float_ru(__dsqrt_ru(sx)),__double2float_ru(__dsqrt_ru(sd)));
    }
}

__device__ __forceinline__ void head_two(float2& p, float f) {
    if (f >= p.x) { p.y=p.x; p.x=f; }
    else p.y=fmaxf(p.y,f);
}
__device__ __forceinline__ float2 head_warp_two(float2 p) {
    for (int off=16;off;off/=2) {
        float hi=__shfl_down_sync(~0u,p.x,off),lo=__shfl_down_sync(~0u,p.y,off);
        head_two(p,hi); head_two(p,lo);
    }
    return p;
}
__global__ void kd_head_bounds(const float* y, int rows, HeadCertificate c) {
    const int t=blockIdx.y, i=blockIdx.x*cert_tile+threadIdx.x;
    float lower=-INFINITY;
    if (i<rows) {
        const float2 w=c.weight[i], n=c.input[t];
        const float eps=0x1p-12f;
        // The factor covers both dot products' FP32 accumulation. Model weights and
        // inputs within 2^30 bound all intermediates. The offset covers FTZ absolutely.
        const float bw=__fadd_ru(w.x,w.y), bq=__fadd_ru(n.x,n.y);
        const float round=__fmul_ru(eps,__fadd_ru(__fmul_ru(bw,n.x),__fmul_ru(w.y,bq)));
        const float err=__fadd_ru(0x1p-64f,__fadd_ru(round,__fadd_ru(__fmul_ru(w.x,n.x),__fmul_ru(w.y,n.y))));
        const float v=y[(size_t)t*rows+i];
        const bool good=isfinite(n.x)&&isfinite(n.y)&&isfinite(err)&&isfinite(v);
        float upper=good?__fadd_ru(v,err):INFINITY;
        if(good) lower=__fsub_rd(v,err);
        if(fabsf(upper)<0x1p-126f) upper=0x1p-126f;
        if(fabsf(lower)<0x1p-126f) lower=-0x1p-126f;
        c.upper[(size_t)t*rows+i]=upper;
    }
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    float2 p=head_warp_two(make_float2(lower,-INFINITY));
    __shared__ float2 best[8];
    if (!lane) best[warp]=p;
    __syncthreads();
    if (!warp) {
        p=head_warp_two(lane<8?best[lane]:make_float2(-INFINITY,-INFINITY));
        if (!lane) c.lower[t*gridDim.x+blockIdx.x]=p;
    }
    if (!threadIdx.x && !blockIdx.x && !t) *c.count=0;
}
// One warp per verification position derives its cut from the per-tile lower
// endpoints. Inclusive comparison retains every tie at the exact cutoff.
template<int T> __global__ void kd_head_list(float* y,int rows,int tiles,bool two,HeadCertificate c) {
    const int lane=threadIdx.x%32, warp=threadIdx.x/32, i=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ float cut[T];
    if (warp<T) {
        float2 p=make_float2(-INFINITY,-INFINITY);
        for (int z=lane;z<tiles;z+=32) {float2 b=c.lower[warp*tiles+z];head_two(p,b.x);head_two(p,b.y);}
        p=head_warp_two(p);
        if (!lane) cut[warp]=two?p.y:p.x;
    }
    __syncthreads();
    bool keep=false;
    if(i<rows) {
#pragma unroll
        for(int t=0;t<T;++t) keep |= c.upper[(size_t)t*rows+i]>=cut[t];
        if(!keep) {
#pragma unroll
            for(int t=0;t<T;++t)y[(size_t)t*rows+i]=-INFINITY;
        }
    }
    const unsigned mask=__ballot_sync(~0u,keep);
    if(mask) {
        int base=lane==0?atomicAdd(c.count,__popc(mask)):0;
        base=__shfl_sync(~0u,base,0);
        if(keep) c.ids[base+__popc(mask & ((1u<<lane)-1))]=i;
    }
}
template<int T> __global__ void kd_head_rerank(const uint16_t* weights,const float* x,int width,int rows,
                                              float* y,HeadCertificate c) {
    const int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int step=gridDim.x*(blockDim.x/32);
    for(int i=warp;i<*c.count;i+=step)
        kd_head_bf16_row<T>(weights,x,width,rows,y,c.ids[i]);
}
} // namespace strata::hy4f::kern
