// Full SWA windows, 16 local heads. Keep the original split-K QK rounding,
// 32-key softmax recurrence and left-to-right WMMA probability/value product.
#pragma once

namespace strata::hy4f::kern {
template<int H>
__global__ void kb_mqa_queries(const float* qabs, const float* q, int ldq, int qs,
                               int nope, float theta, int start, float scale, __half* dst, const int* device_start = nullptr) {
    constexpr int D = 576;
    const int t = blockIdx.y, c0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (c0 >= H*D) return;
    const int h = c0 / D, c = c0 % D;
    float v;
    if (c < 512) v = qabs[(((size_t)t*Groups+blockIdx.z)*H+h)*512+c];
    else {
        int j = c-512, jj=j/2;
        const float* pe = q + (size_t)t*ldq + h*qs + nope + 2*jj;
        float sn, cs;
        const int pos = (device_start ? *device_start : start) + t;
        sincosf((float)pos * powf(theta, -2.0f*jj/64), &sn, &cs);
        v = (j&1) ? pe[0]*sn + pe[1]*cs : pe[0]*cs - pe[1]*sn;
    }
    dst[(size_t)t*H*D+c0] = __float2half_rn(v*scale);
}

template<int Groups = 1>
__global__ void kb_swa_scores(const __half* queries, const __half* kv, int base, int nt, float* scores) {
    using namespace nvcuda;
    constexpr int Q=4, N=32, H=16, D=576, K=144, LD=160;
    __shared__ __align__(32) __half a[Q][H][LD], b[N+Q-1][LD];
    const int tid=threadIdx.x, warp=tid/32, ti=warp/2, nj=(warp%2)*16;
    const int tb=Q*blockIdx.y, gb=N*blockIdx.x;
    const int active_q=min(Q,nt-tb);
    wmma::fragment<wmma::accumulator,16,16,16,float> sum;
    wmma::fill_fragment(sum,0.f);
    for (int k0=0;k0<D;k0+=K) {
        for (int i=tid;i<Q*H*(K/8);i+=256) {
            const int row=i/(K/8), c=i%(K/8)*8;
            uint4 v=make_uint4(0,0,0,0);
            if (row/H<active_q) v=*reinterpret_cast<const uint4*>(queries+(((size_t)tb+row/H)*H*Groups+blockIdx.z*H+row%H)*D+k0+c);
            *reinterpret_cast<uint4*>(&a[row/H][row%H][c])=v;
        }
        for (int i=tid;i<(N+active_q-1)*(K/8);i+=256) {
            const int row=i/(K/8), c=i%(K/8)*8;
            *reinterpret_cast<uint4*>(&b[row][c])=
                *reinterpret_cast<const uint4*>(kv+(size_t)(base+tb+gb+row)*D+k0+c);
        }
        __syncthreads();
        if (ti<active_q) {
            wmma::fragment<wmma::accumulator,16,16,16,float> part;
            wmma::fill_fragment(part,0.f);
#pragma unroll
            for (int k=0;k<K;k+=16) {
                wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
                wmma::load_matrix_sync(fa,&a[ti][0][k],LD);
                wmma::load_matrix_sync(fb,&b[ti+nj][k],LD);
                wmma::mma_sync(part,fa,fb,part);
            }
#pragma unroll
            for (int z=0;z<sum.num_elements;++z) sum.x[z]+=part.x[z];
        }
        __syncthreads();
    }
    if (ti<active_q)
        wmma::store_matrix_sync(scores+((size_t)(tb+ti)*Groups+blockIdx.z)*H*512+gb+nj,sum,512,wmma::mem_row_major);
}

template<int Groups = 1>
__global__ void kb_swa_probs(const float* scores, const float* sinks, __half* probs) {
    const int h=threadIdx.x/32, lane=threadIdx.x%32;
    const size_t i=(((size_t)blockIdx.x*Groups+blockIdx.y)*16+h)*512+lane;
    float m=-INFINITY,l=0.f,score[16];
#pragma unroll
    for (int s=0;s<16;++s) {
        float v=scores[i+s*32];
        score[s]=v;
        float next=fmaxf(m,warp_max(v));
        const float e=warp_sum(__expf(v-next));
        l=l*(m == -INFINITY ? 0.0f : __expf(m-next))+e;
        m=next;
    }
    const float end=fmaxf(m,sinks[blockIdx.y*16+h]);
    const float inv=1.0f/(l*__expf(m-end)+__expf(sinks[blockIdx.y*16+h]-end));
#pragma unroll
    for (int s=0;s<16;++s) probs[i+s*32]=__float2half_rn(__expf(score[s]-end)*inv);
}

template<int Groups = 1>
__global__ void kb_swa_values(const __half* probs, const __half* kv, int base, float* out) {
    using namespace nvcuda;
    constexpr int H=16, D=576, N=128, LD=136;
    __shared__ __align__(32) __half p[H][520], v[32][LD];
    const int tid=threadIdx.x,w=tid/32,t=blockIdx.y, col0=blockIdx.x*N;
    for (int i=tid;i<H*64;i+=256) {
        const int h=i/64,c=i%64*8;
        *reinterpret_cast<uint4*>(&p[h][c])=
            *reinterpret_cast<const uint4*>(probs+(((size_t)t*Groups+blockIdx.z)*H+h)*512+c);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for (int g=0;g<512;g+=32) {
        for (int i=tid;i<32*(N/8);i+=256) {
            const int row=i/(N/8), c=i%(N/8)*8;
            *reinterpret_cast<uint4*>(&v[row][c])=
                *reinterpret_cast<const uint4*>(kv+(size_t)(base+t+g+row)*D+col0+c);
        }
        __syncthreads();
#pragma unroll
        for(int k=0;k<32;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
            wmma::load_matrix_sync(a,&p[0][g+k],520);
            wmma::load_matrix_sync(b,&v[k][16*w],LD);
            wmma::mma_sync(acc,a,b,acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(out+((size_t)t*Groups+blockIdx.z)*H*512+col0+16*w,acc,512,wmma::mem_row_major);
}
}  // namespace strata::hy4f::kern
