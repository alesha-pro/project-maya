// Four-way KV shards, four positions per selected pool. A score CTA covers
// 32 selected rows at once. The two 288-wide QK partials and 32-row softmax
// steps have the same accumulation order as kb_mqa_attn<32>.
#pragma once

namespace strata::hy4f::kern {
__device__ __forceinline__ uint4 dsa_half8(const uint8_t* row, int col) {
    uint4 v = make_uint4(0,0,0,0);
    if (row) {
        const int2 packed = *reinterpret_cast<const int2*>(row+col);
        const float scale = __half2float(reinterpret_cast<const __half*>(row+KV_ROW)[col/64]);
        const auto* bytes = reinterpret_cast<const int8_t*>(&packed);
        auto* dst = reinterpret_cast<__half2*>(&v);
#pragma unroll
        for (int i=0; i<4; ++i) dst[i] = __floats2half2_rn(bytes[2*i]*scale, bytes[2*i+1]*scale);
    }
    return v;
}

__global__ void kb_dsa_expand(const uint8_t* kv, __half* out) {
    const int row=blockIdx.x, col=threadIdx.x*8;
    if(col<KV_ROW) *reinterpret_cast<uint4*>(out+(size_t)row*KV_ROW+col)=dsa_half8(kv+(size_t)row*KV_RB,col);
}

__global__ void kb_dsa_scores(const __half* queries, const uint8_t* kv, const int* cells,
                              const uint16_t* wire, int groups, int top_k, int cap, int start,
                              int shard, float* scores, const __half* expanded) {
    using namespace nvcuda;
    constexpr int D=576, H=32, K=144, LD=152, N=32;
    __shared__ __align__(32) __half a[2][H][LD], b[2][N][LD];
    __shared__ const uint8_t* ptr[N];
    __shared__ const __half* fp[N];
    int t=blockIdx.y, pos=start+t, tid=threadIdx.x, warp=tid/32;
    int tile=blockIdx.x*N, split=warp/4, h0=((warp/2)&1)*16, r0=(warp&1)*16;
    int n=grouped_rows(pos,top_k,groups,cap,shard);
    if (tile >= n) return;
    if (tid<N) {
        int j=tile+tid;
        const int id=j<n?grouped_kv_row(j,pos,top_k,groups,cells+(size_t)t*cap,
            wire ? wire+(size_t)t*((groups+7)/8*8) : nullptr):0;
        ptr[tid]=j<n?kv+(size_t)id*KV_RB:nullptr;
        fp[tid]=j<n && expanded ? expanded+(size_t)id*D:nullptr;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for (int step=0;step<2;++step) {
        for (int i=tid;i<2*H*(K/8);i+=256) {
            const int sp=i/(H*(K/8)), row=(i/(K/8))%H, c=(i%(K/8))*8;
            const int k=sp*288+step*K+c;
            *reinterpret_cast<uint4*>(&a[sp][row][c])=*reinterpret_cast<const uint4*>(queries+((size_t)t*H+row)*D+k);
            *reinterpret_cast<uint4*>(&b[sp][row][c])=expanded ? (fp[row]?*reinterpret_cast<const uint4*>(fp[row]+k):make_uint4(0,0,0,0)):dsa_half8(ptr[row],k);
        }
        __syncthreads();
#pragma unroll
        for(int k=0;k<K;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> fb;
            wmma::load_matrix_sync(fa,&a[split][h0][k],LD);
            wmma::load_matrix_sync(fb,&b[split][r0][k],LD);
            wmma::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads();
    }
    float* sum=reinterpret_cast<float*>(a);
    wmma::store_matrix_sync(sum+(split*H+h0)*N+r0,acc,N,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<H*N;i+=256)
        scores[((size_t)t*H+i/N)*512+tile+i%N] = (0.f+sum[i])+sum[H*N+i];
}

__global__ void kb_dsa_probs(const float* scores, int start, int top_k, int groups, int cap,
                             int shard, __half* probs, float* part, int stride) {
    const int t=blockIdx.x,h=threadIdx.x/32+blockIdx.y*8,lane=threadIdx.x%32;
    const int n=grouped_rows(start+t,top_k,groups,cap,shard);
    const size_t off=((size_t)t*32+h)*512+lane;
    float m=-INFINITY,l=0.f,s[16];
    for(int r=0;r<n;r+=32) {
        const bool valid=r+lane<n;
        const float v=valid?scores[off+r]:-INFINITY;
        s[r/32]=v;
        const float next=fmaxf(m,warp_max(v));
        const float e=warp_sum(valid?__expf(v-next):0.f);
        l=l*(m==-INFINITY?0.f:__expf(m-next))+e;
        m=next;
    }
    for(int r=0;r<n;r+=32)
        probs[off+r]=__float2half_rn(r+lane<n && m!=-INFINITY ? __expf(s[r/32]-m):0.f);
    if(lane==0) {
        part[((size_t)t*32+h)*stride+512]=m;
        part[((size_t)t*32+h)*stride+513]=l;
    }
}

__global__ void kb_dsa_values(const __half* probs,const uint8_t* kv,const int* cells,
                              const uint16_t* wire,int groups,int top_k,int cap,int start,
                              int shard,float* out,int stride,const __half* expanded) {
    using namespace nvcuda;
    constexpr int C=64, LD=72;
    __shared__ __align__(32) __half p[32][40], v[32][LD];
    __shared__ const uint8_t* ptr[32];
    __shared__ const __half* fp[32];
    const int t=blockIdx.y,tid=threadIdx.x,w=tid/32,pos=start+t;
    const int n=grouped_rows(pos,top_k,groups,cap,shard),col=blockIdx.x*C;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
    for(int g=0;g<n;g+=32) {
        if(tid<32) {
            int id=g+tid<n?grouped_kv_row(g+tid,pos,top_k,groups,cells+(size_t)t*cap,
                wire?wire+(size_t)t*((groups+7)/8*8):nullptr):0;
            ptr[tid]=g+tid<n?kv+(size_t)id*KV_RB:nullptr;
            fp[tid]=g+tid<n && expanded?expanded+(size_t)id*KV_ROW:nullptr;
        }
        if(tid<128) {
            const int h=tid/4,k=tid%4*8;
            *reinterpret_cast<uint4*>(&p[h][k])=*reinterpret_cast<const uint4*>(probs+((size_t)t*32+h)*512+g+k);
        }
        __syncthreads();
        const int row=tid/8,c=tid%8*8;
        *reinterpret_cast<uint4*>(&v[row][c])=expanded ? (fp[row]?*reinterpret_cast<const uint4*>(fp[row]+col+c):make_uint4(0,0,0,0)):dsa_half8(ptr[row],col+c);
        __syncthreads();
#pragma unroll
        for(int k=0;k<32;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb;
            wmma::load_matrix_sync(fa,&p[(w/4)*16][k],40);
            wmma::load_matrix_sync(fb,&v[k][(w%4)*16],LD);
            wmma::mma_sync(acc,fa,fb,acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(out+((size_t)t*32+(w/4)*16)*stride+col+(w%4)*16,acc,stride,wmma::mem_row_major);
}
} // namespace strata::hy4f::kern
