#pragma once

namespace strata::hy4f::packed_q3 {
// Two adjacent original Q3_K blocks, one per half warp. Ranked integer
// scales occupy four 24-bit fields per block. Their six-bit signed values
// preserve the source's offset-binary scales exactly.
struct Pair {
    unsigned lo[32], hi[16], scales[6];
    __half d[2];
};
static_assert(sizeof(Pair) == 220);
struct Q81 {
    __half2 ds;
    int qs[8];
};
__device__ __forceinline__ float sumwarp(float v) {
#pragma unroll
    for (int i = 16; i; i >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, i);
    return v;
}
// Lane-to-block assignment and the final XOR sum match native grouped
// MMVQ. Only the weight loads and integer unpacking differ.
template <int Slots, int Rows>
__device__ __forceinline__ void dot(const Pair *row, const Q81 *x, const int *slots, float *y, int r) {
    const int lane = threadIdx.x & 31, half = lane >> 4, iq = lane & 15;
    const int ix = (half * 4 + iq / 4) * 3, bit = 8 * (ix % 4);
    int sid[Slots];
    float sum[Slots] = {};
#pragma unroll
    for (int t = 0; t < Slots; t++)
        sid[t] = slots[t];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const Pair &p = row[k];
        unsigned l = p.lo[lane], h = p.hi[half * 8 + iq % 8] >> (4 * (iq / 8));
        unsigned ab = p.scales[ix / 4], b = ix % 4 > 1 ? p.scales[ix / 4 + 1] : 0u;
        unsigned sc = __funnelshift_r(ab, b, bit);
        float sub[Slots] = {};
        const Q81 *a0 = x + (k * 2 + half) * 8 + 4 * (iq / 8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int f;
            asm("bfe.s32 %0,%1,%2,6;" : "=r"(f) : "r"(sc), "r"(6 * j));
            const unsigned v = ((l >> (2 * j)) & 0x03030303u) | (63u * (((h >> j) << 2) & 0x04040404u));
#pragma unroll
            for (int t = 0; t < Slots; t++) {
                const auto &a = a0[sid[t] * 48 + j];
                sub[t] += __low2float(a.ds) * float(__dp4a(int(v), a.qs[iq % 8], 0) * f);
            }
        }
#pragma unroll
        for (int t = 0; t < Slots; t++)
            sum[t] += __half2float(p.d[half]) * sub[t];
    }
#pragma unroll
    for (int t = 0; t < Slots; t++) {
        float v = sumwarp(sum[t]);
        if (lane == 0)
            y[sid[t] * Rows + r] = v;
    }
}
template <int Rows>
__global__ void packed1536(const Pair *weights, size_t eb, const int *ng, const int *gexp, const int *off, const int *slots,
                           const Q81 *x, float *y) {
    const int n = *ng * Rows;
    for (int it = blockIdx.x * 4 + (threadIdx.x >> 5); it < n; it += gridDim.x * 4) {
        int g = it / Rows, r = it % Rows;
        const auto *wr = weights + eb * gexp[g] + 3 * r;
        int o = off[g], s = off[g + 1] - o;
        if (s == 1)
            dot<1, Rows>(wr, x, slots + o, y, r);
        else if (s == 2)
            dot<2, Rows>(wr, x, slots + o, y, r);
        else if (s == 3)
            dot<3, Rows>(wr, x, slots + o, y, r);
        else if (s == 4)
            dot<4, Rows>(wr, x, slots + o, y, r);
    }
}
inline void forward(const void *w, size_t eb, const int *ng, const int *gexp, const int *off, const int *slot, const void *x,
                    float *y, int rows, int blocks, cudaStream_t st) {
    if (rows == 1536)
        packed1536<1536>
            <<<blocks * 2, 128, 0, (cudaStream_t)st>>>((const Pair *)w, eb / 220, ng, gexp, off, slot, (const Q81 *)x, y);
    else
        packed1536<3072>
            <<<blocks * 2, 128, 0, (cudaStream_t)st>>>((const Pair *)w, eb / 220, ng, gexp, off, slot, (const Q81 *)x, y);
}

// Each warp owns one pair. Stage it before writing because this operation
// permutes bytes in place. The two layouts contain exactly the same 220 bytes
// of information (including the signed per-group scales).
__device__ __forceinline__ int raw_scale(const unsigned char *s, int idx) {
    return ((s[idx & 7] >> (4 * (idx >> 3))) & 15) | (((s[8 + (idx & 3)] >> (2 * (idx >> 2))) & 3) << 4);
}
__device__ __forceinline__ int get_scale(const unsigned char *p, int half, int index) {
    const int z = (index / 8) * 2 + (index & 1);
    const unsigned char *a = p + 192 + 12 * half + 3 * z;
    const unsigned w = unsigned(a[0]) | (unsigned(a[1]) << 8) | (unsigned(a[2]) << 16);
    return int((w >> (6 * ((index & 7) / 2))) & 63u) ^ 32;
}
template <bool Pack> __global__ void convert_pairs(unsigned *data, int pairs) {
    constexpr int lanes = 32, words = 55;
    __shared__ unsigned raw[8][words];
    const int warp = threadIdx.x / lanes, lane = threadIdx.x % lanes;
    const auto *z = reinterpret_cast<const unsigned char *>(raw[warp]);
    for (int pair = blockIdx.x * 8 + warp; pair < pairs; pair += gridDim.x * 8) {
        unsigned *w = data + size_t(pair) * words;
        auto *dst = reinterpret_cast<unsigned char *>(w);
        raw[warp][lane] = w[lane];
        if (lane + lanes < words)
            raw[warp][lane + lanes] = w[lane + lanes];
        __syncwarp();
        if constexpr (Pack) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const auto *b = z + 110 * h;
                dst[64 * h + lane] = b[32 + lane];
                dst[64 * h + 32 + lane] = b[64 + lane];
                dst[128 + 32 * h + lane] = b[lane] ^ 255;
                if (lane < 12) {
                    const int q = lane / 3, byte = lane % 3;
                    unsigned sc = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        sc |= unsigned(raw_scale(b + 96, (q / 2) * 8 + q % 2 + 2 * j) ^ 32) << (6 * j);
                    dst[192 + 12 * h + lane] = (sc >> (8 * byte)) & 255;
                }
                if (lane < 2)
                    dst[216 + h * 2 + lane] = b[108 + lane];
            }
        } else {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                auto *b = dst + 110 * h;
                b[lane] = z[128 + 32 * h + lane] ^ 255;
                b[32 + lane] = z[64 * h + lane];
                b[64 + lane] = z[64 * h + 32 + lane];
                if (lane < 8)
                    b[96 + lane] = (get_scale(z, h, lane) & 15) | ((get_scale(z, h, lane + 8) & 15) << 4);
                if (lane < 4) {
                    unsigned hi = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        hi |= unsigned(get_scale(z, h, lane + 4 * j) >> 4) << (2 * j);
                    b[104 + lane] = hi;
                }
                if (lane < 2)
                    b[108 + lane] = z[216 + h * 2 + lane];
            }
        }
        __syncwarp();
    }
}

} // namespace strata::hy4f::packed_q3
