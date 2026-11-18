// src/hy4f/hy4f_vision_pre.cpp - host side of the Hy4 flash vision encoder: image decode (stb_image) and the exact
// llama.cpp preprocessing (smart resize size + Pillow-compatible Lanczos-3 resample, tools/mtmd/mtmd-image.cpp).
#include "hy4f_vision.hpp"
#include "hy4f_vision_pre.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace strata::hy4f::vision {

bool load_image(const std::string& path, Image& out, std::string& err) {
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        err = "load_image: " + path + ": " + (stbi_failure_reason() ? stbi_failure_reason() : "decode failed");
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgb.assign(data, data + (size_t) w * h * 3);
    stbi_image_free(data);
    return true;
}

bool load_image_mem(const uint8_t* bytes, size_t n, Image& out, std::string& err) {
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load_from_memory(bytes, (int) n, &w, &h, &c, 3);
    if (!data) {
        err = std::string("load_image_mem: ") + (stbi_failure_reason() ? stbi_failure_reason() : "decode failed");
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgb.assign(data, data + (size_t) w * h * 3);
    stbi_image_free(data);
    return true;
}

namespace detail {

// img_tool::calc_size_preserved_ratio with longest_edge = 0, round_half_even = false (the hy_v4_flash hparams)
void smart_size(int width, int height, int align, int min_pixels, int max_pixels, int& tw, int& th) {
    if (width <= 0 || height <= 0) {
        tw = th = 0;
        return;
    }
    auto round_by_factor = [f = align](float x) {
        const float q = x / static_cast<float>(f);
        return static_cast<int>(std::round(q)) * f;
    };
    auto ceil_by_factor = [f = align](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
    auto floor_by_factor = [f = align](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };
    int w_bar = std::max(align, round_by_factor((float) width));
    int h_bar = std::max(align, round_by_factor((float) height));
    if (max_pixels > 0 && h_bar * w_bar > max_pixels) {
        const auto beta = std::sqrt(static_cast<float>(height) * width / max_pixels);
        h_bar = std::max(align, floor_by_factor(height / beta));
        w_bar = std::max(align, floor_by_factor(width / beta));
    } else if (min_pixels > 0 && h_bar * w_bar < min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(min_pixels) / (static_cast<float>(height) * width));
        h_bar = ceil_by_factor(height * beta);
        w_bar = ceil_by_factor(width * beta);
    }
    tw = w_bar;
    th = h_bar;
}

namespace {

constexpr int PRECISION_BITS = 32 - 8 - 2;

double lanczos3(double x) {
    if (-3.0 <= x && x < 3.0) {
        auto sinc = [](double v) {
            if (v == 0.0) return 1.0;
            const double pi_v = v * 3.141592653589793238462643383279502884;
            return std::sin(pi_v) / pi_v;
        };
        return sinc(x) * sinc(x / 3.0);
    }
    return 0.0;
}

inline uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t) v; }

// Pillow's precompute_coeffs (ImagingResample), as ported in mtmd-image.cpp
int precompute_weights(int inSize, int outSize, std::vector<int>& bounds, std::vector<int32_t>& weights) {
    const double filter_support = 3.0;
    double filterscale, scale;
    filterscale = scale = static_cast<double>(inSize) / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = filter_support * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<double> pre((size_t) outSize * ksize);
    bounds.resize((size_t) outSize * 2);
    for (int xx = 0; xx < outSize; xx++) {
        const double center = (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; x++) {
            const double w = lanczos3((x + xmin - center + 0.5) * ss);
            pre[(size_t) xx * ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++)
            if (ww != 0.0) pre[(size_t) xx * ksize + x] /= ww;
        for (; x < ksize; x++) pre[(size_t) xx * ksize + x] = 0;
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    weights.resize((size_t) outSize * ksize);
    const double fxp_scale = std::ldexp(1.0, PRECISION_BITS);
    for (size_t i = 0; i < (size_t) outSize * ksize; i++) {
        const double rounded = pre[i] * fxp_scale + (pre[i] < 0 ? -0.5 : 0.5);
        weights[i] = static_cast<int32_t>(rounded);
    }
    return ksize;
}

template <class F> void parallel_rows(int n, F&& f) {
    const int nt = std::max(1, std::min<int>(std::min(16u, std::max(1u, std::thread::hardware_concurrency())), n / 32));
    if (nt <= 1) {
        f(0, n);
        return;
    }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        const int a = (int) ((int64_t) n * t / nt), b = (int) ((int64_t) n * (t + 1) / nt);
        th.emplace_back([&f, a, b] { f(a, b); });
    }
    for (auto& x : th) x.join();
}

std::vector<uint8_t> resample_horizontal(const uint8_t* src, int in_nx, int in_ny, int out_nx, int ksize,
                                         const std::vector<int>& bounds, const std::vector<int32_t>& weights) {
    std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);
    parallel_rows(in_ny, [&](int y0, int y1) {
        for (int yy = y0; yy < y1; yy++) {
            const uint8_t* src_row = src + (size_t) yy * in_nx * 3;
            uint8_t* dst_row = out.data() + (size_t) yy * out_nx * 3;
            for (int xx = 0; xx < out_nx; xx++) {
                const int xmin = bounds[xx * 2 + 0], xcnt = bounds[xx * 2 + 1];
                const int32_t* k = &weights[(size_t) xx * ksize];
                const uint8_t* p = src_row + (size_t) xmin * 3;
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xcnt; x++) {
                    s0 += p[0] * k[x];
                    s1 += p[1] * k[x];
                    s2 += p[2] * k[x];
                    p += 3;
                }
                dst_row[xx * 3 + 0] = clip8(s0 >> PRECISION_BITS);
                dst_row[xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
                dst_row[xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
            }
        }
    });
    return out;
}

std::vector<uint8_t> resample_vertical(const uint8_t* src, int in_nx, int out_ny, int ksize, const std::vector<int>& bounds,
                                       const std::vector<int32_t>& weight) {
    const size_t row = (size_t) in_nx * 3;
    std::vector<uint8_t> out(row * out_ny);
    parallel_rows(out_ny, [&](int y0, int y1) {
        std::vector<int32_t> acc(row);
        for (int yy = y0; yy < y1; yy++) {
            const int ymin = bounds[yy * 2 + 0], ycnt = bounds[yy * 2 + 1];
            const int32_t* k = &weight[(size_t) yy * ksize];
            std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
            for (int y = 0; y < ycnt; y++) {
                const uint8_t* s = src + (size_t) (ymin + y) * row;
                const int32_t w = k[y];
                for (size_t i = 0; i < row; i++) acc[i] += s[i] * w;
            }
            uint8_t* d = out.data() + (size_t) yy * row;
            for (size_t i = 0; i < row; i++) d[i] = clip8(acc[i] >> PRECISION_BITS);
        }
    });
    return out;
}

}  // namespace

void resize_lanczos(const Image& src, int tw, int th, Image& dst) {
    if (tw <= 0 || th <= 0 || tw > 65536 || th > 65536) throw std::runtime_error("resize target out of range");
    dst.w = tw;
    dst.h = th;
    if (tw == src.w && th == src.h) {
        dst.rgb = src.rgb;
        return;
    }
    const bool nh = tw != src.w, nv = th != src.h;
    std::vector<int> bh, bv;
    std::vector<int32_t> wh, wv;
    int kh = 0, kv = 0;
    if (nh) kh = precompute_weights(src.w, tw, bh, wh);
    if (nv) kv = precompute_weights(src.h, th, bv, wv);
    if (nh && nv) {
        auto tmp = resample_horizontal(src.rgb.data(), src.w, src.h, tw, kh, bh, wh);
        dst.rgb = resample_vertical(tmp.data(), tw, th, kv, bv, wv);
    } else if (nh) {
        dst.rgb = resample_horizontal(src.rgb.data(), src.w, src.h, tw, kh, bh, wh);
    } else {
        dst.rgb = resample_vertical(src.rgb.data(), src.w, th, kv, bv, wv);
    }
}

}  // namespace detail
}  // namespace strata::hy4f::vision
