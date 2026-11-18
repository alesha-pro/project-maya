// src/hy4f/hy4f_vision_test.cpp - encode an image with the maya-hy4f vision encoder, dump the embeddings / positions and
// compare them with a llama.cpp reference dump (tools/hy4f_vision_ref).
//
// usage: hy4f_vision_test --mmproj <mmproj.gguf> --image <file> [--gpu N] [--out DIR] [--ref DIR] [--iters N] [--mrope]
//                         [--arena-mib X | --min | --fast] [--dev-out] [--share-cublas] [--plan WxH,WxH,...]
//   arena: --fast (default) = all heads x 64 queries per attention pass, --min = the smallest that works,
//   --arena-mib = a fixed budget.  --dev-out: embeddings into a device buffer (copied back for the check).
//   --plan: print n_tokens and the min / fast arena for sizes and exit.
//   out: embd.f32 [n_tokens][n_embd], pos.i32 [n_tokens][4] = (t, w, h, z) like mtmd's (t, x, y, z), meta.txt
#include "hy4f_vision.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::hy4f::vision;

static bool read_file(const std::string& p, std::vector<char>& b) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    b.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    std::string mmproj, image, out_dir, ref_dir;
    int gpu = 0, iters = 3;
    bool mrope = false, use_min = false, dev_out = false, share_cublas = false;
    double arena_mib = 0;
    std::string plan;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--image") image = next();
        else if (a == "--gpu") gpu = std::atoi(next().c_str());
        else if (a == "--out") out_dir = next();
        else if (a == "--ref") ref_dir = next();
        else if (a == "--iters") iters = std::max(1, std::atoi(next().c_str()));
        else if (a == "--mrope") mrope = true;
        else if (a == "--min") use_min = true;
        else if (a == "--fast") use_min = false, arena_mib = 0;
        else if (a == "--arena-mib") arena_mib = std::atof(next().c_str());
        else if (a == "--dev-out") dev_out = true;
        else if (a == "--share-cublas") share_cublas = true;
        else if (a == "--plan") plan = next();
        else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (mmproj.empty() || (image.empty() && plan.empty())) {
        std::fprintf(stderr, "usage: %s --mmproj F --image F [--gpu N] [--out DIR] [--ref DIR] [--iters N] [--mrope]\n", argv[0]);
        return 2;
    }
    std::string err;
    size_t free0 = 0, total = 0, free1 = 0;
    cudaSetDevice(gpu);
    cudaFree(nullptr);
    cudaMemGetInfo(&free0, &total);
    auto t0 = std::chrono::steady_clock::now();
    Encoder enc;
    Options o;
    o.verbose = true;
    if (!enc.load(mmproj, o, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const double ms_load = ms_since(t0);
    cudaMemGetInfo(&free1, &total);
    std::printf("encoder load %.0f ms (pinned host), device memory after load %+.1f MiB\n", ms_load,
                ((double) free0 - (double) free1) / 1048576.0);
    if (!plan.empty()) {
        std::stringstream ss(plan);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int w = 0, h = 0;
            if (std::sscanf(item.c_str(), "%dx%d", &w, &h) != 2) continue;
            int ow = 0, oh = 0, tw = 0, th = 0;
            const int nt = image_tokens(w, h, &ow, &oh, &tw, &th);
            std::printf("plan %dx%d (%.2f MP) -> %dx%d, %d patches, %d tokens (Encoder::n_tokens %d); arena min %.1f MiB, "
                        "fast %.1f MiB\n",
                        w, h, w * h / 1e6, tw, th, (tw / 16) * (th / 16), nt, enc.n_tokens(w, h),
                        enc.arena_bytes(w, h, false) / 1048576.0, enc.arena_bytes(w, h, true) / 1048576.0);
        }
        if (image.empty()) return 0;
    }
    t0 = std::chrono::steady_clock::now();
    Image img;
    if (!load_image(image, img, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const double ms_decode = ms_since(t0);
    const size_t abytes = arena_mib > 0 ? (size_t) (arena_mib * 1048576.0) : enc.arena_bytes(img.w, img.h, !use_min);
    std::printf("image %dx%d, decode %.1f ms; arena %.1f MiB (min %.1f, fast %.1f)\n", img.w, img.h, ms_decode,
                abytes / 1048576.0, enc.arena_bytes(img.w, img.h, false) / 1048576.0,
                enc.arena_bytes(img.w, img.h, true) / 1048576.0);
    void* arena_ptr = nullptr;
    if (cudaMalloc(&arena_ptr, abytes) != cudaSuccess) {
        std::fprintf(stderr, "cannot allocate the arena\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    Arena arena;
    arena.dev_ptr = arena_ptr;
    arena.bytes = abytes;
    arena.gpu = gpu;
    arena.stream = stream;
    cublasHandle_t caller_cb = nullptr;
    if (share_cublas) {  // stands in for the LM's handle; created before the baseline so it is not counted
        cublasCreate(&caller_cb);
        arena.cublas = caller_cb;
        cudaMemGetInfo(&free0, &total);
        free0 += abytes;  // the baseline before the arena (dst is allocated below)
    }
    const int nt_plan = enc.n_tokens(img.w, img.h);
    float* dst = nullptr;
    if (dev_out && cudaMalloc(&dst, (size_t) nt_plan * enc.n_embd() * 4) != cudaSuccess) {
        std::fprintf(stderr, "cannot allocate the output\n");
        return 1;
    }

    Output out;
    for (int it = 0; it < iters; ++it) {
        t0 = std::chrono::steady_clock::now();
        if (!enc.encode(img, out, err, arena, dst, mrope ? PosMode::Mrope2D : PosMode::Sequential)) {
            std::fprintf(stderr, "encode: %s\n", err.c_str());
            return 1;
        }
        std::printf("iter %d: resized %dx%d grid %dx%d (+newline) -> %d tokens (planned %d); attention tile %d heads x "
                    "%d queries; preprocess %.1f ms, gpu %.1f ms, total %.1f ms\n",
                    it, out.img_w, out.img_h, out.ow, out.oh, out.n_tokens, nt_plan, out.attn_heads, out.attn_queries,
                    out.ms_preprocess, out.ms_gpu, ms_since(t0));
    }
    size_t free2 = 0;
    cudaMemGetInfo(&free2, &total);
    std::printf("n_pos (text advance) %d; device memory besides the arena%s: %.1f MiB\n", out.n_pos,
                dev_out ? " and the output" : "",
                ((double) free0 - (double) free2 - (double) abytes - (dev_out ? (double) nt_plan * enc.n_embd() * 4 : 0)) /
                    1048576.0);
    if (dev_out) {
        out.embd.resize((size_t) out.n_tokens * out.n_embd);
        cudaMemcpy(out.embd.data(), dst, out.embd.size() * 4, cudaMemcpyDeviceToHost);
    }

    std::vector<int32_t> pos4((size_t) out.n_tokens * 4);
    for (int i = 0; i < out.n_tokens; ++i) {
        pos4[i * 4 + 0] = out.pos[i].t;
        pos4[i * 4 + 1] = out.pos[i].w;
        pos4[i * 4 + 2] = out.pos[i].h;
        pos4[i * 4 + 3] = mrope ? 0 : out.pos[i].t;  // mtmd NORMAL: t = x = y = z
    }
    if (!out_dir.empty()) {
        std::FILE* f = std::fopen((out_dir + "/embd.f32").c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write to %s\n", out_dir.c_str());
            return 1;
        }
        std::fwrite(out.embd.data(), 4, out.embd.size(), f);
        std::fclose(f);
        f = std::fopen((out_dir + "/pos.i32").c_str(), "wb");
        std::fwrite(pos4.data(), 4, pos4.size(), f);
        std::fclose(f);
        f = std::fopen((out_dir + "/meta.txt").c_str(), "w");
        std::fprintf(f, "n_tokens %d n_embd %d ow %d oh %d img %dx%d n_pos %d\n", out.n_tokens, out.n_embd, out.ow, out.oh,
                     out.img_w, out.img_h, out.n_pos);
        std::fclose(f);
    }

    if (!ref_dir.empty()) {
        std::vector<char> re, rp;
        if (!read_file(ref_dir + "/embd.f32", re) || !read_file(ref_dir + "/pos.i32", rp)) {
            std::fprintf(stderr, "cannot read the reference in %s\n", ref_dir.c_str());
            return 1;
        }
        const size_t n = out.embd.size();
        if (re.size() != n * 4) {
            std::printf("MISMATCH: reference has %zu floats, we have %zu\n", re.size() / 4, n);
            return 1;
        }
        const float* r = (const float*) re.data();
        const float* m = out.embd.data();
        const int E = out.n_embd;
        std::vector<double> cs(out.n_tokens);
        double max_abs = 0, sum_abs = 0, ref_abs = 0, ref_max = 0;
        int nl_rows = 0, nl_exact = 0;
        for (int t = 0; t < out.n_tokens; ++t) {
            double dot = 0, na = 0, nb = 0;
            bool exact = true;
            for (int c = 0; c < E; ++c) {
                const double a = m[(size_t) t * E + c], b = r[(size_t) t * E + c];
                dot += a * b;
                na += a * a;
                nb += b * b;
                const double d = std::fabs(a - b);
                max_abs = std::max(max_abs, d);
                sum_abs += d;
                ref_abs += std::fabs(b);
                ref_max = std::max(ref_max, std::fabs(b));
                exact = exact && a == b;
            }
            cs[t] = dot / (std::sqrt(na * nb) + 1e-30);
            if (t % (out.ow + 1) == out.ow) {
                ++nl_rows;
                nl_exact += exact;
            }
        }
        std::vector<double> s = cs;
        std::sort(s.begin(), s.end());
        auto q = [&](double f) { return s[std::min(s.size() - 1, (size_t) (f * (s.size() - 1)))]; };
        double mean = 0;
        for (double v : s) mean += v;
        mean /= s.size();
        std::printf("parity vs %s: cos per row median %.6f mean %.6f p01 %.6f min %.6f (row %d); max_abs %.4g mean_abs %.4g "
                    "(ref mean |x| %.4g, max |x| %.4g); newline rows exact %d/%d\n",
                    ref_dir.c_str(), q(0.5), mean, q(0.01), s[0],
                    (int) (std::min_element(cs.begin(), cs.end()) - cs.begin()), max_abs, sum_abs / n, ref_abs / n, ref_max,
                    nl_exact, nl_rows);
        const int32_t* p = (const int32_t*) rp.data();
        int pos_bad = 0;
        if (rp.size() != pos4.size() * 4) pos_bad = -1;
        else
            for (size_t i = 0; i < pos4.size(); ++i) pos_bad += p[i] != pos4[i];
        std::printf("positions vs reference: %s\n", pos_bad == 0 ? "identical" : (pos_bad < 0 ? "size mismatch" : "DIFFER"));
    }
    return 0;
}
