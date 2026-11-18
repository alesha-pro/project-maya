// src/hy4f/hy4f_main.cpp - maya-hy4f v0 driver: feed token ids one by one, optionally dump the seams, generate greedily.
//
//   maya-hy4f --gguf <shard-00001.gguf> [--gguf <shard-00002.gguf>] (--tokens-i32 FILE | --ids 1,2,3)
//             [--gen N] [--dump DIR] [--max-context N] [--gpus 0,1,2,3] [--split 11,22,33] [-v]
//   maya-hy4f --serve --gguf ... [--max-context N] [--eos 120025] [--chunk 64] [--model-name NAME]
//
// --serve: Maya's engine protocol on stdin / stdout for serve/server.py (hy4f_serve.cpp has the protocol).
//
// --tokens-i32: a raw int32 file (tools/hy4f_ref_dump's tokens.i32).  --dump: the seams appended per token under the
// llama.cpp cb names (attn_norm-N, attn_out-N, ffn_out-N, ffn_moe_out-N, ffn_shexp-N, kv_cmpr-N, engram_delta-N,
// result_norm, logits).  The decode rate is printed over the steps after the first.
#include "hy4f.hpp"
#include "hy4f_serve.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> csv(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(std::atoi(x.c_str()));
    return v;
}

int main(int argc, char** argv) {
    strata::hy4f::Options o;
    std::vector<int32_t> toks;
    int gen = 0;
    bool serve_mode = false;
    strata::hy4f::ServeOptions so;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-hy4f: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gguf") o.shards.push_back(next());
        else if (a == "--tokens-i32") {
            std::ifstream f(next(), std::ios::binary);
            int32_t t;
            while (f.read((char*) &t, 4)) toks.push_back(t);
        } else if (a == "--ids") for (int t : csv(next())) toks.push_back(t);
        else if (a == "--gen") gen = std::atoi(next());
        else if (a == "--dump") o.dump_dir = next();
        else if (a == "--max-context") o.max_context = std::atoll(next());
        else if (a == "--gpus") o.gpus = csv(next());
        else if (a == "--split") o.split = csv(next());
        else if (a == "--layer-split") {   // (serve/server.py adds it with several GPUs: "auto", or as --split)
            const std::string v = next();
            if (v != "auto") o.split = csv(v.c_str());
        }
        else if (a == "-v") o.verbose = true;
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos") {
            so.eos.clear();
            for (int t : csv(next())) so.eos.push_back(t);
        } else if (a == "--chunk") so.chunk = std::atoi(next());
        else if (a == "--model-name") so.model_name = next();
        else {
            std::fprintf(stderr, "maya-hy4f: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (o.shards.empty() || (toks.empty() && !serve_mode)) {
        std::fprintf(stderr, "usage: maya-hy4f --gguf SHARD1 [--gguf SHARD2] (--tokens-i32 F | --ids a,b) [--gen N] "
                             "[--dump DIR] [--max-context N] [--gpus ..] [--split ..] [-v]\n"
                             "       maya-hy4f --serve --gguf SHARD1 [--gguf SHARD2] [--max-context N] [--eos ID,..] "
                             "[--chunk N] [--model-name NAME]\n");
        return 2;
    }
    strata::hy4f::Model m;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-hy4f: load failed: %s\n", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (serve_mode) {   // (stdout is the protocol's: the load's note goes as a NOTE line, and to the log)
        std::fprintf(stderr, "maya-hy4f: %s (loaded in %.1f s)\n", m.summary().c_str(), load_s);
        std::printf("NOTE maya-hy4f: %s (loaded in %.1f s)\n", m.summary().c_str(), load_s);
        return strata::hy4f::serve(m, so);
    }
    std::printf("%s (loaded in %.1f s)\n", m.summary().c_str(), load_s);
    std::vector<float> logits((size_t) m.vocab());
    int64_t pos = 0;
    auto tick = std::chrono::steady_clock::now();
    double steps_s = 0;
    int steps = 0;
    auto run = [&](int32_t t) {
        const auto a = std::chrono::steady_clock::now();
        if (!m.step(t, pos++, logits.data(), err)) {
            std::fprintf(stderr, "maya-hy4f: step %lld failed: %s\n", (long long) pos - 1, err.c_str());
            std::exit(1);
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        if (pos > 1) steps_s += dt, ++steps;
    };
    for (int32_t t : toks) run(t);
    std::printf("prompt: %zu tokens\n", toks.size());
    std::vector<int32_t> out;
    for (int g = 0; g < gen; ++g) {
        const int32_t t = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        out.push_back(t);
        run(t);
    }
    if (!out.empty()) {
        std::printf("gen:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\n");
    }
    (void) tick;
    if (steps) std::printf("decode: %d steps, %.2f ms/step, %.2f tok/s (v0, one token per step)\n", steps,
                           1000.0 * steps_s / steps, steps / steps_s);
    return 0;
}
