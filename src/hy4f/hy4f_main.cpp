// src/hy4f/hy4f_main.cpp - maya-hy4f v0 driver: feed token ids one by one, optionally dump the seams, generate greedily.
//
//   maya-hy4f --gguf <shard-00001.gguf> [--gguf <shard-00002.gguf>] (--tokens-i32 FILE | --ids 1,2,3)
//             [--gen N] [--dump DIR] [--max-context N] [--gpus 0,1,2,3] [--split 11,22,33] [-v]
//
// --tokens-i32: a raw int32 file (tools/hy4f_ref_dump's tokens.i32).  --dump: the seams appended per token under the
// llama.cpp cb names (attn_norm-N, attn_out-N, ffn_out-N, ffn_moe_out-N, ffn_shexp-N, kv_cmpr-N, engram_delta-N,
// result_norm, logits).  The decode rate is printed over the steps after the first.
#include "hy4f.hpp"

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
    int gen = 0, pchunk = 0, ptail = 0, repeat = 1, spec = 0, vtail = 0, vbench = 0;
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
        else if (a == "--prefill") pchunk = std::atoi(next());
        else if (a == "--prefill-tail") ptail = std::atoi(next());
        else if (a == "--repeat") repeat = std::atoi(next());
        else if (a == "--verify-tail") vtail = std::atoi(next());   // the prompt tail through verify(T) chunks
        else if (a == "--verify-bench") vbench = std::atoi(next());   // time verify(T) for T = 1..4 (N reps)
        else if (a == "--spec") spec = std::atoi(next());   // MTP drafts per step (1)
        else if (a == "--gen") gen = std::atoi(next());
        else if (a == "--dump") o.dump_dir = next();
        else if (a == "--max-context") o.max_context = std::atoll(next());
        else if (a == "--gpus") o.gpus = csv(next());
        else if (a == "--split") o.split = csv(next());
        else if (a == "-v") o.verbose = true;
        else {
            std::fprintf(stderr, "maya-hy4f: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (o.shards.empty() || toks.empty()) {
        std::fprintf(stderr, "usage: maya-hy4f --gguf SHARD1 [--gguf SHARD2] (--tokens-i32 F | --ids a,b) [--gen N] "
                             "[--dump DIR] [--max-context N] [--gpus ..] [--split ..] [-v]\n");
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
    if (repeat > 1) {
        const std::vector<int32_t> base = toks;
        for (int r = 1; r < repeat; ++r) toks.insert(toks.end(), base.begin(), base.end());
    }
    if (pchunk > 0) {
        const int np = (int) toks.size() - std::min(ptail, (int) toks.size() - 1);
        const auto a = std::chrono::steady_clock::now();
        if (!m.prefill(toks.data(), np, 0, logits.data(), pchunk, err)) {
            std::fprintf(stderr, "maya-hy4f: prefill failed: %s\n", err.c_str());
            return 1;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        std::printf("prefill: %d tokens in %.3f s, %.1f tok/s (chunk %d, first call includes setup)\n", np, dt, np / dt, pchunk);
        pos = np;
        if (vtail > 0) {
            std::vector<float> lg((size_t) 4 * m.vocab());
            int32_t am[4];
            for (size_t i = np; i < toks.size();) {
                const int T = std::min<int>(vtail, (int) (toks.size() - i));
                if (!m.verify(toks.data() + i, T, pos, lg.data(), am, err)) {
                    std::fprintf(stderr, "maya-hy4f: verify failed: %s\n", err.c_str());
                    return 1;
                }
                if (!o.dump_dir.empty()) {
                    std::FILE* f = std::fopen((o.dump_dir + "/logits.f32").c_str(), "ab");
                    if (f) std::fwrite(lg.data(), 4, (size_t) T * m.vocab(), f), std::fclose(f);
                }
                std::copy(lg.begin() + (size_t) (T - 1) * m.vocab(), lg.begin() + (size_t) T * m.vocab(), logits.begin());
                pos += T;
                i += T;
            }
        } else {
            for (size_t i = np; i < toks.size(); ++i) run(toks[i]);
        }
        steps_s = 0, steps = 0;
    } else {
        for (int32_t t : toks) run(t);
    }
    std::printf("prompt: %zu tokens\n", toks.size());
    if (vbench > 0) {
        int32_t am[4], vt[4] = {toks[0], toks[1 % toks.size()], toks[2 % toks.size()], toks[3 % toks.size()]};
        for (int T = 1; T <= 4; ++T) {
            m.verify(vt, T, pos, nullptr, am, err);
            const auto a = std::chrono::steady_clock::now();
            for (int r = 0; r < vbench; ++r)
                if (!m.verify(vt, T, pos, nullptr, am, err)) {
                    std::fprintf(stderr, "maya-hy4f: verify failed: %s\n", err.c_str());
                    return 1;
                }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / vbench;
            std::printf("verify T=%d: %.2f ms\n", T, dt * 1e3);
        }
        if (m.has_mtp())
            for (int T = 1; T <= 2; ++T) {
                int32_t dr;
                const auto a = std::chrono::steady_clock::now();
                for (int r = 0; r < vbench; ++r) m.mtp(vt, T, pos, &dr, err);
                std::printf("mtp T=%d: %.2f ms\n", T,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / vbench * 1e3);
            }
    }
    if (spec > 0 && gen > 0) {
        if (!m.has_mtp()) {
            std::fprintf(stderr, "maya-hy4f: --spec needs the MTP layer\n");
            return 1;
        }
        // greedy speculative decode, one MTP draft per step: verify [last, draft] -> accept -> MTP over the new rows
        std::vector<int32_t> out;
        int32_t last = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin()), draft = 0, am[4];
        out.push_back(last);
        if (!m.mtp(&last, 1, pos - 1, &draft, err)) {
            std::fprintf(stderr, "maya-hy4f: mtp failed: %s\n", err.c_str());
            return 1;
        }
        int steps2 = 0, acc = 0;
        const auto a = std::chrono::steady_clock::now();
        while ((int) out.size() < gen) {
            const int32_t vt[2] = {last, draft};
            if (!m.verify(vt, 2, pos, nullptr, am, err)) {
                std::fprintf(stderr, "maya-hy4f: verify failed: %s\n", err.c_str());
                return 1;
            }
            ++steps2;
            int32_t nt[2];
            int nn;
            if (am[0] == draft) {
                ++acc;
                nt[0] = draft, nt[1] = am[1], nn = 2;
            } else {
                nt[0] = am[0], nn = 1;
            }
            for (int i = 0; i < nn; ++i) out.push_back(nt[i]);
            if (!m.mtp(nt, nn, pos, &draft, err)) {
                std::fprintf(stderr, "maya-hy4f: mtp failed: %s\n", err.c_str());
                return 1;
            }
            last = nt[nn - 1];
            pos += nn;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        std::printf("gen:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\nspec: %zu tokens in %d steps, accept %.1f%%, %.2f tok/s, %.2f ms/step\n", out.size() - 1, steps2,
                    100.0 * acc / std::max(1, steps2), (out.size() - 1) / dt, 1000.0 * dt / std::max(1, steps2));
        return 0;
    }
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
