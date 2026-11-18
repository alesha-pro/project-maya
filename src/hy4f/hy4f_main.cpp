// src/hy4f/hy4f_main.cpp - maya-hy4f v0 driver: feed token ids one by one, optionally dump the seams, generate greedily.
//
//   maya-hy4f --gguf <shard-00001.gguf> [--gguf <shard-00002.gguf>] (--tokens-i32 FILE | --ids 1,2,3)
//             [--gen N] [--dump DIR] [--max-context N] [--gpus 0,1,2,3] [--split 11,22,33] [-v]
//   maya-hy4f --serve --gguf ... [--max-context N] [--eos 120025] [--chunk 2048] [--spec-k auto|0..3] [--mmproj F]
//             [--model-name NAME]
//
// --serve: Maya's engine protocol on stdin / stdout for serve/server.py (hy4f_serve.cpp has the protocol).
//
// --tokens-i32: a raw int32 file (tools/hy4f_ref_dump's tokens.i32).  --dump: the seams appended per token under the
// llama.cpp cb names (attn_norm-N, attn_out-N, ffn_out-N, ffn_moe_out-N, ffn_shexp-N, kv_cmpr-N, engram_delta-N,
// result_norm, logits).  The decode rate is printed over the steps after the first.
#include "hy4f.hpp"

#include <cuda_profiler_api.h>
#include "hy4f_serve.hpp"

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
    std::vector<int> depths;
    bool vcmp = false;
    std::vector<int> replay;   // --replay a,b,c,..: these tokens after the prompt via verify(T) chunks (T = --replay-t)
    int replay_t = 1;
    bool serve_mode = false;
    strata::hy4f::ServeOptions so;
    std::string mmproj, image;
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
        else if (a == "--verify-cmp") vcmp = true;
        else if (a == "--replay") replay = csv(next());
        else if (a == "--replay-t") replay_t = std::atoi(next());   // verify T=3 rows vs three T=1 steps after the prompt
        else if (a == "--depth-bench") depths = csv(next());   // prefill to each depth, then time decode there
        else if (a == "--verify-bench") vbench = std::atoi(next());   // time verify(T) for T = 1..4 (N reps)
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--image") image = next();   // replaces the prompt's image pad token 120120
        else if (a == "--spec") spec = std::atoi(next());   // MTP drafts per step (1)
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
        else if (a == "--spec-k") {   // --serve: MTP drafts per step: auto (adaptive 1..3), 0 (none), 1..3
            const std::string v = next();
            so.spec = v == "auto" ? -1 : std::atoi(v.c_str());
        }
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
                             "[--chunk N] [--spec-k auto|0..3] [--mmproj F] [--model-name NAME]\n");
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
        if (!mmproj.empty()) {   // (pictures: GENI; the encoder's weights in pinned host RAM)
            if (!m.load_vision(mmproj, err)) {
                std::fprintf(stderr, "maya-hy4f: vision: %s\n", err.c_str());
                std::printf("NOTE maya-hy4f: the vision encoder did not load (%s): text only\n", err.c_str());
            } else {
                std::printf("NOTE maya-hy4f: vision encoder %s loaded\n", mmproj.c_str());
            }
        }
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
    if (!depths.empty()) {   // depth sweep: prefill segment rate, verify(T) / MTP timings and a k=2 spec decode at each depth
        using clk = std::chrono::steady_clock;
        auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
        std::vector<float> lg((size_t) m.vocab());
        int64_t cur = 0;
        const int ch = pchunk > 0 ? pchunk : 2048, K = 2, NG = 96;
        for (int d : depths) {
            if (d + 8 > (int) toks.size()) break;
            const bool prof = std::getenv("HY4F_PROF_SEG") && std::atoi(std::getenv("HY4F_PROF_SEG")) == d;
            if (prof) cudaProfilerStart();
            auto a = clk::now();
            if (!m.prefill(toks.data() + cur, d - (int) cur, cur, lg.data(), ch, err)) {
                std::fprintf(stderr, "maya-hy4f: prefill failed: %s\n", err.c_str());
                return 1;
            }
            const double pf = (d - cur) / secs(a);
            if (prof) cudaProfilerStop();
            int32_t vt[4] = {toks[d], toks[d + 1], toks[d + 2], toks[d + 3]}, am[4], dr[4];
            double vms[5] = {0, 0, 0, 0, 0};
            const bool profd = std::getenv("HY4F_PROF_DEC") && std::atoi(std::getenv("HY4F_PROF_DEC")) == d;
            if (profd) cudaProfilerStart();
            for (int T = 1; T <= 4; ++T) {
                m.verify(vt, T, d, nullptr, am, err);
                a = clk::now();
                for (int r = 0; r < 5; ++r) m.verify(vt, T, d, nullptr, am, err);
                vms[T] = secs(a) / 5 * 1e3;
            }
            a = clk::now();
            for (int r = 0; r < 5; ++r) m.mtp(vt, 1, d - 1, dr, err);
            const double mms = secs(a) / 5 * 1e3;
            if (profd) cudaProfilerStop();
            // greedy spec decode, k = 2, from the prompt's last logits at position d
            int64_t pos = d;
            int32_t last = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            int ngen = 1, steps = 0;
            m.mtp(&last, 1, pos - 1, &dr[0], err);
            for (int i = 1; i < K; ++i) m.mtp_chain(dr[i - 1], pos + i - 1, &dr[i], err);
            a = clk::now();
            while (ngen < NG) {
                int32_t v2[4] = {last, dr[0], dr[1], dr[2]};
                if (!m.verify(v2, K + 1, pos, nullptr, am, err)) break;
                ++steps;
                int n = 0;
                while (n < K && am[n] == dr[n]) ++n;
                int32_t nt[4];
                for (int i = 0; i < n; ++i) nt[i] = dr[i];
                nt[n] = am[n];
                m.mtp(nt, n + 1, pos, &dr[0], err);
                last = nt[n];
                pos += n + 1;
                ngen += n + 1;
                for (int i = 1; i < K; ++i) m.mtp_chain(dr[i - 1], pos + i - 1, &dr[i], err);
            }
            const double st = secs(a);
            std::printf("DEPTH %d prefill_seg %lld..%d %.1f tok/s | verify T1 %.2f T2 %.2f T3 %.2f T4 %.2f ms | mtp %.2f ms | "
                        "spec k=2 %.1f tok/s (%.2f tok/step, %.2f ms/step)\n",
                        d, (long long) cur, d, pf, vms[1], vms[2], vms[3], vms[4], mms, (ngen - 1) / st,
                        (double) (ngen - 1) / std::max(1, steps), 1e3 * st / std::max(1, steps));
            std::fflush(stdout);
            cur = d;
        }
        return 0;
    }
    std::vector<float> img_rows;
    std::vector<strata::hy4f::EmbdSpan> spans;
    if (!image.empty()) {
        if (mmproj.empty() || !m.load_vision(mmproj, err)) {
            std::fprintf(stderr, "maya-hy4f: vision: %s\n", mmproj.empty() ? "--image needs --mmproj" : err.c_str());
            return 1;
        }
        if (pchunk <= 0) pchunk = 2048;
        int nimg = 0;
        const auto a = std::chrono::steady_clock::now();
        if (!m.encode_image(image, img_rows, nimg, err)) {
            std::fprintf(stderr, "maya-hy4f: encode_image: %s\n", err.c_str());
            return 1;
        }
        std::printf("image: %d tokens, encoded in %.0f ms\n", nimg,
                    1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
        const auto it = std::find(toks.begin(), toks.end(), 120120);
        if (it == toks.end()) {
            std::fprintf(stderr, "maya-hy4f: the prompt has no image pad token 120120\n");
            return 1;
        }
        const int at = (int) (it - toks.begin());
        toks.erase(it);
        toks.insert(toks.begin() + at, (size_t) nimg, 120120);
        spans.push_back({at, nimg, img_rows.data()});
    }
    if (repeat > 1) {
        const std::vector<int32_t> base = toks;
        for (int r = 1; r < repeat; ++r) toks.insert(toks.end(), base.begin(), base.end());
    }
    if (pchunk > 0) {
        const int np = (int) toks.size() - std::min(ptail, (int) toks.size() - 1);
        const auto a = std::chrono::steady_clock::now();
        if (!m.prefill(toks.data(), np, 0, logits.data(), pchunk, err, spans.empty() ? nullptr : &spans)) {
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
    if (!replay.empty()) {   // feed the tokens in verify(T) chunks; print the top-3 of the row after the last one
        const int V = m.vocab();
        std::vector<float> l((size_t) 4 * V);
        int32_t am[4];
        size_t i = 0;
        while (i < replay.size()) {
            const int T = std::min<int>(replay_t, (int) (replay.size() - i));
            std::vector<int32_t> tk(replay.begin() + i, replay.begin() + i + T);
            m.verify(tk.data(), T, pos, l.data(), am, err);
            pos += T;
            i += T;
            if (i == replay.size()) {
                const float* r = l.data() + (size_t) (T - 1) * V;
                std::vector<int> idx(V);
                for (int v = 0; v < V; ++v) idx[v] = v;
                std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&](int a2, int b2) { return r[a2] > r[b2]; });
                std::printf("replay T=%d: top %d %.4f | %d %.4f | %d %.4f\n", replay_t, idx[0], r[idx[0]], idx[1], r[idx[1]], idx[2], r[idx[2]]);
            }
        }
        return 0;
    }
    if (vcmp) {
        const int V = m.vocab();
        std::vector<float> l1((size_t) 3 * V), l3((size_t) 3 * V);
        int32_t am[4], tk[3] = {(int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin()), 0, 0};
        for (int i = 0; i < 3; ++i) {
            m.verify(&tk[i], 1, pos + i, l1.data() + (size_t) i * V, am, err);
            if (i < 2) tk[i + 1] = am[0];
        }
        m.verify(tk, 3, pos, l3.data(), am, err);
        for (int i = 0; i < 3; ++i) {
            double dot = 0, na = 0, nb2 = 0;
            for (int v = 0; v < V; ++v) {
                const double x = l1[(size_t) i * V + v], y = l3[(size_t) i * V + v];
                dot += x * y, na += x * x, nb2 += y * y;
            }
            const int a1 = (int) (std::max_element(l1.begin() + (size_t) i * V, l1.begin() + (size_t) (i + 1) * V) - l1.begin() - (size_t) i * V);
            const int a3 = (int) (std::max_element(l3.begin() + (size_t) i * V, l3.begin() + (size_t) (i + 1) * V) - l3.begin() - (size_t) i * V);
            std::printf("verify-cmp row %d: cos %.6f argmax T1 %d T3 %d\n", i, dot / std::sqrt(na * nb2), a1, a3);
        }
        return 0;
    }
    if (spec > 0 && gen > 0) {
        if (!m.has_mtp()) {
            std::fprintf(stderr, "maya-hy4f: --spec needs the MTP layer\n");
            return 1;
        }
        // greedy speculative decode with k = spec MTP drafts per step: verify [last, d0..dk-1] -> accept the matching
        // prefix + the bonus token -> MTP over the accepted rows (draft d0) -> chain d1.. from the MTP's own state
        const int k = std::min(spec, 3);
        std::vector<int32_t> out;
        int32_t last = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin()), dr[4], am[4];
        out.push_back(last);
        auto draft_chain = [&](int64_t mpos_next) {
            for (int i = 1; i < k; ++i)
                if (!m.mtp_chain(dr[i - 1], mpos_next + i - 1, &dr[i], err)) {
                    std::fprintf(stderr, "maya-hy4f: mtp_chain failed: %s\n", err.c_str());
                    std::exit(1);
                }
        };
        if (!m.mtp(&last, 1, pos - 1, &dr[0], err)) {
            std::fprintf(stderr, "maya-hy4f: mtp failed: %s\n", err.c_str());
            return 1;
        }
        draft_chain(pos);
        int steps2 = 0;
        std::vector<int> acc_hist(k + 1, 0);
        const auto a = std::chrono::steady_clock::now();
        while ((int) out.size() < gen) {
            int32_t vt[4] = {last, dr[0], dr[1], dr[2]};
            if (!m.verify(vt, k + 1, pos, nullptr, am, err)) {
                std::fprintf(stderr, "maya-hy4f: verify failed: %s\n", err.c_str());
                return 1;
            }
            ++steps2;
            int n = 0;
            while (n < k && am[n] == dr[n]) ++n;
            ++acc_hist[n];
            int32_t nt[4];
            for (int i = 0; i < n; ++i) nt[i] = dr[i];
            nt[n] = am[n];
            for (int i = 0; i <= n; ++i) out.push_back(nt[i]);
            if (!m.mtp(nt, n + 1, pos, &dr[0], err)) {
                std::fprintf(stderr, "maya-hy4f: mtp failed: %s\n", err.c_str());
                return 1;
            }
            last = nt[n];
            pos += n + 1;
            draft_chain(pos);
        }
        int acc = 0;
        for (int i = 0; i <= k; ++i) acc += i * acc_hist[i];
        std::printf("accepted per step:");
        for (int i = 0; i <= k; ++i) std::printf(" %d:%d", i, acc_hist[i]);
        std::printf("\n");
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        std::printf("gen:");
        for (int32_t t : out) std::printf(" %d", t);
        std::printf("\nspec k=%d: %zu tokens in %d steps, %.2f tokens/step, draft accept %.1f%%, %.2f tok/s, %.2f ms/step\n", k,
                    out.size() - 1, steps2, (double) (out.size() - 1) / std::max(1, steps2),
                    100.0 * acc / std::max(1, steps2 * k), (out.size() - 1) / dt, 1000.0 * dt / std::max(1, steps2));
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
