// src/hy4f/hy4f_serve.cpp - maya-hy4f --serve: Maya's engine protocol on stdin / stdout, as serve/server.py's StrataEngine
// speaks it to maya-mm3 / maya-ds4 (src/mm3/mm3_main.cpp has the reference):
//
//   engine -> server   NOTE <text>, INFO key=value ..., then `READY <context> stop`
//   server -> engine   GEN <max_new> [key=value ...] <id,id,...>      one request (one conversation at a time)
//   engine -> server   PP <read> <total> <ms> <tok/s> <reused>         while the prompt is read (every --chunk tokens)
//                      T <id>                                          each answer token
//                      STAT tok_s=F ms_tok=F                           every 8 tokens, and at the end
//                      DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> 0 0
//   server -> engine   STOP (ends the request: DONE ... cancel), WRAP (the server is stopping: a reasoning block still
//                      open is closed with think_end and the answer follows), QUIT
//
// Keys: temperature (<= 0 or absent: greedy, the argmax, ties to the lower id), top_p, top_k, min_p, seed,
// penalty_last_n / penalty_repeat / penalty_freq / penalty_present (llama.cpp's, over the last n tokens of the
// conversation), logit_bias=id:bias,..., think_budget + think_end (an open reasoning block is closed with think_end after
// that many answer tokens); other keys are taken and ignored.  Sampling runs on the host over the whole row of logits.
// The answer ends at an eos id (--eos; default 120025 = <｜hy_end:opensource｜>), at max_new, or at the context's end.
//
// The conversation in the cache: a prompt that starts with the tokens the cache holds reads only the rest (the last
// prompt token is always read again: its logits are the answer's first); one that departs from them earlier keeps the
// shared start where Model::rewind can (see the serve block of hy4f.cu: position-indexed caches, the SWA / indexer rings
// and the copy Model::mark takes after each prompt - the next chat turn, whose template drops the answer's reasoning,
// departs right there), else the cache starts again at 0.  A step that fails ends the engine (END, DONE ... cancel,
// exit 1): the server starts a new one.
#include "hy4f_serve.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

namespace strata::hy4f {
namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    long long think_budget = 0;
    int think_end = -1;
    std::vector<std::pair<int, float>> bias;
    std::vector<int32_t> ids;
};

bool parse_gen(const std::string& line, Req& r) {
    std::istringstream is(line.substr(4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {   // the ids: comma-separated, last on the line
            for (char& c : tok)
                if (c == ',') c = ' ';
            std::istringstream ids(tok);
            for (long x; ids >> x;) r.ids.push_back((int32_t) x);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        else if (k == "logit_bias") {   // id:bias,id:bias
            std::istringstream bs(v);
            for (std::string p; std::getline(bs, p, ',');) {
                const size_t c = p.find(':');
                if (c != std::string::npos) r.bias.push_back({std::atoi(p.c_str()), std::strtof(p.c_str() + c + 1, nullptr)});
            }
        }
    }
    return !r.ids.empty();
}

// stdin's lines on a thread of their own: STOP and WRAP are seen while a request runs
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false}, wrap{false};
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[65536];
            for (;;) {
                const ssize_t n = ::read(0, chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                size_t at = 0;
                for (size_t nl; (nl = buf.find('\n', at)) != std::string::npos; at = nl + 1) {
                    std::string l = buf.substr(at, nl - at);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
                buf.erase(0, at);
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// The next token from a row of logits: logit_bias, llama.cpp's penalties over the history's last pen_n tokens (-1: all),
// then greedy (temperature <= 0) or temperature, top_k (0: every token), min_p and top_p, drawn with rng.
struct Sampler {
    std::vector<float> row;
    std::vector<std::pair<float, int>> c;
    int next(const float* lg, int V, const Req& r, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
        const bool pen = r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f);
        if (pen || !r.bias.empty()) {
            row.assign(lg, lg + V);
            for (const auto& [t, b] : r.bias)
                if (t >= 0 && t < V) row[(size_t) t] += b;
            if (pen) {
                std::unordered_map<int, int> cnt;
                const size_t n = r.pen_n < 0 ? hist.size() : std::min(hist.size(), (size_t) r.pen_n);
                for (size_t i = hist.size() - n; i < hist.size(); ++i) ++cnt[hist[i]];
                for (const auto& [t, k] : cnt) {
                    if (t < 0 || t >= V) continue;
                    float& x = row[(size_t) t];
                    if (r.pen_repeat > 0.0f && r.pen_repeat != 1.0f) x = x > 0 ? x / r.pen_repeat : x * r.pen_repeat;
                    x -= r.pen_freq * (float) k + r.pen_present;
                }
            }
            lg = row.data();
        }
        if (!(r.temperature > 0.0f)) {
            int b = 0;
            for (int i = 1; i < V; ++i)
                if (lg[i] > lg[b]) b = i;
            return b;
        }
        c.resize((size_t) V);
        for (int i = 0; i < V; ++i) c[(size_t) i] = {lg[i] / r.temperature, i};
        const size_t k = r.top_k > 0 ? std::min((size_t) r.top_k, c.size()) : c.size();
        std::partial_sort(c.begin(), c.begin() + (long) k, c.end(), [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        std::vector<double> p(k);
        double sum = 0;
        for (size_t i = 0; i < k; ++i) sum += (p[i] = std::exp((double) (c[i].first - c[0].first)));
        size_t keep = k;
        double cum = 0;
        for (size_t i = 0; i < k; ++i) {
            cum += p[i] / sum;
            if (r.min_p > 0.0f && p[i] < r.min_p * p[0]) {
                keep = std::max<size_t>(1, i);
                break;
            }
            if (r.top_p < 1.0f && cum >= r.top_p) {
                keep = i + 1;
                break;
            }
        }
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += p[i];
        double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
        for (size_t i = 0; i < keep; ++i) {
            u -= p[i];
            if (u <= 0) return c[i].second;
        }
        return c[keep - 1].second;
    }
};

// The prompt's tokens at positions pos0 .. pos0 + n - 1; the last one's logits to `logits` (when not null).  One step a
// token today: a batched Model::prefill(ids, n, pos0, logits) (the hy4f branch's next step) drops in here unchanged -
// the serve loop already calls this per chunk of ServeOptions::chunk tokens and needs only the last row.
bool prefill(Model& m, const int32_t* ids, int n, int64_t pos0, float* logits, std::string& err) {
    for (int i = 0; i < n; ++i)
        if (!m.step(ids[i], pos0 + i, i + 1 == n ? logits : nullptr, err)) return false;
    return true;
}

}  // namespace

int serve(Model& m, const ServeOptions& so) {
    const int V = m.vocab();
    const long long C = m.max_context();
    std::printf("INFO engine=hy4f engine_kind=hy4f model=%s ctx=%lld kv=q8 logit_bias=1\n", so.model_name.c_str(), C);
    std::printf("READY %lld stop\n", C);
    std::fflush(stdout);
    std::fprintf(stderr, "maya hy4f serve: GEN <max_new> [keys] <ids>; STOP, WRAP, QUIT\n");
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    Lines in;
    in.start();
    std::vector<int32_t> state;   // the tokens in the cache, positions 0 .. (m.position() == state.size())
    std::vector<float> lg((size_t) V);
    Sampler sampler;
    unsigned long long counter = 0;
    bool mark_ok = true;
    std::string line, err;
    auto fatal = [&](const std::string& why, long long produced, size_t n_prompt) {
        std::fprintf(stderr, "maya-hy4f: %s\n", why.c_str());
        std::printf("END hy4f: %s\n", why.c_str());
        std::printf("DONE %lld %zu 0.0 0.0 cancel 0 0 0 0 0\n", produced, n_prompt);
        std::fflush(stdout);
        std::exit(1);
    };
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, STOP, WRAP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...>\n");
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            std::fflush(stdout);
            continue;
        }
        const size_t n = r.ids.size();
        bool bad_id = false;
        for (int32_t t : r.ids) bad_id |= t < 0 || t >= V;
        if (bad_id || (long long) n + std::max(0LL, r.max_new) > C) {
            if (bad_id) std::printf("ERR a token id outside the vocabulary (%d)\n", V);
            else std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", n, r.max_new, C);
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
            std::fflush(stdout);
            continue;
        }
        const double t0 = now_ms();
        // the cache's shared start (the last prompt token is always read again)
        size_t from = 0;
        while (from < state.size() && from + 1 < n && state[from] == r.ids[from]) ++from;
        const int64_t kept = m.rewind((int64_t) from, err);
        if (kept < 0) fatal("rewind: " + err, 0, n);
        if ((size_t) kept != from)
            std::fprintf(stderr, "maya hy4f serve: the prompt departs from the cache at %zu (the rings no longer hold it): "
                                 "read from %lld\n", from, (long long) kept);
        from = (size_t) kept;
        state.resize(from);
        bool cancel = false;
        auto pp_line = [&](size_t done) {
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done, n, ms, ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        const int CH = std::max(1, so.chunk);
        for (size_t a = from; a < n;) {
            const int T = (int) std::min<size_t>((size_t) CH, n - a);
            const bool last = a + (size_t) T == n;
            if (!prefill(m, r.ids.data() + a, T, (int64_t) a, last ? lg.data() : nullptr, err))
                fatal("prompt step at " + std::to_string(m.position()) + ": " + err, 0, n);
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + T);
            a += (size_t) T;
            pp_line(a);
            if (in.stop.load() && !last) {
                cancel = true;
                break;
            }
        }
        const double t1 = now_ms();
        if (cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", n, t1 - t0, from);
            std::fflush(stdout);
            continue;
        }
        if (mark_ok && !m.mark(err)) {   // (the rings at the prompt's end: the next turn may go back here)
            std::fprintf(stderr, "maya hy4f serve: no copy of the rings (%s) - a turn that departs from the answer "
                                 "reads its prompt again\n", err.c_str());
            mark_ok = false;
        }
        ++counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        auto stat_line = [&]() {
            const double ms = now_ms() - t1;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f\n", ms > 0 ? 1000.0 * (double) produced / ms : 0.0,
                        produced > 0 ? ms / (double) produced : 0.0);
        };
        int tok = sampler.next(lg.data(), V, r, state, rng);
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (in_think && (produced >= r.think_budget || in.wrap.load()) && tok != r.think_end)
                tok = r.think_end;   // (the thinking budget, or WRAP)
            std::printf("T %d\n", tok);
            ++produced;
            if (in_think && tok == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            if (std::find(so.eos.begin(), so.eos.end(), tok) != so.eos.end()) {
                finish = "stop";
                break;
            }
            if (produced >= r.max_new || (long long) state.size() >= C) break;
            if (!m.step(tok, (int64_t) state.size(), lg.data(), err))
                fatal("decode step at " + std::to_string(state.size()) + ": " + err, produced, n);
            state.push_back(tok);
            tok = sampler.next(lg.data(), V, r, state, rng);
        }
        stat_line();
        const double t2 = now_ms();
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0\n", produced, n, t1 - t0, t2 - t1, finish, from);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "maya hy4f serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s), %s\n",
                     n, from, n - from, t1 - t0, t1 > t0 ? 1000.0 * (double) (n - from) / (t1 - t0) : 0.0, produced,
                     t2 - t1, t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0, finish);
    }
    return 0;
}

}  // namespace strata::hy4f
