// src/hy4f/hy4f_serve.cpp - maya-hy4f --serve: Maya's engine protocol on stdin / stdout, as serve/server.py's StrataEngine
// speaks it to maya-mm3 / maya-ds4 (src/mm3/mm3_main.cpp has the reference):
//
//   engine -> server   NOTE <text>, INFO key=value ..., then `READY <context> stop`
//   server -> engine   GEN <max_new> [key=value ...] <id,id,...>      one request (one conversation at a time)
//                      GENI <max_new> [key=value ...] <file> <ids>     ... with pictures (--mmproj; INFO vision=hy4)
//   engine -> server   PP <read> <total> <ms> <tok/s> <reused>         after each prefill chunk (--chunk, 2048)
//                      T <id>                                          each answer token
//                      STAT tok_s=F ms_tok=F [drafts=F draft_acc=N draft_off=N k=N]   every 8 tokens, and at the end
//                      DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> <drafts kept> <offered>
//                           <reused> 0 0
//   server -> engine   STOP (ends the request: DONE ... cancel), WRAP (the server is stopping: a reasoning block still
//                      open is closed with think_end and the answer follows), QUIT
//
// Keys: temperature (<= 0 or absent: greedy, the argmax, ties to the lower id), top_p, top_k, min_p, seed,
// penalty_last_n / penalty_repeat / penalty_freq / penalty_present (llama.cpp's, over the last n tokens of the
// conversation), logit_bias=id:bias,..., think_budget + think_end (an open reasoning block is closed with think_end after
// that many answer tokens), spec=0|1|2..4 (this request's verify window: 0 / 1 no drafts, w: w - 1 drafts fixed);
// other keys are taken and ignored.  Sampling runs on the host over the whole row of logits.
// The answer ends at an eos id (--eos; default 120025 = <｜hy_end:opensource｜>), at max_new, or at the context's end.
//
// The prompt goes through Model::prefill in chunks of --chunk tokens (the batched path, which also runs the MTP layer
// over it).  The answer: with the MTP layer, speculative steps - verify [the last token, d0 .. dk-1] in one multi-token
// step, then for row i draw y_i from the target's own distribution (greedy: its argmax; sampling: the host sampler with
// the history so far) and keep d_i while y_i == d_i (exact for deterministic drafts), the first y_i that differs is the
// bonus token; then mtp() over the kept rows drafts d0 and mtp_chain() the rest.  k adapts per request (--spec-k auto:
// start 2, from the last 16 steps' acceptance of the last draft given the ones before it: k up to 3 when it is mostly
// kept, down to 1 when rarely); a rejected tail needs no rewind (the next verify writes over it).
//
// Pictures (serve/hy4_vision.py): the file is MMI1 (int32 'MMI1', int32 count, per picture int64 start, int32 width,
// int32 height, RGB bytes - the picture as decoded, not resized: the encoder resizes it itself, Lanczos as llama.cpp);
// in the ids each picture is 120118, n x 120120 (placeholders at start ..), 120119 with n = the encoder's rows, which
// are checked here.  The rows go to prefill as embedding spans; a cached picture counts as shared only where the same
// pixels sit at the same start.
//
// The conversation in the cache: a prompt that starts with the tokens the cache holds reads only the rest (the last
// prompt token is always read again: its logits are the answer's first); one that departs from them earlier keeps the
// shared start where Model::rewind can (see the serve block of hy4f.cu: position-indexed caches, the SWA / indexer rings
// and the copy Model::mark takes after each prompt - the next chat turn, whose template drops the answer's reasoning,
// departs right there), else the cache starts again at 0.  A step that fails ends the engine (END, DONE ... cancel,
// exit 1): the server starts a new one.
#include "hy4f_serve.hpp"
#include "hy4f_host_prefix.hpp"
#include "hy4f_draft_width.hpp"
#include "hy4f_prompt_lookup.hpp"
#include "hy4f_jacobi_probe.hpp"

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
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace strata::hy4f {
namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Req {
    std::string file;   // GENI: the pictures
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
    int spec = -1;   // spec=W: verify window (W - 1 drafts); 0 / 1: none; -1: the engine's (--spec-k)
};

bool parse_gen(const std::string& line, Req& r) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    std::istringstream is(line.substr(geni ? 5 : 4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
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
        else if (k == "spec") r.spec = std::atoi(v.c_str());
        else if (k == "logit_bias") {   // id:bias,id:bias
            std::istringstream bs(v);
            for (std::string p; std::getline(bs, p, ',');) {
                const size_t c = p.find(':');
                if (c != std::string::npos) r.bias.push_back({std::atoi(p.c_str()), std::strtof(p.c_str() + c + 1, nullptr)});
            }
        }
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

// GENI's pictures (MMI1, as serve/mm3_vision.py writes it)
struct Pic {
    long long start = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    uint64_t hash = 0;   // (FNV-1a over its size and pixels)
};

bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    v.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = path + ": cannot open";
        return false;
    }
    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    char magic[4];
    int32_t n = 0;
    bool ok = rd(magic, 4) && std::memcmp(magic, "MMI1", 4) == 0 && rd(&n, 4) && n >= 0 && n <= 256;
    if (!ok) err = path + ": not a pictures file (MMI1)";
    for (int i = 0; ok && i < n; ++i) {
        Pic q;
        int64_t st = 0;
        int32_t wh[2] = {0, 0};
        ok = rd(&st, 8) && rd(wh, 8) && wh[0] > 0 && wh[1] > 0 && (long long) wh[0] * wh[1] <= (1LL << 27);
        if (ok) {
            q.start = st, q.width = wh[0], q.height = wh[1];
            q.rgb.resize((size_t) wh[0] * wh[1] * 3);
            ok = rd(q.rgb.data(), q.rgb.size());
        }
        if (!ok) {
            err = path + ": picture " + std::to_string(i) + " is cut short";
            break;
        }
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const uint8_t* b, size_t k) {
            for (size_t j = 0; j < k; ++j) h = (h ^ b[j]) * 1099511628211ull;
        };
        mix((const uint8_t*) wh, 8);
        mix(q.rgb.data(), q.rgb.size());
        q.hash = h;
        v.push_back(std::move(q));
    }
    std::fclose(f);
    return ok;
}

// stdin's lines on a thread of their own: STOP and WRAP are seen while a request runs
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false, running = false, pending_wrap = false;  // guarded by mu
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
                    std::lock_guard<std::mutex> lk(mu);
                    if (l == "WRAP") {
                        if (running) wrap.store(true);
                        else if (!q.empty()) pending_wrap = true;  // GEN and WRAP may arrive in the same read.
                        continue;
                    }
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
        running = false;
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        running = true;
        wrap.store(pending_wrap);
        pending_wrap = false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
    void finished() {
        std::lock_guard<std::mutex> lk(mu);
        running = false;
    }
};

// The next token from a row of logits: logit_bias, llama.cpp's penalties over the history's last pen_n tokens (-1: all),
// then greedy (temperature <= 0) or temperature, top_k (0: every token), min_p and top_p, drawn with rng.
struct Sampler {
    std::vector<float> row;
    std::vector<std::pair<float, int>> c;
    std::vector<int> cand;
    // the warped distribution of a row as dense probabilities P [V] (0 outside the kept set): logit_bias, the penalties,
    // temperature, then top_k / min_p / top_p on the candidates (no sort over the whole vocabulary unless a cutoff
    // needs it: the candidates are the tokens within 30 nats of the best)
    const float* warp_row(const float* lg, int V, const Req& r, const std::vector<int32_t>& hist) {
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
            return row.data();
        }
        return lg;
    }
    void probs(const float* lg, int V, const Req& r, const std::vector<int32_t>& hist, std::vector<float>& P) {
        lg = warp_row(lg, V, r, hist);
        P.assign((size_t) V, 0.0f);
        const float inv = 1.0f / r.temperature;
        float mx = lg[0];
        for (int i = 1; i < V; ++i) mx = std::max(mx, lg[i]);
        const bool cut = r.top_k > 0 || r.top_p < 1.0f || r.min_p > 0.0f;
        double sum = 0;
        if (!cut) {   // (float expf; tokens more than 20 nats below the best weigh < 2e-9: left at 0)
            float fs = 0.0f;
            for (int i = 0; i < V; ++i) {
                const float z = (lg[i] - mx) * inv;
                const float e = z > -20.0f ? expf(z) : 0.0f;
                P[(size_t) i] = e;
                fs += e;
            }
            sum = fs;
        } else {
            cand.clear();
            for (int i = 0; i < V; ++i)
                if ((lg[i] - mx) * inv > -20.0f) cand.push_back(i);
            c.resize(cand.size());
            for (size_t j = 0; j < cand.size(); ++j) c[j] = {lg[cand[j]] * inv, cand[j]};
            const size_t k = r.top_k > 0 ? std::min((size_t) r.top_k, c.size()) : c.size();
            std::partial_sort(c.begin(), c.begin() + (long) k, c.end(), [](const auto& a2, const auto& b2) {
                return a2.first > b2.first || (a2.first == b2.first && a2.second < b2.second);
            });
            std::vector<double> e(k);
            double tot = 0;
            for (size_t i = 0; i < k; ++i) tot += (e[i] = std::exp((double) (c[i].first - c[0].first)));
            size_t keep = k;
            double cum = 0;
            for (size_t i = 0; i < k; ++i) {
                cum += e[i] / tot;
                if (r.min_p > 0.0f && e[i] < r.min_p * e[0]) {
                    keep = std::max<size_t>(1, i);
                    break;
                }
                if (r.top_p < 1.0f && cum >= r.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            for (size_t i = 0; i < keep; ++i) sum += (P[(size_t) c[i].second] = (float) e[i]);
        }
        const float is = (float) (1.0 / sum);
        for (int i = 0; i < V; ++i) P[(size_t) i] *= is;
    }
    static int draw(const std::vector<float>& P, std::mt19937_64& rng) {
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng), acc = 0;
        int last = 0;
        for (size_t i = 0; i < P.size(); ++i)
            if (P[i] > 0.0f) {
                acc += P[i];
                last = (int) i;
                if (u <= acc) return (int) i;
            }
        return last;
    }
    using Entry = std::pair<int, float>;
    using Sparse = std::vector<Entry>;  // increasing token ID, matching dense draw order
    static float at(const Sparse& p, int id) {
        auto i = std::lower_bound(p.begin(), p.end(), id, [](const Entry& v, int key) { return v.first < key; });
        return i == p.end() || i->first != id ? 0.f : i->second;
    }
    static int sparse_draw(const Sparse& p, std::mt19937_64& rng) {
        const double u = std::uniform_real_distribution<double>(0., 1.)(rng);
        double acc = 0.;
        for (const auto& v : p) {
            acc += v.second;
            if (u <= acc) return v.first;
        }
        return p.empty() ? 0 : p.back().first;
    }
    static void residual(const Sparse& p, const Sparse& q, Sparse& out) {
        out.clear();
        auto j = q.begin();
        double sum = 0.;
        for (const auto& v : p) {
            while (j != q.end() && j->first < v.first) ++j;
            const float diff = v.second - (j != q.end() && j->first == v.first ? j->second : 0.f);
            if (diff > 0.f) { out.emplace_back(v.first, diff); sum += diff; }
        }
        if (sum > 0.) {
            const float inv = (float)(1. / sum);
            for (auto& v : out) v.second *= inv;
        }
    }
    static float max_logit(const float* lg, int n) {
#if defined(__SSE2__)
        // With lg[0] as the right operand, later NaNs leave the previous maximum alone.
        if (!std::isnan(lg[0])) {
            __m128 a = _mm_set1_ps(lg[0]), b = a, c = a, d = a;
            int i = 0;
            for (; i + 16 <= n; i += 16) {
                a = _mm_max_ps(_mm_loadu_ps(lg + i), a);
                b = _mm_max_ps(_mm_loadu_ps(lg + i + 4), b);
                c = _mm_max_ps(_mm_loadu_ps(lg + i + 8), c);
                d = _mm_max_ps(_mm_loadu_ps(lg + i + 12), d);
            }
            float lane[4];
            _mm_storeu_ps(lane, _mm_max_ps(_mm_max_ps(a,b), _mm_max_ps(c,d)));
            float best = *std::max_element(lane, lane + 4);
            for (; i < n; ++i) best = std::max(best, lg[i]);
            return best;
        }
#endif
        return *std::max_element(lg, lg+n);
    }
    void sparse_probs(const float* input, int V, const Req& r, const std::vector<int32_t>& hist, Sparse& out) {
        const float* lg = warp_row(input, V, r, hist);
        const float mx = max_logit(lg, V);
        const float inv = 1.f / r.temperature;
        const size_t k = (size_t)r.top_k;
        auto better = [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        };
        c.clear();
        auto offer = [&](int id) {
            if (!((lg[id] - mx) * inv > -20.f)) return;
            const std::pair<float,int> v = {lg[id] * inv, id};
            if (c.size() < k) {
                c.push_back(v);
                std::push_heap(c.begin(), c.end(), better);
            } else if (better(v, c.front())) {
                std::pop_heap(c.begin(), c.end(), better);
                c.back() = v;
                std::push_heap(c.begin(), c.end(), better);
            }
        };
        int t = 0;
        for (; t < V && c.size() < k; ++t) offer(t);
#if defined(__SSE2__)
        // Ties with the heap floor cannot replace an earlier (lower-ID) token.
        const __m128 factor = _mm_set1_ps(inv);
        for (; t+4 <= V; t+=4) {
            const __m128 score = _mm_mul_ps(_mm_loadu_ps(lg+t), factor);
            unsigned mask = (unsigned)_mm_movemask_ps(_mm_cmpgt_ps(score, _mm_set1_ps(c.front().first)));
            while (mask) { offer(t + __builtin_ctz(mask)); mask &= mask-1; }
        }
#endif
        for (; t < V; ++t) offer(t);
        std::sort_heap(c.begin(), c.end(), better);
        normalize_sparse(r,out);
    }
    void ranked_probs(const std::vector<HeadValue>& values, const Req& r, Sparse& out) {
        const float inv=1.f/r.temperature;
        float mx=-INFINITY;
        for (const auto& v : values) mx=std::max(mx,v.logit);
        c.clear();
        for (const auto& v : values) if ((v.logit-mx)*inv>-20.f)
            c.emplace_back(v.logit*inv,v.id);
        std::sort(c.begin(),c.end(),[](const auto& a,const auto& b) {
            return a.first>b.first || (a.first==b.first && a.second<b.second);
        });
        if (c.size()>(size_t)r.top_k) c.resize(r.top_k);
        normalize_sparse(r,out);
    }
    void normalize_sparse(const Req& r, Sparse& out) {
        // Use the same types, rounding points and summation order as probs().
        exponent.resize(c.size());
        double total = 0.;
        for (size_t j = 0; j < c.size(); ++j)
            total += (exponent[j] = std::exp((double)(c[j].first - c[0].first)));
        out.clear();
        double cum = 0., sum = 0.;
        for (size_t j = 0; j < c.size(); ++j) {
            const auto e = exponent[j];
            cum += e / total;
            if (j > 0 && r.min_p > 0.f && e < r.min_p * exponent.front()) break;
            out.emplace_back(c[j].second, (float)e);
            sum += (float)e;
            if (r.top_p < 1.f && cum >= r.top_p) break;
        }
        const float norm = (float)(1. / sum);
        for (auto& v : out) v.second *= norm;
        std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.first < b.first; });
    }
    std::vector<double> exponent;
    std::vector<float> pd;
    int next(const float* lg, int V, const Req& r, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
        if (!(r.temperature > 0.0f)) {
            const float* l = lg;
            if (!r.bias.empty() || (r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f))) {
                // (greedy with a bias / penalty: the warp at temperature 1, the argmax of it)
                Req r1 = r;
                r1.temperature = 1.0f, r1.top_k = 0, r1.top_p = 1.0f, r1.min_p = 0.0f;
                probs(lg, V, r1, hist, pd);
                return (int) (std::max_element(pd.begin(), pd.end()) - pd.begin());
            }
            int b2 = 0;
            for (int i = 1; i < V; ++i)
                if (l[i] > l[b2]) b2 = i;
            return b2;
        }
        probs(lg, V, r, hist, pd);
        return draw(pd, rng);
    }
};

// The draft count k (1..3) from the last steps: for the last draft offered, how often it was kept when every draft
// before it was (rate(k - 1)); up when mostly kept, down when rarely.  The cost: a verify of 4 rows is little more than
// one of 3 on this engine, so a kept draft pays at moderate rates.
struct KCtl {
    int k = 2;
    std::deque<std::pair<int, int>> h;   // (drafts offered, kept) per step
    double rate(int j, int& n) const {   // P(draft j kept | drafts 0..j-1 kept, j offered)
        int seen = 0, kept = 0;
        for (const auto& [off, acc] : h)
            if (off > j && acc >= j) ++seen, kept += acc > j;
        n = seen;
        return seen ? (double) kept / seen : 0.0;
    }
    void step(int offered, int kept) {
        h.push_back({offered, kept});
        if (h.size() > 16) h.pop_front();
        int n = 0;
        const double r = rate(k - 1, n);
        if (n < 8) return;
        const int was = k;
        if (k < 3 && r >= 0.7) ++k;
        else if (k == 3 && r < 0.45) --k;
        else if (k == 2 && r < 0.3) --k;
        if (k != was) h.clear();
    }
};

}  // namespace

int serve(Model& m, const ServeOptions& so) {
    const int V = m.vocab();
    const long long C = m.max_context();
    const int CH = std::max(1, so.chunk);
    const bool mtp = m.has_mtp() && so.spec != 0;
    std::string err;
    {   // the prefill arena for --chunk tokens, set up before READY (a card too full for it fails here, not mid-request)
        const int32_t bos = 120000;
        if (!m.prefill(&bos, 1, 0, nullptr, CH, err)) {
            std::fprintf(stderr, "maya-hy4f: the prefill arena (chunk %d): %s\n", CH, err.c_str());
            return 1;
        }
        m.reset();
    }
    const char* host_setting = std::getenv("HY4F_HOST_PREFIX_SLOTS");
    const bool host_supported = CH == int(HostPrefixIndex::block);
    const int host_slots = host_setting && host_supported &&
        (std::strcmp(host_setting, "2") == 0 || std::strcmp(host_setting, "3") == 0) ? host_setting[0] - '0' : 0;
    HostPrefixIndex host_keys(host_slots, std::getenv("HY4F_DEEP_HOST_PREFIX") != nullptr);
    if (host_setting && !host_slots)
        std::fprintf(stderr, "maya hy4f serve: host prefix cache requires slots 2 or 3 and chunk 2048\n");
    const char* reserve = std::getenv("HY4F_HOST_PREFIX_PREALLOC");
    if (host_slots && reserve && std::strcmp(reserve, "1") == 0) {
        const double started = now_ms();
        const bool ok = m.preallocate_host_prefixes(host_slots, HostPrefixIndex::checkpoint(C),
                                                    host_keys.deep_checkpoint(C), err);
        std::fprintf(stderr, "maya hy4f serve: host prefix prealloc before READY ms=%.1f status=%s\n",
                     now_ms() - started, ok ? "ready" : "on-demand");
        if (!ok) std::fprintf(stderr, "maya hy4f serve: host prefix prealloc failed (%s)\n", err.c_str());
    }
    std::printf("INFO engine=hy4f engine_kind=hy4f model=%s ctx=%lld kv=q8 logit_bias=1 prefill_chunk=%d mtp=%d "
                "spec_k=%s\n", so.model_name.c_str(), C, CH, mtp ? 1 : 0,
                !mtp ? "0" : so.spec < 0 ? "auto" : std::to_string(std::min(so.spec, 3)).c_str());
    if (m.has_vision())   // (serve/hy4_vision.py's Geometry.from_info; the mmproj's limits, hy4f_vision.hpp)
        std::printf("INFO vision=hy4 image_token=120120 image_start=120118 image_end=120119 vision_patch=16 "
                    "vision_merge=2 vision_min_pixels=4096 vision_max_pixels=4194304\n");
    else
        std::printf("INFO vision=off\n");
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
    const bool turn_mark = std::getenv("HY4F_PREFIX_TURN_MARK") != nullptr;
    size_t saved_prompt = 0;
    bool prefix_mark = std::getenv("HY4F_PREFIX_CHECKPOINT") != nullptr;
    std::string line;
    std::vector<float> lgT((size_t) 4 * V);   // verify's rows (sampling, penalties, a bias)
    struct Held {
        long long start, n;
        uint64_t hash;
    };
    std::vector<Held> held;   // the pictures in the cache
    auto fatal = [&](const std::string& why, long long produced, size_t n_prompt) {
        std::fprintf(stderr, "maya-hy4f: %s\n", why.c_str());
        std::printf("END hy4f: %s\n", why.c_str());
        std::printf("DONE %lld %zu 0.0 0.0 cancel 0 0 0 0 0\n", produced, n_prompt);
        std::fflush(stdout);
        std::exit(1);
    };
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP, WRAP and QUIT)\n",
                        line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> (GENI: the pictures' file before the ids)\n");
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
        std::vector<Pic> pics;
        if (!r.file.empty()) {
            std::string perr;
            if (!m.has_vision()) perr = "this engine was started without --mmproj: no pictures";
            else if (!read_pics(r.file, pics, perr)) {}
            else
                for (const Pic& q : pics)
                    if (q.start < 1 || q.start >= (long long) n || r.ids[(size_t) q.start - 1] != 120118 ||
                        r.ids[(size_t) q.start] != 120120)
                        perr = "a picture's start " + std::to_string(q.start) + " is not after 120118 at a 120120";
            if (!perr.empty()) {
                std::printf("ERR %s\n", perr.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
                std::fflush(stdout);
                continue;
            }
        }
        // the cache's shared start (the last prompt token is always read again); a picture before it in the cache or
        // the request must be in both, the same pixels at the same start (its rows have no token ids to compare)
        size_t from = 0;
        while (from < state.size() && from < n && state[from] == r.ids[from]) ++from;
        const size_t shared_tokens = from;
        for (const Held& q : held)
            if (q.start < (long long) from && std::none_of(pics.begin(), pics.end(), [&](const Pic& x) {
                    return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) q.start;
        for (const Pic& x : pics)
            if (x.start < (long long) from && std::none_of(held.begin(), held.end(), [&](const Held& q) {
                    return x.start == q.start && x.hash == q.hash; }))
                from = (size_t) x.start;
        // Changing pixels can leave a common text prefix ending in the middle
        // of a prefill chunk. Replay that chunk with the new image's rows.
        if (turn_mark && from < shared_tokens) from -= from % CH;
        const bool full_hit = turn_mark && saved_prompt == n && from == n;
        if (!full_hit) {
            saved_prompt = 0;
            from = std::min(from, n - 1);
        }
        bool ring_cold_aligned = false;
        const int64_t kept = m.rewind((int64_t) from, err, full_hit, (turn_mark || host_keys.enabled()) ? &ring_cold_aligned : nullptr);
        if (kept < 0) fatal("rewind: " + err, 0, n);
        if ((size_t) kept != from)
            std::fprintf(stderr, "maya hy4f serve: the prompt departs from the cache at %zu (the rings no longer hold it): "
                                 "read from %lld\n", from, (long long) kept);
        from = (size_t) kept;
        const bool text_cache = host_keys.enabled() && r.file.empty();
        const int host_slot = text_cache && !full_hit ? host_keys.lookup(r.ids, from) : -1;
        bool host_restored = false;
        if (host_slot >= 0) {
            const size_t pos = host_keys.length(host_slot);
            const double started = now_ms();
            if (m.load_host_prefix(host_slot, pos, err)) {
                from = pos;
                state.assign(r.ids.begin(), r.ids.begin() + pos);
                held.clear();
                host_restored = true;
                std::fprintf(stderr, "maya hy4f serve: host prefix hit slot=%d rows=%zu ms=%.1f\n",
                             host_slot, pos, now_ms() - started);
            } else {
                // A failed H2D copy can leave a partly restored branch. Rebuild all state from the beginning.
                std::fprintf(stderr, "maya hy4f serve: host prefix restore failed (%s), restarting cold\n", err.c_str());
                host_keys.clear(host_slot);
                m.reset();
                from = 0;
            }
        }
        // Capture only a lineage that used the same 2048-token chunk boundaries as a cold prefill.
        const bool cold_aligned = from == 0 || host_restored || ring_cold_aligned;
        const size_t host_checkpoint = text_cache && cold_aligned ? HostPrefixIndex::checkpoint(n) : 0;
        const size_t deep_checkpoint = text_cache && cold_aligned ? host_keys.deep_checkpoint(n) : 0;
        state.resize(from);
        held.erase(std::remove_if(held.begin(), held.end(), [&](const Held& q) { return q.start + q.n > (long long) from; }),
                   held.end());
        // the pictures whose rows the prompt still reads: encoded now (between prefill / verify calls)
        std::vector<std::vector<float>> prow(pics.size());
        std::vector<int> pn(pics.size(), 0);   // each picture's image tokens in the prompt
        std::vector<EmbdSpan> spans;
        bool pic_bad = false;
        for (size_t i = 0; i < pics.size() && !pic_bad; ++i) {
            Pic& q = pics[i];
            long long z = q.start;
            while (z < (long long) n && r.ids[(size_t) z] == 120120) ++z;
            pn[i] = (int) (z - q.start);
            if (z <= (long long) from) continue;   // (in the cache already: the same pixels at the same start)
            const std::string ppm = "P6\n" + std::to_string(q.width) + " " + std::to_string(q.height) + "\n255\n";
            std::vector<uint8_t> buf(ppm.begin(), ppm.end());
            buf.insert(buf.end(), q.rgb.begin(), q.rgb.end());
            int nt = 0;
            const double te = now_ms();
            if (!m.encode_image_mem(buf.data(), buf.size(), prow[i], nt, err)) {
                std::printf("ERR picture %zu: %s\n", i, err.c_str());
                pic_bad = true;
                break;
            }
            if (z - q.start != nt || z >= (long long) n || r.ids[(size_t) z] != 120119) {
                std::printf("ERR picture %zu: the encoder made %d rows, the prompt has %lld image tokens there\n", i, nt,
                            z - q.start);
                pic_bad = true;
                break;
            }
            std::fprintf(stderr, "maya hy4f serve: picture %zu: %dx%d -> %d rows in %.0f ms\n", i, q.width, q.height, nt,
                         now_ms() - te);
            if (q.start + nt > (long long) from) spans.push_back({(int) q.start, nt, prow[i].data()});
        }
        if (pic_bad) {
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 %zu 0 0\n", n, from);
            std::fflush(stdout);
            continue;
        }
        bool cancel = false;
        auto pp_line = [&](size_t done) {
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done, n, ms, ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        if (full_hit && from == n) pp_line(n);
        for (size_t a = from; a < n;) {
            const int T = (int) std::min<size_t>((size_t) CH, n - a);
            const bool last = a + (size_t) T == n;
            std::vector<EmbdSpan> cs;   // (the spans relative to this call's tokens; prefill clips them)
            for (const EmbdSpan& e : spans) cs.push_back({e.offset - (int) a, e.n, e.rows});
            if (!m.prefill(r.ids.data() + a, T, (int64_t) a, last ? lg.data() : nullptr, CH, err, cs.empty() ? nullptr : &cs))
                fatal("prefill at " + std::to_string(a) + ": " + err, 0, n);
            state.insert(state.end(), r.ids.begin() + (long) a, r.ids.begin() + (long) a + T);
            a += (size_t) T;
            if (a == host_checkpoint || a == deep_checkpoint) {
                const int slot = host_keys.destination(state);
                if (slot >= 0) {
                    const double started = now_ms();
                    host_keys.clear(slot);
                    double d2h_ms = 0;
                    if (m.save_host_prefix(slot, err, &d2h_ms)) {
                        host_keys.publish(slot, state);
                        std::fprintf(stderr, "maya hy4f serve: host prefix saved slot=%d rows=%zu ms=%.1f d2h_ms=%.1f\n",
                                     slot, a, now_ms() - started, d2h_ms);
                    } else {
                        std::fprintf(stderr, "maya hy4f serve: host prefix save failed (%s)\n", err.c_str());
                    }
                }
            }
            if (prefix_mark && a >= 2*(size_t)CH && a%CH == 0 && a<n && n-a <= (size_t)CH) {
                if (!m.mark_prefix(err, cold_aligned)) {
                    std::fprintf(stderr,"maya hy4f serve: prefix checkpoint unavailable (%s)\n",err.c_str());
                    prefix_mark=false;
                }
            }
            pp_line(a);
            if (in.stop.load() && !last) {
                cancel = true;
                break;
            }
        }
        const double t1 = now_ms();
        for (size_t i = 0; i < pics.size(); ++i)   // (the pictures now in the cache)
            if (pics[i].start + pn[i] <= (long long) state.size() &&
                std::none_of(held.begin(), held.end(), [&](const Held& q) { return q.start == pics[i].start; }))
                held.push_back({pics[i].start, (long long) pn[i], pics[i].hash});
        if (cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", n, t1 - t0, from);
            std::fflush(stdout);
            continue;
        }
        if (mark_ok && !full_hit && !m.mark(err, from % CH == 0 && n % CH == 0, cold_aligned && n % CH == 0)) {   // (the rings at the prompt's end: the next turn may go back here)
            std::fprintf(stderr, "maya hy4f serve: no copy of the rings (%s) - a turn that departs from the answer "
                                 "reads its prompt again\n", err.c_str());
            mark_ok = false;
        }
        // An unaligned warm suffix and a cold prefill can select different GEMM
        // shapes. Preserve the established final-token replay for that lineage.
        if (turn_mark && mark_ok && (full_hit || cold_aligned)) saved_prompt = n;
        ++counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        long long produced = 0;
        const char* finish = "length";
        bool in_think = r.think_budget > 0 && r.think_end >= 0;
        long long d_acc = 0, d_off = 0;
        KCtl kc;
        if (so.spec > 0) kc.k = std::min(so.spec, 3);
        const bool adapt = so.spec < 0 && r.spec < 0;
        if (r.spec >= 0) kc.k = std::min(std::max(r.spec - 1, 0), 3);
        const bool spec = mtp && kc.k > 0;
        auto stat_line = [&]() {
            const double ms = now_ms() - t1;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f", ms > 0 ? 1000.0 * (double) produced / ms : 0.0,
                        produced > 0 ? ms / (double) produced : 0.0);
            if (d_off > 0)
                std::printf(" drafts=%.3f draft_acc=%lld draft_off=%lld k=%d", (double) d_acc / (double) d_off, d_acc, d_off,
                            kc.k);
            std::printf("\n");
        };
        const bool rows = r.temperature > 0.0f || !r.bias.empty() ||
                          (r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f));
        // the token for a row: drawn from the target (greedy without a bias / penalty: the verify's argmax), then the
        // thinking budget or WRAP closing an open reasoning block
        const bool sparse_sample = std::getenv("HY4F_SPARSE_SAMPLE") && r.temperature > 0.f && r.top_k > 0 && r.top_k <= 256;
        Sampler::Sparse sp, sq[4], sr;
        m.set_greedy_draft_head(r.temperature <= 0.0f);
        const bool ranked=m.set_ranked_output(sparse_sample && r.top_k<=64 && r.bias.empty() &&
            !(r.pen_n && (r.pen_repeat!=1.f || r.pen_freq!=0.f || r.pen_present!=0.f)));
        std::vector<HeadValue> head_values;
        auto sparse_row = [&](float* row, int i, const Req& req, Sampler::Sparse& probs) {
            if (ranked && i>=0 && m.ranked_row(i,req.top_k,1.f/req.temperature,head_values,row))
                sampler.ranked_probs(head_values,req,probs);
            else sampler.sparse_probs(row,V,req,state,probs);
        };
        auto pick = [&](float* row, int32_t amx, int i=-1) -> int {
            int t;
            if (sparse_sample) { sparse_row(row,i,r,sp); t = Sampler::sparse_draw(sp, rng); }
            else t = rows ? sampler.next(row, V, r, state, rng) : amx;
            if (in_think && (produced >= r.think_budget || in.wrap.load()) && t != r.think_end) t = r.think_end;
            return t;
        };
        // one token out: false when the answer ends with it (an end token, max_new)
        PromptLookup lookup(r.ids, spec && !rows);
        JacobiProbe jacobi(spec && !rows);
        bool borrowed = false;
        auto emit = [&](int t) -> bool {
            lookup.append(t);
            std::printf("T %d\n", t);
            ++produced;
            jacobi.emitted(t);
            if (in_think && t == r.think_end) in_think = false;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            if (std::find(so.eos.begin(), so.eos.end(), t) != so.eos.end()) {
                finish = "stop";
                return false;
            }
            return produced < r.max_new;
        };
        int32_t amx0 = 0;
        if (!rows) amx0 = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        int tok = pick(lg.data(), amx0);   // the next token: drawn, not yet in the cache
        bool more = !in.stop.load() && emit(tok);
        if (in.stop.load()) finish = "cancel";
        int32_t dr[4] = {0, 0, 0, 0}, am[4];
        // drafts d0 (the MTP over the kept rows, or after the prompt its last state) and d1.. chained
        // sampling: drafts drawn from the MTP's own (warped) distribution q_i, kept with probability min(1, p/q) and on a
        // rejection the token drawn from max(0, p - q) (speculative sampling: the output is distributed exactly as the
        // target's; the acceptance is 1 - TV(p, q) instead of p(d) for an argmax draft)
        const bool ssamp = r.temperature > 0.0f;
        Req rq = r;   // the drafts' warp: the MTP distribution at a lower temperature (any q keeps the output exact)
        if (const char* e = std::getenv("HY4F_DRAFT_TEMP_SCALE")) rq.temperature = r.temperature * (float) std::atof(e);
        (void) 0;   // (default: the same temperature - measured best: 0.6 and 0.8 accepted less)
        std::vector<float> mlg(ssamp ? (size_t) V : 0);
        std::vector<std::vector<float>> Q(ssamp ? 4 : 0);
        DraftWidth width(adapt && spec && !rows);
        if (width.enabled()) kc.k = 3;
        int offered = kc.k;
        auto draft = [&](const int32_t* nt, int T, int64_t mpos) {
            if (!m.mtp(nt, T, mpos, &dr[0], err, ssamp && !ranked ? mlg.data() : nullptr))
                fatal("mtp at " + std::to_string(mpos) + ": " + err, produced, n);
            if (sparse_sample) {
                sparse_row(mlg.data(),0,rq,sq[0]);
                dr[0] = Sampler::sparse_draw(sq[0], rng);
            } else if (ssamp) {
                sampler.probs(mlg.data(), V, rq, state, Q[0]);
                dr[0] = Sampler::draw(Q[0], rng);
            }
            const int64_t next = mpos + T;
            width.begin();
            offered = 1;
            width.offer(0, m, dr[0]);
            const int room = width.enabled() ? (int) std::min<long long>({(long long) kc.k, C - next - 1, r.max_new - produced}) : kc.k;
            const int copied = lookup.extend(dr[0], room-1, dr+1);
            borrowed = copied > 0;
            if (borrowed) { offered += copied; return; }
            for (int i = 1; i < room && width.extend(i - 1); ++i) {
                if (!m.mtp_chain(dr[i - 1], next + i - 1, &dr[i], err, ssamp && !ranked ? mlg.data() : nullptr))
                    fatal("mtp_chain at " + std::to_string(next + i - 1) + ": " + err, produced, n);
                if (sparse_sample) {
                    sparse_row(mlg.data(),0,rq,sq[i]);
                    dr[i] = Sampler::sparse_draw(sq[i], rng);
                } else if (ssamp) {
                    sampler.probs(mlg.data(), V, rq, state, Q[i]);
                    dr[i] = Sampler::draw(Q[i], rng);
                }
                if (!width.offer(i, m, dr[i])) break;
                ++offered;
            }
        };
        std::vector<float> Pr, Res;
        if (more && spec) draft(&tok, 1, (int64_t) state.size() - 1);
        while (more) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            const int64_t pos = (int64_t) state.size();
            if (pos >= C) break;   // (the context is full: length)
            const int kk = spec ? (int) std::min<long long>({(long long) offered, C - pos - 1, r.max_new - produced}) : 0;
            const int32_t vt[4] = {tok, dr[0], dr[1], dr[2]};
            if (!m.verify(vt, kk + 1, pos, rows && !ranked ? lgT.data() : nullptr, am, err))
                fatal("verify at " + std::to_string(pos) + ": " + err, produced, n);
            state.push_back(tok);
            int acc = 0;
            for (int i = 0;; ++i) {
                int y;
                bool kept;
                if (ssamp && i < kk) {
                    float* logits = lgT.data() + (size_t)i * V;
                    if (sparse_sample) sparse_row(logits,i,r,sp);
                    else sampler.probs(logits, V, r, state, Pr);
                    const float pq = sparse_sample ? Sampler::at(sp, dr[i]) : Pr[(size_t)dr[i]];
                    const float qq = sparse_sample ? Sampler::at(sq[i], dr[i]) : Q[i][(size_t)dr[i]];
                    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                    kept = qq > 0.0f && u < std::min(1.0, (double) pq / (double) qq);
                    if (kept) {
                        y = dr[i];
                    } else if (sparse_sample) {
                        Sampler::residual(sp, sq[i], sr);
                        y = Sampler::sparse_draw(sr.empty() ? sp : sr, rng);
                    } else {
                        Res.assign((size_t) V, 0.0f);
                        double rs = 0;
                        for (int v = 0; v < V; ++v) {
                            const float d2 = Pr[(size_t) v] - Q[i][(size_t) v];
                            if (d2 > 0.0f) rs += (Res[(size_t) v] = d2);
                        }
                        if (rs > 0) {
                            const float is = (float) (1.0 / rs);
                            for (auto& x : Res) x *= is;
                            y = Sampler::draw(Res, rng);
                        } else {
                            y = Sampler::draw(Pr, rng);
                        }
                    }
                    if (in_think && (produced >= r.think_budget || in.wrap.load()) && y != r.think_end) {
                        y = r.think_end;
                        kept = y == dr[i];
                    }
                } else {
                    y = pick(lgT.data() + (size_t) i * V, am[i], i);
                    kept = i < kk && y == dr[i];
                }
                if (i < kk && kept) {   // the draft kept: its row is the cache's now
                    ++acc;
                    state.push_back(y);
                    if (!emit(y)) {
                        more = false;
                        break;
                    }
                    continue;
                }
                tok = y;   // the bonus: drawn from row i, not in the cache yet
                if (!emit(y)) more = false;
                break;
            }
            jacobi.step(int(pos - (int64_t)n), kk, acc, am, dr);
            d_off += kk;
            d_acc += acc;
            if (!more || !spec) continue;
            if (width.enabled()) width.observe(borrowed ? std::min(1, kk) : kk, acc);
            else if (adapt) kc.step(kk, acc);
            int32_t nt[4];
            for (int i = 0; i < acc; ++i) nt[i] = dr[i];
            nt[acc] = tok;
            draft(nt, acc + 1, pos);   // (the verify's rows 0 .. acc, at pos ..)
        }
        stat_line();
        const double t2 = now_ms();
        jacobi.report(counter, n);
        in.finished();   // A WRAP after DONE can only belong to a later queued request.
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, n, t1 - t0, t2 - t1, finish, d_acc, d_off,
                    from);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "maya hy4f serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s), drafts %lld of %lld kept (k %d at the end), %s\n",
                     n, from, n - from, t1 - t0, t1 > t0 ? 1000.0 * (double) (n - from) / (t1 - t0) : 0.0, produced,
                     t2 - t1, t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0, d_acc, d_off, kc.k, finish);
    }
    return 0;
}

}  // namespace strata::hy4f
