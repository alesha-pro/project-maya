// src/hy4f/hy4f.cu - maya-hy4f v3: the Hy4 flash decode, tensor parallel over the GPUs, one CUDA graph per GPU per step.
//
// Sharding (n_shards = 4 GPUs, docs/MAYA-PORT.md "TP4"):
//   attention   heads split: q_b / k_b / v_b / attn_gate rows and attn_output columns of the shard's heads; the
//               partial attn_output summed over the shards (all-reduce of n_embd floats).  q_a, kv_a (the MQA latent)
//               and the k-pool indexer replicated.
//   KV (Q8)     SWA layers: a 512-row ring, replicated.  DSA layers: context parallel - shard g keeps the positions
//               p % S == g; q_b / k_b of the DSA layers replicated, each shard computes the partial softmax of ALL heads
//               over its cells, the partials are exchanged and each shard combines its own heads (with the sink).
//   MoE         experts split by index (n_exp / S per shard), the router replicated; each shard runs the selected
//               experts it owns (the others -1: zero), routed_up of its partial latent sum + its slice of the shared
//               expert -> all-reduce.  Dense layer 0: rows / columns split.
//   HC, norms, Engram, embedding: replicated.  Head: vocabulary rows split, the logits gathered on the host.
// Every kernel reads the position / token from device memory (kern::Io), so each shard's step is one captured graph.
#include "hy4f.hpp"
#include "hy4f_kern.cuh"
#include "hy4f_select.cuh"
#include "hy4f_bkern.cuh"
#include "hy4f_mqa.cuh"
#include "hy4f_dkern.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cublas_v2.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace strata::hy4f {

namespace kn = strata::kernels;
using namespace kern;

namespace {

#define HY_CK(x)                                                                                                        \
    do {                                                                                                                \
        cudaError_t e_ = (x);                                                                                           \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)

enum : int { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q3_K = 11, T_Q4_K = 12, T_Q6_K = 14, T_BF16 = 30, T_F8 = 43 };

size_t row_bytes(int type, int64_t n) {
    switch (type) {
    case T_F32: return (size_t) n * 4;
    case T_F16:
    case T_BF16: return (size_t) n * 2;
    case T_F8: return (size_t) n;
    case T_Q4_0: return (size_t) n / 32 * 18;
    case T_Q8_0: return (size_t) n / 32 * 34;
    case T_Q3_K: return (size_t) n / 256 * 110;
    case T_Q4_K: return (size_t) n / 256 * 144;
    case T_Q6_K: return (size_t) n / 256 * 210;
    default: throw std::runtime_error("hy4f: unsupported tensor type " + std::to_string(type));
    }
}
int block_elems(int type) {
    switch (type) {
    case T_Q4_0:
    case T_Q8_0: return 32;
    case T_Q3_K:
    case T_Q4_K:
    case T_Q6_K: return 256;
    default: return 1;
    }
}
int nb(int n, int t = 256) { return (n + t - 1) / t; }

struct DW {
    int type = -1;
    int64_t ne[3] = {1, 1, 1};
    void* d = nullptr;
    size_t rb = 0;
};

struct Shard {
    int g = 0, dev = 0;
    cudaStream_t st = nullptr;
    std::map<std::string, DW> w;
    std::map<int, uint8_t*> kv;       // layer -> Q8 rows (SWA ring / DSA context-parallel slots)
    std::map<int, float*> ring;
    std::map<int, __half*> pooled;   // the k-pool keys, F16 (replicated: the score reads every group)
    Io* dio = nullptr;
    int* hist = nullptr;
    void* q8 = nullptr;
    void* q8b = nullptr;
    float *X = nullptr, *R = nullptr, *cur = nullptr, *x = nullptr, *y = nullptr, *post = nullptr, *hcp = nullptr,
          *hcs = nullptr;
    float *qa = nullptr, *qr = nullptr, *q = nullptr, *kvrow = nullptr, *qabs = nullptr, *o = nullptr, *a = nullptr,
          *gate = nullptr, *ao = nullptr, *part = nullptr, *sc = nullptr;
    float *iq = nullptr, *ik = nullptr, *ikn = nullptr, *iv = nullptr, *ig = nullptr, *iw = nullptr, *score = nullptr;
    int* cells = nullptr;
    float *lg = nullptr, *ew = nullptr, *xl = nullptr, *ga = nullptr, *up = nullptr, *moe = nullptr, *mu = nullptr,
          *sh = nullptr, *ffn = nullptr, *ga16 = nullptr, *up16 = nullptr, *dn16 = nullptr;
    int *eids = nullptr, *eloc = nullptr;
    float *eg_r = nullptr, *eg_kv = nullptr;
    int64_t* eg_ids = nullptr;
    int64_t* eg_map = nullptr;
    uint64_t *eg_mult = nullptr, *eg_vsz = nullptr, *eg_off = nullptr;
    float *head_cur = nullptr, *head_x = nullptr, *logits = nullptr;
    float* ar_buf[2] = {nullptr, nullptr};
    int* ar_flag = nullptr;
    float* cb_buf[2] = {nullptr, nullptr};
    int* cb_flag = nullptr;
    int *ar_ctr = nullptr, *cb_ctr = nullptr;
    Peers ar_peers{}, cb_peers{};
    // v3.1: merged projections' outputs and their Q8_1 inputs
    float *xp = nullptr, *qp = nullptr, *ivg = nullptr, *xm = nullptr, *gu = nullptr, *kvdump = nullptr;
    Q81 *xq = nullptr, *qrq = nullptr, *ikq = nullptr, *aq = nullptr, *xlq = nullptr, *moeq = nullptr, *shq = nullptr,
        *guq = nullptr, *q16 = nullptr;
    cudaGraphExec_t exec = nullptr;
};

}  // namespace

struct Model::Impl {
    Options o;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    int E = 0, hc = 0, L = 0, V = 0, q_lora = 0, kvl = 0, nope = 0, rope = 0, vh = 0, Hd = 0, Hs = 0, win = 0, IH = 0,
        IK = 0, top_k = 0, kpool = 0, tail = 0, n_exp = 0, n_used = 0, n_ffx = 0, latent = 0, dense_lead = 0, n_ff = 0;
    float eps = 1e-5f, hc_eps = 1e-6f, hc_mag = 2.0f, w_scale = 1.0f, clampv = 10.0f, theta = 1e7f, theta_swa = 1e4f;
    std::vector<int> swa, eg_layers;
    std::vector<uint64_t> eg_mult, eg_off, eg_vsz;
    std::vector<int64_t> eg_map;
    int eg_ngram = 3, eg_heads = 8, eg_D = 96, eg_pad = 0;
    uint8_t* eg_table = nullptr;
    int S = 4;
    std::vector<int> ebound;   // shard g owns the experts [ebound[g], ebound[g+1])
    int nl_max = 0;
    std::vector<Shard> sh;
    Io* hio = nullptr;
    float* hlogits = nullptr;
    int64_t pos_next = 0;
    int n_cap = 2048, max_sel = 2048;
    bool graphs = std::getenv("HY4F_NOGRAPH") == nullptr;

    const strata::TensorInfo* find(const std::string& n, const strata::GgufFile** f) const {
        for (const auto& g : files)
            if (const strata::TensorInfo* t = g->find(n)) {
                *f = g.get();
                return t;
            }
        return nullptr;
    }
    uint64_t mu_(const std::string& k) const {
        const strata::MetaValue* v = files[0]->get("hy_v4_flash." + k);
        if (!v) throw std::runtime_error("hy4f: missing key " + k);
        return v->type == strata::MetaType::F32 || v->type == strata::MetaType::F64 ? (uint64_t) v->f : v->u;
    }
    double mf_(const std::string& k) const {
        const strata::MetaValue* v = files[0]->get("hy_v4_flash." + k);
        if (!v) throw std::runtime_error("hy4f: missing key " + k);
        return v->num();
    }
    template <class TT> std::vector<TT> ma_(const std::string& k, bool exact_u) const {
        strata::MetaValue v;
        if (!files[0]->get_full("hy_v4_flash." + k, v)) throw std::runtime_error("hy4f: missing key " + k);
        std::vector<TT> out;
        for (const auto& it : v.items)
            out.push_back(exact_u ? (TT) it.u : (TT) (it.type == strata::MetaType::F32 ? it.f : (double) it.u));
        return out;
    }
    void geometry() {
        E = (int) mu_("embedding_length");
        hc = (int) mu_("hyper_connection.count");
        L = (int) mu_("block_count") - (int) mu_("nextn_predict_layers");
        V = (int) mu_("vocab_size");
        q_lora = (int) mu_("attention.q_lora_rank");
        kvl = (int) mu_("attention.kv_lora_rank");
        rope = (int) mu_("rope.dimension_count");
        nope = (int) mu_("attention.key_length_mla") - rope;
        vh = (int) mu_("attention.value_length_mla");
        win = (int) mu_("attention.sliding_window");
        IH = (int) mu_("attention.indexer.head_count");
        IK = (int) mu_("attention.indexer.key_length");
        top_k = (int) mu_("attention.indexer.top_k");
        kpool = (int) mu_("attention.indexer.kpool");
        tail = (int) mu_("attention.indexer.kpool_select_tail");
        n_exp = (int) mu_("expert_count");
        n_used = (int) mu_("expert_used_count");
        n_ffx = (int) mu_("expert_feed_forward_length");
        latent = (int) mu_("expert_latent_length");
        dense_lead = (int) mu_("leading_dense_block_count");
        n_ff = (int) mu_("feed_forward_length");
        eps = (float) mf_("attention.layer_norm_rms_epsilon");
        hc_eps = (float) mf_("hyper_connection.epsilon");
        hc_mag = (float) mf_("hyper_connection.magnitude");
        w_scale = (float) mf_("expert_weights_scale");
        theta = (float) mf_("rope.freq_base");
        theta_swa = (float) mf_("rope.freq_base_swa");
        swa = ma_<int>("attention.sliding_window_pattern", false);
        const std::vector<int> heads = ma_<int>("attention.head_count", false);
        clampv = ma_<float>("swiglu_clamp_exp", false)[0];
        for (int il = 0; il < L; ++il) (swa[il] ? Hs : Hd) = heads[il];
        eg_layers = ma_<int>("ple.layers", false);
        eg_mult = ma_<uint64_t>("ple.layer_multipliers", true);
        eg_off = ma_<uint64_t>("ple.head_offsets", true);
        eg_vsz = ma_<uint64_t>("ple.head_vocab_sizes", true);
        eg_map = ma_<int64_t>("engram.token_map", true);
        eg_ngram = (int) mu_("ple.ngram_size");
        eg_heads = (int) mu_("ple.heads_per_ngram");
        eg_D = (int) mu_("embedding_length_per_layer_input");
        eg_pad = (int) mu_("engram.pad_token_id");
        n_cap = std::max(top_k, kpool * (top_k / kpool - tail) + kpool);
        max_sel = std::max(n_cap, win);
        if (Hd % S || Hs % S || n_exp % S || n_ffx % S || n_ff % S || V % S)
            throw std::runtime_error("hy4f: the geometry does not split over " + std::to_string(S) + " GPUs");
    }

    float* dalloc(size_t n) {
        float* p = nullptr;
        HY_CK(cudaMalloc(&p, n * sizeof(float)));
        HY_CK(cudaMemset(p, 0, n * sizeof(float)));
        return p;
    }
    // a weight's device memory with a zeroed tail: MMQ reads whole 256-value k iterations, past the last row's end
    void* walloc(size_t bytes) {
        void* p = nullptr;
        HY_CK(cudaMalloc(&p, bytes + 4096));
        HY_CK(cudaMemset(static_cast<uint8_t*>(p) + bytes, 0, 4096));
        return p;
    }
    const strata::TensorInfo* T(const std::string& n, const strata::GgufFile** f) const {
        const strata::TensorInfo* t = find(n, f);
        if (!t) throw std::runtime_error("hy4f: missing tensor " + n);
        return t;
    }
    DW up_rows(Shard& s, const std::string& n, int64_t r0, int64_t r1) {
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = T(n, &f);
        DW w;
        w.type = (int) t->type;
        for (size_t i = 0; i < 3 && i < t->shape.size(); ++i) w.ne[i] = (int64_t) t->shape[i];
        w.rb = row_bytes(w.type, w.ne[0]);
        const int64_t rows = w.ne[1] * w.ne[2];
        if (r1 < 0) r1 = rows;
        if (r0 < 0 || r1 > rows || r0 >= r1) throw std::runtime_error("hy4f: bad row slice of " + n);
        if (w.ne[2] > 1) w.ne[2] = (r1 - r0) / w.ne[1];
        else w.ne[1] = r1 - r0;
        const size_t bytes = w.rb * (size_t) (r1 - r0);
        w.d = walloc(bytes);
        HY_CK(cudaMemcpy(w.d, f->tensor_data(*t) + w.rb * (size_t) r0, bytes, cudaMemcpyHostToDevice));
        s.w[n] = w;
        return w;
    }
    DW up_full(Shard& s, const std::string& n) { return up_rows(s, n, 0, -1); }
    DW up_cols(Shard& s, const std::string& n, int64_t c0, int64_t c1) {
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = T(n, &f);
        const int type = (int) t->type, be = block_elems(type);
        const int64_t ne0 = (int64_t) t->shape[0], ne1 = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
        if (c0 % be || c1 % be) throw std::runtime_error("hy4f: column slice of " + n + " is not block aligned");
        const size_t rb = row_bytes(type, ne0), off = row_bytes(type, c0), nrb = row_bytes(type, c1 - c0);
        std::vector<uint8_t> h(nrb * (size_t) ne1);
        const uint8_t* src = f->tensor_data(*t);
        for (int64_t r = 0; r < ne1; ++r) std::memcpy(h.data() + nrb * r, src + rb * r + off, nrb);
        DW w;
        w.type = type;
        w.ne[0] = c1 - c0;
        w.ne[1] = ne1;
        w.rb = nrb;
        w.d = walloc(h.size());
        HY_CK(cudaMemcpy(w.d, h.data(), h.size(), cudaMemcpyHostToDevice));
        s.w[n] = w;
        return w;
    }

    // rows of several tensors (one type, one ne0) concatenated into one weight: pieces (name, r0, r1), r1 < 0 = all
    DW up_concat(Shard& s, const std::string& out, const std::vector<std::tuple<std::string, int64_t, int64_t>>& pieces) {
        int type = -1;
        int64_t ne0 = 0, rows = 0;
        std::vector<uint8_t> h;
        for (const auto& [n, r0, r1_] : pieces) {
            const strata::GgufFile* f = nullptr;
            const strata::TensorInfo* t = T(n, &f);
            const int64_t tr = (t->shape.size() > 1 ? (int64_t) t->shape[1] : 1) * (t->shape.size() > 2 ? (int64_t) t->shape[2] : 1);
            const int64_t r1 = r1_ < 0 ? tr : r1_;
            if (type < 0) type = (int) t->type, ne0 = (int64_t) t->shape[0];
            if ((int) t->type != type || (int64_t) t->shape[0] != ne0)
                throw std::runtime_error("hy4f: cannot concatenate " + n + " into " + out);
            const size_t rb = row_bytes(type, ne0);
            const uint8_t* src = f->tensor_data(*t) + rb * (size_t) r0;
            h.insert(h.end(), src, src + rb * (size_t) (r1 - r0));
            rows += r1 - r0;
        }
        DW w;
        w.type = type;
        w.ne[0] = ne0;
        w.ne[1] = rows;
        w.rb = row_bytes(type, ne0);
        w.d = walloc(h.size());
        HY_CK(cudaMemcpy(w.d, h.data(), h.size(), cudaMemcpyHostToDevice));
        s.w[out] = w;
        return w;
    }

    // the token embedding in pinned host memory, read by every GPU in place (a few rows per step: ~390 MB of VRAM saved)
    DW emb_h;
    DW emb_host() {
        if (emb_h.d) return emb_h;
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = T("token_embd.weight", &f);
        emb_h.type = (int) t->type;
        emb_h.ne[0] = (int64_t) t->shape[0];
        emb_h.ne[1] = (int64_t) t->shape[1];
        emb_h.rb = row_bytes(emb_h.type, emb_h.ne[0]);
        const size_t bytes = emb_h.rb * (size_t) emb_h.ne[1];
        HY_CK(cudaHostAlloc(&emb_h.d, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
        std::memcpy(emb_h.d, f->tensor_data(*t), bytes);
        return emb_h;
    }

    // the experts per GPU: HY4F_EXPERT_SPLIT=a,b,c,d, else evened out by the free VRAM (GPUs that hold other processes'
    // memory get fewer experts; one expert index = its gate/up/down in every MoE layer)
    void split_experts(const std::vector<int>& devs) {
        std::vector<int> n(S, n_exp / S);
        if (const char* e = std::getenv("HY4F_EXPERT_SPLIT")) {
            std::vector<int> v;
            for (const char* p = e; *p;) {
                v.push_back(std::atoi(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
            if ((int) v.size() == S) n = v;
        } else if (S > 1) {
            double per = 0;   // bytes of one expert index over the layers
            for (int il = dense_lead; il < L + (has_mtp ? 1 : 0); ++il) {
                const std::string P = "blk." + std::to_string(il) + ".";
                for (const char* k : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                    const strata::GgufFile* f = nullptr;
                    const strata::TensorInfo* t = T(P + k, &f);
                    per += (double) row_bytes((int) t->type, (int64_t) t->shape[0]) * (double) t->shape[1];
                }
            }
            std::vector<double> fr(S);
            double mean = 0;
            for (int g = 0; g < S; ++g) {
                size_t f = 0, tot = 0;
                HY_CK(cudaSetDevice(devs[g]));
                HY_CK(cudaMemGetInfo(&f, &tot));
                fr[g] = (double) f;
                mean += fr[g] / S;
            }
            int sum = 0;
            for (int g = 0; g < S; ++g) {
                n[g] = (int) std::lround(n_exp / (double) S + (fr[g] - mean) / per);
                sum += n[g];
            }
            n[S - 1] += n_exp - sum;
        }
        ebound.assign(S + 1, 0);
        for (int g = 0; g < S; ++g) ebound[g + 1] = ebound[g] + n[g];
        if (ebound[S] != n_exp) throw std::runtime_error("hy4f: the expert split does not add up to " + std::to_string(n_exp));
        nl_max = 0;
        for (int g = 0; g < S; ++g) nl_max = std::max(nl_max, n[g]);
        if (o.verbose) {
            std::fprintf(stderr, "hy4f: experts per GPU:");
            for (int g = 0; g < S; ++g) std::fprintf(stderr, " %d", n[g]);
            std::fprintf(stderr, "\n");
        }
    }

    void load_engram() {
        const strata::GgufFile* f = nullptr;
        const strata::TensorInfo* t = T("per_layer_token_embd.weight", &f);
        if ((int) t->type != T_F8) throw std::runtime_error("hy4f: Engram table is not F8_E4M3");
        const size_t bytes = (size_t) t->shape[0] * (size_t) t->shape[1];
        const uint8_t* src = f->tensor_data(*t);
        const uintptr_t pg = 4096, a0 = reinterpret_cast<uintptr_t>(src) & ~(pg - 1);
        const size_t len = ((reinterpret_cast<uintptr_t>(src) + bytes + pg - 1) & ~(pg - 1)) - a0;
        HY_CK(cudaHostRegister(reinterpret_cast<void*>(a0), len,
                               cudaHostRegisterMapped | cudaHostRegisterPortable | cudaHostRegisterReadOnly));
        eg_table = const_cast<uint8_t*>(src);
    }

    void load_shard(Shard& s) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking));
        const int g = s.g;
        const int64_t C = o.max_context;
        s.w["token_embd.weight"] = emb_host();
        for (const char* n : {"output_norm.weight", "output_hc_fn.weight", "output_hc_scale.weight", "output_hc_base.weight"})
            up_full(s, n);
        up_rows(s, "output.weight", (int64_t) g * V / S, (int64_t) (g + 1) * V / S);
        const int Hsl = Hs / S, Hdl = Hd / S, QH = nope + rope;
        for (int il = 0; il < L + (has_mtp ? 1 : 0); ++il) {
            const std::string P = "blk." + std::to_string(il) + ".";
            for (const char* n : {"attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                                  "post_ffw_norm.weight", "attn_sinks.weight"})
                up_full(s, P + n);
            if (il < L)
                for (const char* n : {"hc_attn_fn.weight", "hc_attn_scale.weight", "hc_attn_base.weight", "hc_ffn_fn.weight",
                                      "hc_ffn_scale.weight", "hc_ffn_base.weight"})
                    up_full(s, P + n);
            else
                for (const char* n : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.eh_proj.weight",
                                      "nextn.shared_head_norm.weight"})
                    up_full(s, P + n);
            for (int e : eg_layers)
                if (e == il)
                    for (const char* n : {"engram_lookup_norm.weight", "engram_kv.weight", "engram_kv_b.weight",
                                          "engram_gate.weight", "engram_hc_scale.weight", "engram_pre_gamma.weight"})
                        up_full(s, P + n);
            const bool sw = swa[il] != 0;
            const int Hl = sw ? Hsl : Hdl, h0 = g * Hl;
            // x -> [q_a | kv_a | attn_gate (own heads) | indexer.attn_k (DSA)]
            std::vector<std::tuple<std::string, int64_t, int64_t>> xpieces = {
                {P + "attn_q_a.weight", 0, -1}, {P + "attn_kv_a_mqa.weight", 0, -1},
                {P + "attn_gate.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh}};
            if (!sw) xpieces.push_back({P + "indexer.attn_k.weight", 0, -1});
            up_concat(s, P + "xproj", xpieces);
            if (sw) {
                up_concat(s, P + "qproj", {{P + "attn_q_b.weight", (int64_t) h0 * QH, (int64_t) (h0 + Hl) * QH}});
                up_rows(s, P + "attn_k_b.weight", (int64_t) h0 * kvl, (int64_t) (h0 + Hl) * kvl);
            } else {
                up_concat(s, P + "qproj", {{P + "attn_q_b.weight", 0, -1}, {P + "indexer.attn_q_b.weight", 0, -1}});
                up_full(s, P + "attn_k_b.weight");
                up_concat(s, P + "ivg", {{P + "indexer_compressor_value.weight", 0, -1},
                                         {P + "indexer_compressor_gate.weight", 0, -1}});
                for (const char* n : {"indexer.k_norm.weight", "indexer.proj.weight", "indexer_compressor_ape.weight",
                                      "indexer_compressor_norm.weight"})
                    up_full(s, P + n);
            }
            up_rows(s, P + "attn_v_b.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh);
            up_cols(s, P + "attn_output.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh);
            if (il < dense_lead) {
                const int64_t f0 = (int64_t) g * n_ff / S, f1 = (int64_t) (g + 1) * n_ff / S;
                up_concat(s, P + "dense_gu", {{P + "ffn_gate.weight", f0, f1}, {P + "ffn_up.weight", f0, f1}});
                up_cols(s, P + "ffn_down.weight", f0, f1);
            } else {
                for (const char* n : {"ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_routed_up.weight"}) up_full(s, P + n);
                const int64_t e0 = ebound[g], e1 = ebound[g + 1];
                {   // gate | up of each expert as one [2 n_ff, latent] matrix (MMQ run_glu's layout; one decode launch)
                    const strata::GgufFile *fg = nullptr, *fu = nullptr;
                    const strata::TensorInfo* tg = T(P + "ffn_gate_exps.weight", &fg);
                    const strata::TensorInfo* tu = T(P + "ffn_up_exps.weight", &fu);
                    if (tg->type != tu->type) throw std::runtime_error("hy4f: gate/up expert types differ in " + P);
                    const int64_t per = (int64_t) tg->shape[1];
                    const size_t rb = row_bytes((int) tg->type, (int64_t) tg->shape[0]), eb = rb * (size_t) per;
                    std::vector<uint8_t> h(2 * eb * (size_t) (e1 - e0));
                    for (int64_t e = e0; e < e1; ++e) {
                        std::memcpy(h.data() + 2 * eb * (size_t) (e - e0), fg->tensor_data(*tg) + eb * (size_t) e, eb);
                        std::memcpy(h.data() + 2 * eb * (size_t) (e - e0) + eb, fu->tensor_data(*tu) + eb * (size_t) e, eb);
                    }
                    DW w;
                    w.type = (int) tg->type;
                    w.ne[0] = (int64_t) tg->shape[0];
                    w.ne[1] = 2 * per;
                    w.ne[2] = e1 - e0;
                    w.rb = rb;
                    w.d = walloc(h.size());
                    HY_CK(cudaMemcpy(w.d, h.data(), h.size(), cudaMemcpyHostToDevice));
                    s.w[P + "ffn_gu_exps"] = w;
                    const strata::GgufFile* fd = nullptr;
                    const int64_t pd = (int64_t) T(P + "ffn_down_exps.weight", &fd)->shape[1];
                    up_rows(s, P + "ffn_down_exps.weight", e0 * pd, e1 * pd);
                }
                const int64_t f0 = (int64_t) g * n_ffx / S, f1 = (int64_t) (g + 1) * n_ffx / S;
                // x -> [routed_down | shared gate (own) | shared up (own)]
                up_concat(s, P + "xmoe", {{P + "ffn_routed_down.weight", 0, -1}, {P + "ffn_gate_shexp.weight", f0, f1},
                                          {P + "ffn_up_shexp.weight", f0, f1}});
                up_cols(s, P + "ffn_down_shexp.weight", f0, f1);
            }
            if (sw) {
                HY_CK(cudaMalloc(&s.kv[il], (size_t) (win + RING_PAD) * KV_RB));
                HY_CK(cudaMemset(s.kv[il], 0, (size_t) (win + RING_PAD) * KV_RB));
            } else {
                HY_CK(cudaMalloc(&s.kv[il], (size_t) (C / S + 1) * KV_RB));
                s.ring[il] = dalloc((size_t) kpool * 2 * IK);
                HY_CK(cudaMalloc(&s.pooled[il], sizeof(__half) * (size_t) (C / kpool + 1) * IK));
                HY_CK(cudaMemset(s.pooled[il], 0, sizeof(__half) * (size_t) (C / kpool + 1) * IK));
            }
        }
        const int Hmax = std::max(Hd, Hs);
        HY_CK(cudaMalloc(&s.q8, kn::native_q8_1_bytes(std::max({16384, n_ff, E * (hc + 1)}), 1)));
        HY_CK(cudaMalloc(&s.q8b, kn::native_q8_1_bytes(n_ffx * n_used, 1)));
        HY_CK(cudaMalloc(&s.dio, sizeof(Io)));
        HY_CK(cudaMalloc(&s.hist, sizeof(int) * C));
        s.X = dalloc((size_t) hc * E);
        s.R = dalloc((size_t) hc * E);
        s.cur = dalloc(E);
        s.x = dalloc(E);
        s.y = dalloc(E);
        s.post = dalloc(8);
        s.hcp = dalloc(8 * HC_SPLIT);
        s.hcs = dalloc(HC_SPLIT);
        s.qa = dalloc(q_lora);
        s.qr = dalloc(q_lora);
        s.q = dalloc((size_t) Hmax * QH);
        s.kvrow = dalloc(KV_ROW);
        s.qabs = dalloc((size_t) Hmax * kvl);
        s.o = dalloc((size_t) Hmax * kvl);
        s.a = dalloc((size_t) Hmax * vh);
        s.gate = dalloc((size_t) Hmax * vh);
        s.ao = dalloc(E);
        s.part = dalloc((size_t) Hd * (kvl + 2));
        s.sc = dalloc((size_t) Hmax * max_sel);
        s.iq = dalloc((size_t) IH * IK);
        s.ik = dalloc(IK);
        s.ikn = dalloc(IK);
        s.iv = dalloc(IK);
        s.ig = dalloc(IK);
        s.iw = dalloc(IH);
        s.score = dalloc((size_t) (C / kpool + 1));
        HY_CK(cudaMalloc(&s.cells, sizeof(int) * n_cap));
        s.lg = dalloc(n_exp);
        s.ew = dalloc(n_used);
        HY_CK(cudaMalloc(&s.eids, sizeof(int) * n_used));
        HY_CK(cudaMalloc(&s.eloc, sizeof(int) * n_used));
        s.xl = dalloc(latent);
        s.ga = dalloc(std::max(n_ff, n_ffx));
        s.up = dalloc(std::max(n_ff, n_ffx));
        s.moe = dalloc(latent);
        s.mu = dalloc(E);
        s.sh = dalloc(E);
        s.ffn = dalloc(E);
        s.ga16 = dalloc((size_t) n_used * 2 * n_ffx);
        s.up16 = dalloc((size_t) n_used * n_ffx);
        s.dn16 = dalloc((size_t) n_used * latent);
        auto qalloc = [&](int n) {
            Q81* p = nullptr;
            HY_CK(cudaMalloc(&p, sizeof(Q81) * (size_t) ((n + 31) / 32)));
            return p;
        };
        s.xp = dalloc((size_t) q_lora + KV_ROW + (size_t) Hmax / S * vh + IK);
        s.qp = dalloc((size_t) Hd * QH + (size_t) IH * IK + (size_t) Hs / S * QH);
        s.ivg = dalloc(2 * IK);
        s.xm = dalloc((size_t) latent + 2 * n_ffx / S);
        s.gu = dalloc(2 * (size_t) n_ff / S);
        s.kvdump = dalloc(KV_ROW);
        s.xq = qalloc(E);
        s.qrq = qalloc(q_lora);
        s.ikq = qalloc(IK);
        s.aq = qalloc(Hmax / S * vh);
        s.xlq = qalloc(latent);
        s.moeq = qalloc(latent);
        s.shq = qalloc(n_ffx / S);
        s.guq = qalloc(n_ff / S);
        s.q16 = qalloc(n_used * n_ffx);
        s.eg_r = dalloc((size_t) eg_D * (eg_ngram - 1) * eg_heads);
        s.eg_kv = dalloc((size_t) E * (hc + 1));
        HY_CK(cudaMalloc(&s.eg_ids, sizeof(int64_t) * (eg_ngram - 1) * eg_heads));
        auto up_u64 = [&](uint64_t*& d, const std::vector<uint64_t>& v) {
            HY_CK(cudaMalloc(&d, sizeof(uint64_t) * v.size()));
            HY_CK(cudaMemcpy(d, v.data(), sizeof(uint64_t) * v.size(), cudaMemcpyHostToDevice));
        };
        up_u64(s.eg_mult, eg_mult);
        up_u64(s.eg_vsz, eg_vsz);
        up_u64(s.eg_off, eg_off);
        HY_CK(cudaMalloc(&s.eg_map, sizeof(int64_t) * eg_map.size()));
        HY_CK(cudaMemcpy(s.eg_map, eg_map.data(), sizeof(int64_t) * eg_map.size(), cudaMemcpyHostToDevice));
        s.head_cur = dalloc(E);
        s.head_x = dalloc(E);
        s.logits = dalloc(V / S);
        for (int p = 0; p < 2; ++p) {   // receive areas: [src shard][payload]
            s.ar_buf[p] = dalloc((size_t) S * E);
            s.cb_buf[p] = dalloc((size_t) S * (Hd / S) * (kvl + 2));
        }
        HY_CK(cudaMalloc(&s.ar_flag, sizeof(int) * 4));
        HY_CK(cudaMemset(s.ar_flag, 0, sizeof(int) * 4));
        HY_CK(cudaMalloc(&s.cb_flag, sizeof(int) * 4));
        HY_CK(cudaMemset(s.cb_flag, 0, sizeof(int) * 4));
        HY_CK(cudaMalloc(&s.ar_ctr, sizeof(int)));
        HY_CK(cudaMemset(s.ar_ctr, 0, sizeof(int)));
        HY_CK(cudaMalloc(&s.cb_ctr, sizeof(int)));
        HY_CK(cudaMemset(s.cb_ctr, 0, sizeof(int)));
    }

    void gemv(Shard& s, const DW& w, size_t off, const float* x, float* y, int n_in, int n_out) {
        const uint8_t* d = static_cast<const uint8_t*>(w.d) + off;
        switch (w.type) {
        case T_Q8_0: kn::native_q8_0_f32(d, x, s.q8, y, n_in, n_out, 1, s.st); break;
        case T_Q3_K: kn::native_q3_k_f32(d, x, s.q8, y, n_in, n_out, 1, s.st); break;
        case T_Q4_K: kn::native_q4_k_f32(d, x, s.q8, y, n_in, n_out, 1, s.st); break;
        case T_Q6_K: kn::native_q6_k_f32(d, x, s.q8, y, n_in, n_out, 1, s.st); break;
        case T_Q4_0: kn::native_q4_0_f32(d, x, s.q8, y, n_in, n_out, 1, s.st); break;
        case T_BF16: kn::bf16_gemv_fp32_mmvf(x, reinterpret_cast<const uint16_t*>(d), y, n_in, n_out, s.st); break;
        case T_F32: k_f32_gemv<<<nb(n_out, 8), 256, 0, s.st>>>(reinterpret_cast<const float*>(d), x, y, n_in, n_out); break;
        default: throw std::runtime_error("hy4f: gemv type " + std::to_string(w.type));
        }
    }
    void gemv(Shard& s, const std::string& n, const float* x, float* y) {
        const DW& w = s.w.at(n);
        gemv(s, w, 0, x, y, (int) w.ne[0], (int) (w.ne[1] * w.ne[2]));
    }
    const float* f32(Shard& s, const std::string& n) {
        const DW& w = s.w.at(n);
        if (w.type != T_F32) throw std::runtime_error("hy4f: " + n + " is not F32");
        return static_cast<const float*>(w.d);
    }
    void hc_pre(Shard& s, const std::string& P, const std::string& k, int n_mix, float mag, float* cur) {
        k_hc_mix<<<dim3(n_mix, HC_SPLIT), 256, 0, s.st>>>(s.X, f32(s, P + k + "_fn.weight"), hc * E, s.hcp, s.hcs);
        k_hc_apply<<<nb(E), 256, 0, s.st>>>(s.X, s.hcp, s.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                             E, hc, n_mix, eps, hc_eps, mag, cur, s.post);
    }
    void allreduce(Shard& s, float* x) {
        if (S > 1) k_allreduce<<<1, 1024, 0, s.st>>>(x, E, s.ar_peers, s.g, S, s.ar_ctr);
    }

    void engram(Shard& s, int e, int il) {
        const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
        const std::string P = "blk." + std::to_string(il) + ".";
        k_engram_hash<<<1, 32, 0, s.st>>>(s.hist, s.dio, s.eg_map, (int64_t) eg_map.size(), eg_pad,
                                          s.eg_mult + (size_t) e * ng, s.eg_vsz + (size_t) e * ncol,
                                          s.eg_off + (size_t) e * ncol, ng, nh, s.eg_ids);
        k_engram_rows<<<ncol, 128, 0, s.st>>>(eg_table, s.eg_ids, f32(s, P + "engram_lookup_norm.weight"), s.eg_r, eg_D,
                                              1.1920929e-07f);
        gemv(s, P + "engram_kv.weight", s.eg_r, s.eg_kv);
        k_engram_mix<<<1, 1024, 0, s.st>>>(s.X, s.R, s.eg_kv, f32(s, P + "engram_kv_b.weight"), f32(s, P + "engram_gate.weight"),
                                           f32(s, P + "engram_hc_scale.weight"), f32(s, P + "engram_pre_gamma.weight"), E,
                                           hc, 1.1920929e-07f);
    }

    // the sigmoid router (kd_router for 512 experts, else Maya's router_sigmoid)
    void router(Shard& s, const float* lg, const float* bias, int T, int* ids, float* w) {
        if (n_exp == 512 && !std::getenv("HY4F_OLD_ROUTER"))
            kd_router<512><<<T, 512, 0, s.st>>>(lg, bias, n_used, w_scale, ids, w);
        else
            kn::router_sigmoid(lg, bias, T, n_exp, n_used, w_scale, true, ids, w, s.st);
    }
    void mmvq(Shard& s, const std::string& n, const Q81* xq, float* y) {
        const DW& w = s.w.at(n);
        if (w.type != T_Q8_0) throw std::runtime_error("hy4f: merged weight " + n + " is not Q8_0");
        kn::native_q8_0_mmvq(w.d, xq, y, (int) w.ne[0], (int) w.ne[1], 1, s.st);
    }
    void ar_post(Shard& s, const float* x1, const float* x2, const std::string& norm) {
        k_ar_post<<<1, 1024, 0, s.st>>>(x1, x2, E, s.ar_peers, s.g, S, s.ar_ctr, f32(s, norm), eps, s.X, s.R, s.post, hc,
                                        nullptr);
    }
    void norm_in(Shard& s, const std::string& P, const std::string& k, const std::string& norm, int n_mix, float mag) {
        k_hc_mix<<<dim3(n_mix, HC_SPLIT), 256, 0, s.st>>>(s.X, f32(s, P + k + "_fn.weight"), hc * E, s.hcp, s.hcs);
        k_hc_apply_norm<<<1, 1024, 0, s.st>>>(s.X, s.hcp, s.hcs, f32(s, P + k + "_scale.weight"),
                                              f32(s, P + k + "_base.weight"), f32(s, norm), E, hc, n_mix, eps, hc_eps,
                                              mag, s.x, s.xq, s.post);
    }

    void attention(Shard& s, int il) {
        const std::string P = "blk." + std::to_string(il) + ".";
        const bool sw = swa[il] != 0;
        const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
        const float th = sw ? theta_swa : theta;
        const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
        norm_in(s, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag);
        mmvq(s, P + "xproj", s.xq, s.xp);
        k_rms_q8<<<1, 1024, 0, s.st>>>(s.xp + oq, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, nullptr, s.qrq);
        k_kv_prep<<<1, 576, 0, s.st>>>(s.xp + okv, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th, s.kv[il],
                                       s.dio, sw ? 1 : 2, win + RING_PAD, S, s.g, nullptr);
        mmvq(s, P + "qproj", s.qrq, s.qp);
        const DW& kb = s.w.at(P + "attn_k_b.weight");
        const DW& vb = s.w.at(P + "attn_v_b.weight");
        const float scale = 1.0f / std::sqrt((float) QH);
        if (sw) {
            if (kb.type == T_Q8_0) k_absorb_q8<<<nb(Hl * kvl), 256, 0, s.st>>>((const uint8_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hl);
            else k_absorb_bf16<<<nb(Hl * kvl), 256, 0, s.st>>>((const uint16_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hl);
            k_attn2<<<Hl, 256, 0, s.st>>>(s.qabs, s.qp, QH, nope, th, s.kv[il], nullptr, 0, s.dio, win, 1, 0,
                                          f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, s.o, nullptr);
        } else {
            float* iq = s.qp + (size_t) Hd * QH;
            k_rope<<<nb(IH * rope / 2), 256, 0, s.st>>>(iq, IH, IK, IK - rope, rope, s.dio, th);
            k_rms_q8<<<1, 128, 0, s.st>>>(s.xp + oik, f32(s, P + "indexer.k_norm.weight"), IK, eps, nullptr, s.ikq);
            mmvq(s, P + "ivg", s.ikq, s.ivg);
            k_ring_pool<<<1, 128, 0, s.st>>>(s.ivg, s.ring[il], f32(s, P + "indexer_compressor_ape.weight"),
                                             f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK, kpool, rope,
                                             eps, th, s.dio);
            gemv(s, P + "indexer.proj.weight", s.x, s.iw);
            k_scale<<<1, 32, 0, s.st>>>(s.iw, IH, 1.0f / std::sqrt((float) (IK * IH)));
            k_score<<<nb((int) (o.max_context / kpool + 1) * 32), 256, 0, s.st>>>(iq, s.pooled[il], s.iw, IK, IH, kpool,
                                                                                    top_k, s.dio, s.score);
            k_select_kpool<1024><<<1, 1024, 0, s.st>>>(s.score, &s.dio->pos, kpool, top_k, top_k / kpool - tail, n_cap,
                                                       s.cells);
            if (kb.type == T_Q8_0) k_absorb_q8<<<nb(Hd * kvl), 256, 0, s.st>>>((const uint8_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hd);
            else k_absorb_bf16<<<nb(Hd * kvl), 256, 0, s.st>>>((const uint16_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hd);
            if (S > 1) {
                k_attn2<<<Hd, 256, 0, s.st>>>(s.qabs, s.qp, QH, nope, th, s.kv[il], s.cells, n_cap, s.dio, win, S, s.g,
                                              nullptr, scale, nullptr, s.part);
                k_attn_combine<<<1, 1024, 0, s.st>>>(s.part, Hd, kvl, s.g * Hl, Hl, f32(s, P + "attn_sinks.weight"),
                                                     s.cb_peers, s.g, S, s.cb_ctr, s.o);
            } else {
                k_attn2<<<Hd, 256, 0, s.st>>>(s.qabs, s.qp, QH, nope, th, s.kv[il], s.cells, n_cap, s.dio, win, 1, 0,
                                              f32(s, P + "attn_sinks.weight"), scale, s.o, nullptr);
            }
        }
        if (vb.type == T_Q8_0)
            k_vb_gate_q8<<<nb(Hl * vh * 32), 256, 0, s.st>>>((const uint8_t*) vb.d, s.o, s.xp + og, s.a, kvl, vh, Hl);
        else
            k_vb_gate_bf16<<<nb(Hl * vh * 32), 256, 0, s.st>>>((const uint16_t*) vb.d, s.o, s.xp + og, s.a, kvl, vh, Hl);
        k_q8<<<nb(Hl * vh), 256, 0, s.st>>>(s.a, Hl * vh, s.aq);
        mmvq(s, P + "attn_output.weight", s.aq, s.ao);
        ar_post(s, s.ao, nullptr, P + "post_attention_norm.weight");
    }

    void ffn(Shard& s, int il) {
        const std::string P = "blk." + std::to_string(il) + ".";
        norm_in(s, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag);
        if (il < dense_lead) {
            const int fl = n_ff / S;
            mmvq(s, P + "dense_gu", s.xq, s.gu);
            k_swiglu_q8<<<nb(fl), 256, 0, s.st>>>(s.gu, s.gu + fl, fl, INFINITY, s.guq, nullptr);
            mmvq(s, P + "ffn_down.weight", s.guq, s.ffn);
            ar_post(s, s.ffn, nullptr, P + "post_ffw_norm.weight");
            return;
        }
        gemv(s, P + "ffn_gate_inp.weight", s.x, s.lg);
        router(s, s.lg, f32(s, P + "exp_probs_b.bias"), 1, s.eids, s.ew);
        const int e0 = ebound[s.g], e1 = ebound[s.g + 1], fl = n_ffx / S;
        k_local_ids<<<1, 32, 0, s.st>>>(s.eids, s.eloc, n_used, e0, e1);
        mmvq(s, P + "xmoe", s.xq, s.xm);
        k_q8<<<nb(latent), 256, 0, s.st>>>(s.xm, latent, s.xlq);
        auto ids_mmvq = [&](const DW& w, const void* xq, bool shared, float* y, int n_in, int n_out) {
            const size_t eb = w.rb * (size_t) w.ne[1];
            if (w.type == T_Q3_K) kn::native_q3_k_mmvq_ids_loop(w.d, eb, s.eloc, n_used, xq, shared, y, n_in, n_out, s.st);
            else if (w.type == T_Q4_K) kn::native_q4_k_mmvq_ids_loop(w.d, eb, s.eloc, n_used, xq, shared, y, n_in, n_out, s.st);
            else throw std::runtime_error("hy4f: expert type " + std::to_string(w.type));
        };
        ids_mmvq(s.w.at(P + "ffn_gu_exps"), s.xlq, true, s.ga16, latent, 2 * n_ffx);   // [n_used][gate | up]
        k_swiglu_gu_q8<<<nb(n_used * n_ffx), 256, 0, s.st>>>(s.ga16, n_ffx, n_used * n_ffx, clampv, s.q16);
        ids_mmvq(s.w.at(P + "ffn_down_exps.weight"), s.q16, false, s.dn16, n_ffx, latent);
        k_moe_sum<<<nb(latent), 256, 0, s.st>>>(s.moe, s.dn16, s.ew, s.eloc, n_used, latent);
        k_q8<<<nb(latent), 256, 0, s.st>>>(s.moe, latent, s.moeq);
        mmvq(s, P + "ffn_routed_up.weight", s.moeq, s.mu);
        k_swiglu_q8<<<nb(fl), 256, 0, s.st>>>(s.xm + latent, s.xm + latent + fl, fl, INFINITY, s.shq, nullptr);
        mmvq(s, P + "ffn_down_shexp.weight", s.shq, s.sh);
        ar_post(s, s.mu, s.sh, P + "post_ffw_norm.weight");
    }

    void layer(Shard& s, int il) {
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) engram(s, (int) e, il);
        attention(s, il);
        ffn(s, il);
    }

    void step_ops(Shard& s) {
        HY_CK(cudaMemcpyAsync(s.dio, hio, sizeof(Io), cudaMemcpyHostToDevice, s.st));
        const DW& emb = s.w.at("token_embd.weight");
        k_embed<<<nb(E), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, s.dio, s.X, s.R, E, hc, s.hist);
        for (int il = 0; il < L; ++il) layer(s, il);
        norm_in(s, "", "output_hc", "output_norm.weight", hc, 0.0f);
        gemv(s, "output.weight", s.x, s.logits);
        HY_CK(cudaMemcpyAsync(hlogits + (size_t) s.g * (V / S), s.logits, sizeof(float) * (V / S), cudaMemcpyDeviceToHost,
                              s.st));
    }

    void setup_peers() {
        for (auto& a : sh) {
            HY_CK(cudaSetDevice(a.dev));
            for (auto& b : sh)
                if (a.dev != b.dev) {
                    int can = 0;
                    HY_CK(cudaDeviceCanAccessPeer(&can, a.dev, b.dev));
                    if (!can) throw std::runtime_error("hy4f: no P2P between GPUs " + std::to_string(a.dev) + " and " +
                                                       std::to_string(b.dev));
                    const cudaError_t e = cudaDeviceEnablePeerAccess(b.dev, 0);
                    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) HY_CK(e);
                    cudaGetLastError();
                }
        }
        Peers ar{}, cb{};
        for (int j = 0; j < S; ++j) {
            ar.buf[j][0] = sh[j].ar_buf[0];
            ar.buf[j][1] = sh[j].ar_buf[1];
            ar.flag[j] = sh[j].ar_flag;
            cb.buf[j][0] = sh[j].cb_buf[0];
            cb.buf[j][1] = sh[j].cb_buf[1];
            cb.flag[j] = sh[j].cb_flag;
        }
        for (auto& s : sh) s.ar_peers = ar, s.cb_peers = cb;
    }

    // persistent launcher threads, one per shard (spawning per step costs more than the launch work it hides)
    struct Pool {
        std::vector<std::thread> th;
        std::mutex m;
        std::condition_variable cv, done_cv;
        std::function<void(Shard&)> job;
        uint64_t gen = 0;
        int done = 0;
        bool quit = false;
        std::vector<std::string> errs;
    };
    std::unique_ptr<Pool> pool;
    void start_pool() {
        pool.reset(new Pool);
        pool->errs.resize(sh.size());
        for (size_t i = 0; i < sh.size(); ++i)
            pool->th.emplace_back([this, i] {
                uint64_t seen = 0;
                cudaSetDevice(sh[i].dev);
                for (;;) {
                    std::function<void(Shard&)> job;
                    {
                        std::unique_lock<std::mutex> lk(pool->m);
                        pool->cv.wait(lk, [&] { return pool->quit || pool->gen != seen; });
                        if (pool->quit) return;
                        seen = pool->gen;
                        job = pool->job;
                    }
                    try {
                        job(sh[i]);
                    } catch (const std::exception& e) {
                        pool->errs[i] = e.what();
                    }
                    std::lock_guard<std::mutex> lk(pool->m);
                    if (++pool->done == (int) sh.size()) pool->done_cv.notify_one();
                }
            });
    }
    void launch_parallel(const std::function<void(Shard&)>& job) {
        if (!pool) start_pool();
        {
            std::lock_guard<std::mutex> lk(pool->m);
            pool->job = job;
            pool->done = 0;
            for (auto& e : pool->errs) e.clear();
            ++pool->gen;
        }
        pool->cv.notify_all();
        std::unique_lock<std::mutex> lk(pool->m);
        pool->done_cv.wait(lk, [&] { return pool->done == (int) sh.size(); });
        for (const auto& e : pool->errs)
            if (!e.empty()) throw std::runtime_error(e);
    }
    ~Impl() {
        if (pool) {
            {
                std::lock_guard<std::mutex> lk(pool->m);
                pool->quit = true;
            }
            pool->cv.notify_all();
            for (auto& t : pool->th) t.join();
        }
    }

#include "hy4f_batch.inc"
#include "hy4f_dec.inc"

    void capture() {
        for (auto& s : sh) {
            HY_CK(cudaSetDevice(s.dev));
            cudaGraph_t gr = nullptr;
            HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
            step_ops(s);
            HY_CK(cudaStreamEndCapture(s.st, &gr));
            HY_CK(cudaGraphInstantiate(&s.exec, gr, 0));
            HY_CK(cudaGraphDestroy(gr));
        }
    }
};

Model::Model() : d_(new Impl) {}
Model::~Model() = default;
int Model::vocab() const { return d_->V; }

std::string Model::summary() const {
    return "hy4f v3: " + std::to_string(d_->L) + " layers, tensor parallel over " + std::to_string(d_->S) +
           " GPUs, Q8 KV (DSA context parallel, SWA ring), " + (d_->graphs ? "CUDA graphs" : "no graphs") +
           ", max context " + std::to_string(d_->o.max_context) + (d_->has_mtp ? ", MTP" : "");
}

bool Model::load(const Options& o, std::string& err) {
    try {
        Impl& d = *d_;
        d.o = o;
        for (const auto& s : o.shards) d.files.emplace_back(new strata::GgufFile(s));
        const strata::MetaValue* a = d.files[0]->get("general.architecture");
        if (!a || a->s != "hy_v4_flash") throw std::runtime_error("hy4f: not a hy_v4_flash GGUF");
        int ndev = 0;
        HY_CK(cudaGetDeviceCount(&ndev));
        std::vector<int> devs = o.gpus;
        if (devs.empty())
            for (int i = 0; i < ndev; ++i) devs.push_back(i);
        d.S = (int) devs.size();
        if (d.S != 4 && d.S != 2 && d.S != 1) throw std::runtime_error("hy4f: 1, 2 or 4 GPUs");
        d.geometry();
        d.load_engram();
        HY_CK(cudaHostAlloc((void**) &d.hio, sizeof(Io), cudaHostAllocPortable));
        HY_CK(cudaHostAlloc((void**) &d.hlogits, sizeof(float) * d.V, cudaHostAllocPortable));
        {
            const strata::GgufFile* f = nullptr;
            const strata::TensorInfo* t = d.find("blk." + std::to_string(d.L) + ".nextn.eh_proj.weight", &f);
            const strata::TensorInfo* te = d.find("blk." + std::to_string(d.L) + ".ffn_gate_exps.weight", &f);
            d.has_mtp = t && te && (int) te->type != 30 /*BF16 experts: requantize first (scripts/requant_mtp_q3k.py)*/ &&
                        (int) t->type == 8 && !std::getenv("HY4F_NO_MTP");
        }
        d.split_experts(devs);
        d.sh.resize(d.S);
        for (int g = 0; g < d.S; ++g) {
            d.sh[g].g = g;
            d.sh[g].dev = devs[g];
            d.load_shard(d.sh[g]);
        }
        if (d.S > 1) d.setup_peers();
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaDeviceSynchronize());
        }
        if (d.graphs && std::getenv("HY4F_OLD_DECODE")) d.capture();
        if (d.graphs) d.dec_setup();
        if (o.verbose)
            for (auto& s : d.sh) {
                size_t fr = 0, tot = 0;
                HY_CK(cudaSetDevice(s.dev));
                HY_CK(cudaMemGetInfo(&fr, &tot));
                std::fprintf(stderr, "hy4f: GPU %d free %.2f GiB\n", s.dev, fr / 1073741824.0);
            }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Model::prefill(const int32_t* toks, int n, int64_t pos, float* logits, int chunk, std::string& err) {
    try {
        Impl& d = *d_;
        if (pos > d.pos_next) throw std::runtime_error("hy4f: prefill out of order");
        if (pos + n > d.o.max_context) throw std::runtime_error("hy4f: context full");
        if (chunk <= 0) chunk = 1024;
        if (d.bsh.empty() || d.bsh[0].Tmax < chunk) {
            if (!d.bsh.empty()) throw std::runtime_error("hy4f: prefill chunk grew");
            d.batch_setup(chunk);
        }
        const auto t_start = std::chrono::steady_clock::now();
        for (int c0 = 0; c0 < n; c0 += chunk) {
            const int T = std::min(chunk, n - c0);
            const bool last = c0 + T == n;
            const int p0 = (int) (pos + c0);
            d.launch_parallel([&](Shard& s) {
                Impl::BShard& b = d.bsh[s.g];
                HY_CK(cudaMemcpyAsync(b.toks, toks + c0, sizeof(int) * T, cudaMemcpyHostToDevice, s.st));
                d.b_chunk(s, b, p0, T, last && logits);
            });
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaStreamSynchronize(s.st));
        }
        HY_CK(cudaGetLastError());
        d.pos_next = pos + n;
        if (d.o.verbose)
            std::fprintf(stderr, "hy4f: prefill %d tokens: %.1f ms (%.1f tok/s, setup excluded)\n", n,
                         1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(),
                         n / std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count());
        if (logits) {
            std::memcpy(logits, d.hlogits, sizeof(float) * d.V);
            if (!d.o.dump_dir.empty()) {
                std::FILE* f = std::fopen((d.o.dump_dir + "/logits.f32").c_str(), "ab");
                if (f) std::fwrite(logits, 4, d.V, f), std::fclose(f);
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Model::verify(const int32_t* toks, int T, int64_t pos, float* logits, int32_t* argmax, std::string& err) {
    try {
        Impl& d = *d_;
        if (d.dsh.empty()) throw std::runtime_error("hy4f: multi-token steps need the CUDA graphs");
        if (T < 1 || T > Impl::DT) throw std::runtime_error("hy4f: verify takes 1.." + std::to_string(Impl::DT) + " tokens");
        if (pos > d.pos_next) throw std::runtime_error("hy4f: verify out of order");
        if (pos + T > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_v->pos = (int) pos;
        d.hio_v->n = T;
        for (int t = 0; t < T; ++t) d.hio_v->toks[t] = toks[t];
        d.dec_launch(false, T);
        d.pos_next = pos + T;
        if (logits) std::memcpy(logits, d.hlogT, sizeof(float) * T * d.V);
        if (argmax)
            for (int t = 0; t < T; ++t) {
                float bv = -INFINITY;
                int bi = 0;
                for (int g = 0; g < d.S; ++g) {
                    const float v = d.hargv[g * Impl::DT + t];
                    if (v > bv) bv = v, bi = d.hargi[g * Impl::DT + t];
                }
                argmax[t] = bi;
            }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Model::mtp(const int32_t* next_toks, int T, int64_t mpos, int32_t* draft, std::string& err) {
    try {
        Impl& d = *d_;
        if (!d.has_mtp || d.dsh.empty()) throw std::runtime_error("hy4f: no MTP layer (or no CUDA graphs)");
        if (T < 1 || T > Impl::DT) throw std::runtime_error("hy4f: mtp takes 1.." + std::to_string(Impl::DT) + " tokens");
        if (mpos + T > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_m->pos = (int) mpos;
        d.hio_m->n = T;
        for (int t = 0; t < T; ++t) d.hio_m->toks[t] = next_toks[t];
        d.dec_launch(true, T);
        d.m_next = mpos + T;
        float bv = -INFINITY;
        int bi = 0;
        for (int g = 0; g < d.S; ++g)
            if (d.hmargv[g] > bv) bv = d.hmargv[g], bi = d.hmargi[g];
        if (draft) *draft = bi;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
bool Model::mtp_chain(int32_t tok, int64_t mpos, int32_t* draft, std::string& err) {
    try {
        Impl& d = *d_;
        if (!d.has_mtp || d.dsh.empty()) throw std::runtime_error("hy4f: no MTP layer (or no CUDA graphs)");
        if (mpos + 1 > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_m->pos = (int) mpos;
        d.hio_m->n = 1;
        d.hio_m->toks[0] = tok;
        d.dec_launch(true, 1, true);
        float bv = -INFINITY;
        int bi = 0;
        for (int g = 0; g < d.S; ++g)
            if (d.hmargv[g] > bv) bv = d.hmargv[g], bi = d.hmargi[g];
        if (draft) *draft = bi;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
bool Model::has_mtp() const { return d_->has_mtp; }

bool Model::step(int32_t token, int64_t pos, float* logits, std::string& err) {
    if (!d_->dsh.empty() && !std::getenv("HY4F_OLD_DECODE")) {
        if (!verify(&token, 1, pos, logits, nullptr, err)) return false;
        if (logits && !d_->o.dump_dir.empty()) {
            std::FILE* f = std::fopen((d_->o.dump_dir + "/logits.f32").c_str(), "ab");
            if (f) std::fwrite(logits, 4, d_->V, f), std::fclose(f);
        }
        return true;
    }
    try {
        Impl& d = *d_;
        if (pos > d.pos_next) throw std::runtime_error("hy4f: step out of order");
        if (pos >= d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio->pos = (int) pos;
        d.hio->tok = token;
        if (d.graphs) {
            // the shards' graphs launched from one host thread each: a 2500-node launch costs ~4 ms of CPU, and the
            // first exchange of a step would otherwise wait for the last shard's launch
            d.launch_parallel([&](Shard& s) { HY_CK(cudaGraphLaunch(s.exec, s.st)); });
        } else {
            // one host thread per shard: a shard's ~2000 launches would fill its queue behind the first exchange and
            // block a single enqueuing thread before the peers got theirs
            d.launch_parallel([&](Shard& s) { d.step_ops(s); });
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaStreamSynchronize(s.st));
        }
        HY_CK(cudaGetLastError());
        d.pos_next = pos + 1;
        if (logits) std::memcpy(logits, d.hlogits, sizeof(float) * d.V);
        if (!d.o.dump_dir.empty() && logits) {
            std::FILE* f = std::fopen((d.o.dump_dir + "/logits.f32").c_str(), "ab");
            if (f) std::fwrite(logits, 4, d.V, f), std::fclose(f);
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// serve support (maya-hy4f --serve: hy4f_serve.cpp) - the cache kept across requests.
//
// What a step reads from earlier positions: the DSA layers' KV (context parallel, row pos / S on shard pos % S), the
// indexer's pooled groups (row g = pos / kpool, written when the group's last position is read), the token history
// (Engram) - all indexed by position, so dropping the positions from P on needs nothing but the counter (a verify may
// start below it anyway: a rejected draft's rows are written over).  Two rings alias: the SWA layers' KV keeps
// position p in row p % R (R = win + RING_PAD) and a step at p reads p - win + 1 .. p; the indexer keeps the current
// group's positions in row p % kpool.  A row a kept position needs is lost when a later position sharing it was written
// after it - conservatively, any written since the last reset (hw_: their high-water mark, rejected drafts included).
// So P is kept when
//   SWA:     hw_ <= R + max(0, P - win + 1)  (no position >= P aliases a row of P - win + 1 .. P - 1), and
//   indexer: P % kpool == 0 (the group's rows are not read) or nothing was written past P's group (hw_ <= its end);
// mark() copies the rings at a position Pm (after the prompt): a rewind to Pm or Pm - 1 restores them.  Any rewind below
// Pm drops the copy after use (position Pm - 1 may be written anew).  Else, when only the indexer's rule fails, P goes
// down to its group's start (< kpool tokens read again) if the SWA rule holds there; else 0.
// The MTP layer (blk.L, a DSA layer at MTP positions = trunk positions - 1 for the prompt's pairs) has its own KV,
// pooled groups and indexer ring: its ring is copied and restored with the trunk's; elsewhere it is best effort - a
// stale MTP row only costs draft acceptance, never a token (every draft is verified by the trunk).
struct Model::Snap {
    int64_t pos = -1;                                   // the position mark() copied at (-1: none)
    std::vector<std::map<int, void*>> rows;             // per shard: layer -> its ring's copy
};

int64_t Model::position() const { return d_->pos_next; }
int64_t Model::max_context() const { return d_->o.max_context; }

void Model::reset() {
    std::string err;
    rewind(0, err);
}

namespace {
// the ring a layer keeps (SWA: its KV; DSA, the MTP layer included: the indexer's) and its bytes
std::pair<void*, size_t> serve_ring(Shard& s, int il, bool sw, int win, int kpool, int IK) {
    if (sw) return {(void*) s.kv[il], (size_t) (win + RING_PAD) * KV_RB};
    return {(void*) s.ring[il], (size_t) kpool * 2 * IK * sizeof(float)};
}
}  // namespace

bool Model::mark(std::string& err) {
    try {
        Impl& d = *d_;
        const int Lm = d.L + (d.has_mtp ? 1 : 0);
        if (!snap_) {
            snap_ = std::make_shared<Snap>();
            snap_->rows.resize(d.S);
            for (auto& s : d.sh) {
                HY_CK(cudaSetDevice(s.dev));
                for (int il = 0; il < Lm; ++il) {
                    void* p = nullptr;
                    HY_CK(cudaMalloc(&p, serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK).second));
                    snap_->rows[s.g][il] = p;
                }
            }
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            for (int il = 0; il < Lm; ++il) {
                const auto r = serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK);
                HY_CK(cudaMemcpyAsync(snap_->rows[s.g][il], r.first, r.second, cudaMemcpyDeviceToDevice, s.st));
            }
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaStreamSynchronize(s.st));
        }
        snap_->pos = d.pos_next;
        return true;
    } catch (const std::exception& e) {
        if (snap_) snap_->pos = -1;
        err = e.what();
        return false;
    }
}

int64_t Model::rewind(int64_t pos, std::string& err) {
    try {
        Impl& d = *d_;
        const int Lm = d.L + (d.has_mtp ? 1 : 0);
        hw_ = std::max(hw_, d.pos_next);                // (the positions written since the last rewind)
        pos = std::max<int64_t>(0, std::min(pos, d.pos_next));
        bool has_swa = false, has_dsa = false;
        for (int il = 0; il < d.L; ++il) (d.swa[il] ? has_swa : has_dsa) = true;
        const int64_t R = d.win + RING_PAD;
        auto keeps = [&](int64_t p, int64_t hw) {
            const bool swa_ok = !has_swa || hw <= R + std::max<int64_t>(0, p - d.win + 1);
            const bool idx_ok = !has_dsa || p % d.kpool == 0 || hw <= (p / d.kpool + 1) * d.kpool;
            return swa_ok && idx_ok;
        };
        const int64_t m = snap_ ? snap_->pos : -1;
        if (pos > 0 && !keeps(pos, hw_)) {
            if (m > 0 && (pos == m || pos == m - 1) && keeps(pos, m)) {
                for (auto& s : d.sh) {
                    HY_CK(cudaSetDevice(s.dev));
                    for (int il = 0; il < Lm; ++il) {
                        const auto r = serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK);
                        HY_CK(cudaMemcpyAsync(r.first, snap_->rows[s.g][il], r.second, cudaMemcpyDeviceToDevice, s.st));
                    }
                }
                for (auto& s : d.sh) {
                    HY_CK(cudaSetDevice(s.dev));
                    HY_CK(cudaStreamSynchronize(s.st));
                }
                hw_ = m;                                // (the rings as they were at m)
            } else if (const int64_t down = has_dsa ? pos - pos % d.kpool : 0; down > 0 && keeps(down, hw_)) {
                pos = down;                             // (the indexer group's start: < kpool tokens read again)
            } else {
                pos = 0;
            }
        }
        if (snap_ && pos < m) snap_->pos = -1;          // (positions before the copy's are written anew)
        if (pos == 0) hw_ = 0;
        d.pos_next = pos;
        return pos;
    } catch (const std::exception& e) {
        err = e.what();
        return -1;
    }
}

}  // namespace strata::hy4f
