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
#include "hy4f_index_packed.cuh"
#include "hy4f_score_select.cuh"
#include "hy4f_swa_prefill.cuh"
#include "hy4f_dsa_prefill.cuh"
#include "hy4f_dkern.cuh"
#include "hy4f_head_cert.cuh"
#include "hy4f_ranked.cuh"
#include "hy4f_project.cuh"
#include "hy4f_q3packed.cuh"
#include "hy4f_expert_q8.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/prefill/moe_mmq.hpp"
#ifdef HY4F_VISION
#include "hy4f_vision.hpp"
#endif

#include <cublas_v2.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <immintrin.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <memory>

namespace strata::hy4f {

namespace kn = strata::kernels;
using namespace kern;

namespace {

#define HY_CK(x)                                                                                                        \
    do {                                                                                                                \
        cudaError_t e_ = (x);                                                                                           \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));               \
    } while (0)

enum : int { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q3_K = 11, T_Q4_K = 12, T_Q6_K = 14, T_IQ2_XXS = 16,
             T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ3_S = 21, T_IQ2_S = 22, T_BF16 = 30, T_F8 = 43 };
// the expert types of the tuned Hy4 kernels; any other one runs through Maya's native grouped expert kernels
inline bool tuned_expert_type(int t) { return t == T_Q3_K || t == T_Q4_K || t == T_Q8_0; }

// Internal device layout, never a GGUF tensor type. Eight successive
// K-iterations for each warp lane form one aligned vector.
constexpr int T_ROUTER_PACKED = -2;

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
    case T_IQ2_XXS: return (size_t) n / 256 * 66;
    case T_IQ2_XS: return (size_t) n / 256 * 74;
    case T_IQ2_S: return (size_t) n / 256 * 82;
    case T_IQ3_XXS: return (size_t) n / 256 * 98;
    case T_IQ3_S: return (size_t) n / 256 * 110;
    default: throw std::runtime_error("hy4f: unsupported tensor type " + std::to_string(type));
    }
}
int block_elems(int type) {
    switch (type) {
    case T_Q4_0:
    case T_Q8_0: return 32;
    case T_Q3_K:
    case T_Q4_K:
    case T_Q6_K:
    case T_IQ2_XXS:
    case T_IQ2_XS:
    case T_IQ2_S:
    case T_IQ3_XXS:
    case T_IQ3_S: return 256;
    default: return 1;
    }
}
int nb(int n, int t = 256) { return (n + t - 1) / t; }

struct DW {
    int type = -1;
    int64_t ne[3] = {1, 1, 1};
    void* d = nullptr;
    size_t rb = 0;
    bool packed_projection = false;
    size_t stride = 0;   // experts: the bytes from one expert to the next (0: rb * ne[1])
};
inline size_t expert_stride(const DW& w) { return w.stride ? w.stride : w.rb * (size_t) w.ne[1]; }

// expert offload: one MoE layer on one shard.  Every expert of the shard is a [gate | up | down] record in pinned
// host memory; the n_res hottest also have a VRAM copy.  Both stores lie on one lattice of `stride` bytes, so a
// kernel finds an expert at base + index * stride whichever store holds it.
struct OffLayer {
    int n = 0, n_res = 0;
    int av = 0, ah = 0;   // lattice index = VRAM slot + av, or host record + ah
    size_t stride = 0, gu_bytes = 0;
    uint8_t *vram = nullptr, *host = nullptr;
    std::vector<int> home;   // local expert -> its host record (the adaptation swaps experts without moving records)
    std::vector<int> scratch_index;
    int* device_scratch_index = nullptr;
    uintptr_t record_base = 0;
    int* vmap = nullptr;     // device: local expert -> lattice index
    bool base_type = false;  // the records are of a tuned type (Q3_K / Q4_K / Q8_0)
    // A hybrid layer (HY4F_HOT_VRAM_MB): locals [0, n_hot) keep the base quantization in VRAM (`vram`, `stride`; no
    // host copy - a promotion reads the GGUF); everyone has a cold record (HY4F_EXPERT_GGUF's quantization) in host
    // memory, locals [n_hot, n_res) also in VRAM.  No lattice: the cold kernels take record addresses.
    bool hybrid = false;
    int n_hot = 0;
    int cold_gu = -1, cold_dn = -1;
    size_t cstride = 0, cgu_bytes = 0;
    uint8_t *cvram = nullptr, *chost = nullptr;
    std::vector<int> chome;               // local expert -> its cold host record
    unsigned long long* cptr = nullptr;   // device: local expert -> the address of its cold record
};

struct Shard {
    int g = 0, dev = 0;
    cudaStream_t st = nullptr;
    std::map<std::string, DW> w;
    std::map<int, uint8_t*> kv;       // layer -> Q8 rows (SWA ring / DSA context-parallel slots)
    std::map<int, uint8_t*> prefix_kv;  // dense DSA prefix replicated per device
    uint8_t* full_store = nullptr;
    uint8_t* full_mirror[2]{};
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
    int *eids = nullptr, *eloc = nullptr, *evirt = nullptr;
    std::map<int, OffLayer> off;
    std::vector<int> off_pending;   // MoE layers whose experts load after everything else (the auto budget)
    unsigned* rstat = nullptr;                 // HY4F_ROUTE_STATS: decode picks [layer][local expert]
    std::vector<uint64_t> pf_rows, pf_touch;   // ... and the prefill's rows / chunks that touched the expert
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
    std::vector<std::unique_ptr<strata::GgufFile>> draft_files;
    std::map<std::string, std::pair<const strata::GgufFile*, const strata::TensorInfo*>> draft_tensors;
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
    std::vector<int> mtp_ebound;
    const std::vector<int>& expert_bounds(int layer) const { return layer == L ? mtp_ebound : ebound; }
    int nl_max = 0;
    bool offload = false;                        // HY4F_EXPERT_VRAM_MB: part of the experts stays in host memory
    size_t off_budget = 0, off_stage = 0;
    std::map<int, std::vector<int>> off_perm;    // layer -> [expert id in the engine] = the GGUF's expert
    std::vector<std::map<int, int>> off_res;     // [shard][layer] -> experts resident in VRAM
    std::vector<std::map<int, int>> off_hot;     // ... of them in the hybrid's hot tier
    bool hybrid = false;                         // HY4F_HOT_VRAM_MB + HY4F_EXPERT_GGUF: two quantizations of the experts
    size_t hot_budget = 0;
    // HY4F_EXPERT_VRAM_MB=auto: the budget of a shard is its free VRAM once everything else is loaded, less
    // HY4F_VRAM_RESERVE_MB (2048: the prefill buffers of a 4096-token chunk, the staging buffers, the decode
    // buffers and graphs, a margin; a 2048-token chunk needs about 500 less)
    bool off_auto = false, off_profile = false;
    std::map<int, std::vector<double>> profile_seed;   // the decode picks HY4F_EXPERT_PROFILE gave [layer][GGUF expert]
    size_t off_reserve = 0;
    struct OffItem {
        double p;
        int il, l;
    };
    std::vector<std::vector<OffItem>> off_items;   // [shard]: its experts by heat
    std::map<int, size_t> off_stride, off_cstride;
    std::map<std::string, std::pair<const strata::GgufFile*, const strata::TensorInfo*>> cold_tensors;
    uint64_t st_hot = 0;
    std::string route_stats;
    // HY4F_EXPERT_ADAPT: the VRAM set follows the routing - a cold resident and a hot host expert trade places
    bool scratch_cache = false, scratch_active = false;
    bool adapt = false;
    double adapt_horizon = 1024;                   // tokens after which an old pick weighs 1/e
    double adapt_pf_weight = 0;                    // (a prompt's routing does not predict the answer's: measured)
    int adapt_every = 8, adapt_step_swaps = 1024, adapt_pf_swaps = 2048, adapt_calls = 0;
    uint64_t win_picks = 0, win_hits = 0, win_hot = 0;
    double adapt_margin = 3.0;   // picks a host expert must lead a resident by (a swap costs about as much as ten misses)
    std::vector<uint8_t> swap_rows;
    uint8_t* swap_pinned = nullptr;   // one base record on its way into a hot slot
    size_t swap_pinned_bytes = 0;
    std::map<int, std::vector<double>> off_score;  // layer -> [expert id in the engine] decayed picks
    std::vector<uint64_t> cum_dec, cum_rows, cum_touch;   // statistics [layer][GGUF expert]
    uint64_t st_picks = 0, st_hits = 0, st_pf = 0, st_pf_hit = 0, st_swaps = 0;
    double st_swap_s = 0;
    std::vector<Shard> sh;
    Io* hio = nullptr;
    float* hlogits = nullptr;
    int64_t pos_next = 0;
    int n_cap = 2048, max_sel = 2048;
    bool graphs = std::getenv("HY4F_NOGRAPH") == nullptr;
#ifdef HY4F_VISION
    std::unique_ptr<vision::Encoder> venc;
#endif

    const strata::TensorInfo* find(const std::string& n, const strata::GgufFile** f) const {
        auto draft = draft_tensors.find(n);
        if (draft != draft_tensors.end()) {
            *f = draft->second.first;
            return draft->second.second;
        }
        for (const auto& g : files)
            if (const strata::TensorInfo* t = g->find(n)) {
                *f = g.get();
                return t;
            }
        return nullptr;
    }
    bool replicate_prefix() const {
        return S == 4 && Hd == 32 && graphs && std::getenv("HY4F_SHORT_GRAPH") &&
               std::getenv("HY4F_REPLICATE_PREFIX") && !std::getenv("HY4F_OLD_DECODE") &&
               !std::getenv("HY4F_DEC_ATTN2");
    }
    int sparse_limit = 0, sparse_base = 0, sparse_pitch = 0, sparse_skip = 0, sparse_high = 0;
    bool sparse_valid = true;
    bool full_ready = false, full_disabled = false;
    int full_count = 0, full_layers[2]{}, full_prefix[2]{};
    const int full_requested = [] {
        const char* opt=std::getenv("HY4F_FULL_DSA_LAYERS");
        return opt && (std::strcmp(opt,"1")==0 || std::strcmp(opt,"2")==0) ? opt[0]-'0' : 0;
    }();
    int full_request_end = -1;
    bool full_batch_writes = false, full_prefill_warmed = false;
    void setup_sparse_replica() {
        const char* limit = std::getenv("HY4F_SPARSE_REPLICA");
        if (!limit || !replicate_prefix() || !direct_dsa || kpool != 4) return;
        const int capacity = (int)(o.max_context / S);
        const int requested = std::atoi(limit);
        // Beyond the ordinary capacity, use the existing dense prefix for the
        // earliest nonowner rows. Each of the three foreign stripes shrinks.
        const int skip = requested > capacity && top_k % S == 0 ? top_k / S : 0;
        const int rows = std::min(capacity + (S-1) * skip, requested) & ~3;
        if (rows <= top_k) return;
        sparse_limit = rows;
        sparse_skip = skip;
        sparse_pitch = rows/S - skip;
        sparse_base = capacity + 1 - (S-1) * sparse_pitch - skip;
    }
    void load_draft_experts(const char* path, bool down_only) {
        if (!path || !*path) return;
        auto file = std::make_unique<strata::GgufFile>(path);
        int gate_type = -1;
        for (const char* part : {"gate", "up", "down"}) {
            if (down_only && std::strcmp(part, "down")) continue;
            const std::string name = "blk." + std::to_string(L) + ".ffn_" + part + "_exps.weight";
            const strata::GgufFile* base = nullptr;
            const auto* original = find(name, &base);
            const auto* replacement = file->find(name);
            if (!original || !replacement || original->shape != replacement->shape ||
                (replacement->type != T_Q3_K && replacement->type != T_Q4_K && replacement->type != T_Q8_0))
                throw std::runtime_error("hy4f: incompatible draft expert tensor " + name);
            if (!std::strcmp(part, "gate")) gate_type = (int) replacement->type;
            if (!std::strcmp(part, "up") && (int) replacement->type != gate_type)
                throw std::runtime_error("hy4f: incompatible draft gate/up types");
            draft_tensors[name] = {file.get(), replacement};
        }
        draft_files.emplace_back(std::move(file));
    }
    // HY4F_EXPERT_GGUF: the trunk's routed experts from another GGUF of the same model (another quantization; the
    // attention, head, Engram and MTP stay the base files')
    void load_trunk_experts(const char* path) {
        if (!path || !*path) return;
        auto file = std::make_unique<strata::GgufFile>(path);
        for (int il = dense_lead; il < L; ++il)
            for (const char* part : {"gate", "up", "down"}) {
                const std::string name = "blk." + std::to_string(il) + ".ffn_" + part + "_exps.weight";
                const strata::GgufFile* base = nullptr;
                const auto* original = find(name, &base);
                const auto* replacement = file->find(name);
                if (!original || !replacement || original->shape != replacement->shape)
                    throw std::runtime_error("hy4f: incompatible expert tensor " + name + " in " + path);
                // (with HY4F_HOT_VRAM_MB the base experts stay as the hot tier and these are the cold one)
                if (std::getenv("HY4F_HOT_VRAM_MB")) cold_tensors[name] = {file.get(), replacement};
                else draft_tensors[name] = {file.get(), replacement};
            }
        draft_files.emplace_back(std::move(file));
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
    DW up_rows(Shard& s, const std::string& n, int64_t r0, int64_t r1, bool pack = false) {
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
        const uint8_t* src = f->tensor_data(*t) + w.rb * (size_t)r0;
        if (pack && w.type == T_Q8_0) {
            const auto packed = pack_projection(src, (int)w.ne[0], (int)(r1 - r0));
            HY_CK(cudaMemcpy(w.d, packed.data(), bytes, cudaMemcpyHostToDevice));
            w.packed_projection = true;
        } else HY_CK(cudaMemcpy(w.d, src, bytes, cudaMemcpyHostToDevice));
        s.w[n] = w;
        return w;
    }
    DW up_full(Shard& s, const std::string& n) { return up_rows(s, n, 0, -1); }
    void up_draft_head(Shard& s, int format, bool certificate) {
        const strata::GgufFile* file = nullptr;
        const auto* tensor = T("output.weight", &file);
        if (tensor->type != T_BF16 || tensor->shape.size() != 2 ||
            tensor->shape[0] != (size_t) E || tensor->shape[1] != (size_t) V)
            throw std::runtime_error("hy4f: draft head quantization requires a BF16 output.weight");
        const int rows = V / S;
        const auto type = format == T_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K;
        if (E % ggml_blck_size(type))
            throw std::runtime_error("hy4f: incompatible draft head input width");
        auto& weight = s.w[certificate ? "cert.output.weight" : "mtp.output.weight"];
        weight.type = format;
        weight.ne[0] = E;
        weight.ne[1] = rows;
        weight.rb = ggml_row_size(type, E);
        weight.d = walloc(rows * weight.rb);
        float2* device_norms = nullptr;
        if (certificate) {
            auto& norms = s.w["cert.output.norms"];
            norms.d = walloc((size_t)rows*sizeof(float2));
            device_norms = static_cast<float2*>(norms.d);
        }
        constexpr int chunk = 128;
        std::vector<float> input((size_t) chunk * E);
        std::vector<uint8_t> packed(chunk * weight.rb);
        const auto* bf16 = reinterpret_cast<const ggml_bf16_t*>(file->tensor_data(*tensor)) + (size_t) s.g * rows * E;
        for (int i = 0; i < rows; i += chunk) {
            const int n = std::min(rows-i, chunk);
            ggml_bf16_to_fp32_row(bf16 + (size_t) i * E, input.data(), (int64_t) n * E);
            bool valid[chunk] = {};
            if (certificate) {
                for (int j=0;j<n;++j) {
                    float* row=input.data()+(size_t)j*E;
                    valid[j]=std::all_of(row,row+E,[](float v){return std::isfinite(v) && std::fabs(v)<=0x1p16f;});
                    // A row outside the proof domain is always evaluated in BF16.
                    if (!valid[j]) std::fill(row,row+E,0.0f);
                }
            }
            const auto bytes = ggml_quantize_chunk(type, input.data(), packed.data(), 0, n, E, nullptr);
            if (bytes != n * weight.rb) throw std::runtime_error("hy4f: incomplete draft head quantization");
            if (certificate) {
                float2 norms[chunk];
                for (int j=0;j<n;++j) {
                    double residual=0, quant=0;
                    const float* row=input.data()+(size_t)j*E;
                    const uint8_t* block=packed.data()+j*weight.rb;
                    for (int k=0;k<E;k+=32,block+=34) {
                        ggml_fp16_t h; std::memcpy(&h,block,sizeof(h));
                        const double scale=ggml_fp16_to_fp32(h);
                        for (int l=0;l<32;++l) {
                            const double wq=scale*reinterpret_cast<const int8_t*>(block+2)[l];
                            const double dx=double(row[k+l])-wq;
                            residual+=dx*dx; quant+=wq*wq;
                        }
                    }
                    // Positive double sums of E terms. Cover rounding of the difference,
                    // squares, sum and sqrt, then round the float storage upwards.
                    const auto norm=[&](double ss) {
                        return valid[j] ? std::nextafter(float(std::sqrt(ss)*(1.0+0x1p-38)),INFINITY) : INFINITY;
                    };
                    norms[j]=make_float2(norm(residual),norm(quant));
                }
                HY_CK(cudaMemcpy(device_norms+i,norms,(size_t)n*sizeof(float2),cudaMemcpyHostToDevice));
            }
            HY_CK(cudaMemcpy(static_cast<uint8_t*>(weight.d) + i * weight.rb, packed.data(), bytes, cudaMemcpyHostToDevice));
        }
    }
    void up_router(Shard& s, const std::string& name, const std::vector<int>* perm = nullptr) {
        const strata::GgufFile* file = nullptr;
        const auto* tensor = T(name, &file);
        const uint8_t* data = file->tensor_data(*tensor);
        std::vector<uint8_t> reordered;   // the expert offload renumbers the experts: the router's rows follow
        if (perm) {
            if (tensor->type != T_F32 || tensor->shape.size() != 2 || tensor->shape[1] != perm->size())
                throw std::runtime_error("hy4f: cannot reorder " + name);
            const size_t rb = (size_t) tensor->shape[0] * sizeof(float);
            reordered.resize(rb * perm->size());
            for (size_t r = 0; r < perm->size(); ++r) std::memcpy(reordered.data() + rb * r, data + rb * (*perm)[r], rb);
            data = reordered.data();
        }
        if (std::getenv("HY4F_PACK_ROUTER") && tensor->type == T_F32 && tensor->shape.size() == 2 &&
            tensor->shape[0] % 256 == 0) {
            const auto* src = reinterpret_cast<const uint32_t*>(data);
            size_t count = size_t(tensor->shape[0]) * tensor->shape[1];
            if (std::all_of(src, src + count, [](uint32_t bits) { return (bits & 65535) == 0; })) {
                std::vector<uint16_t> packed(count);
                for (size_t i = 0; i < count; ++i)
                    packed[(i & ~size_t(255)) + (i & 31) * 8 + ((i >> 5) & 7)] = uint16_t(src[i] >> 16);
                DW w;
                w.type = T_ROUTER_PACKED;
                w.ne[0] = tensor->shape[0]; w.ne[1] = tensor->shape[1];
                w.rb = size_t(w.ne[0]) * sizeof(uint16_t);
                w.d = walloc(packed.size() * sizeof(uint16_t));
                HY_CK(cudaMemcpy(w.d, packed.data(), packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
                s.w[name] = w;
                return;
            }
        }
        if (!perm) {
            up_full(s, name);
            return;
        }
        DW w;
        w.type = T_F32;
        w.ne[0] = (int64_t) tensor->shape[0];
        w.ne[1] = (int64_t) tensor->shape[1];
        w.rb = row_bytes(T_F32, w.ne[0]);
        w.d = walloc(reordered.size());
        HY_CK(cudaMemcpy(w.d, reordered.data(), reordered.size(), cudaMemcpyHostToDevice));
        s.w[name] = w;
    }
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
        mtp_ebound = ebound;
        if (const char* split = std::getenv("HY4F_MTP_EXPERT_SPLIT")) {
            const char* cursor = split;
            for (int g = 0; g < S; ++g) {
                char* end = nullptr;
                long count = std::strtol(cursor, &end, 10);
                if (end == cursor || count <= 0 || count > n_exp || (g+1 < S ? *end != ',' : *end != 0))
                    throw std::runtime_error("hy4f: invalid MTP expert split");
                mtp_ebound[g+1] = mtp_ebound[g] + (int)count;
                nl_max = std::max(nl_max, (int)count);
                if (*end) cursor = end+1;
            }
            if (mtp_ebound.back() != n_exp) throw std::runtime_error("hy4f: invalid MTP expert count");
        }
        if (o.verbose) {
            std::fprintf(stderr, "hy4f: experts per GPU:");
            for (int g = 0; g < S; ++g) std::fprintf(stderr, " %d", n[g]);
            std::fprintf(stderr, "\n");
        }
        plan_offload();
    }

    // the routed experts of a layer as one record per expert: [gate | up | down] padded to whole blocks of both types
    struct ExpertSrc {
        const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
        int gu_type = -1, dn_type = -1;
        int64_t ne0 = 0, per = 0, ne0d = 0, perd = 0;
        size_t rb = 0, rbd = 0, eb = 0, ebd = 0, gu_bytes = 0, stride = 0;
    };
    bool hybrid_layer(int il) const {
        return hybrid && il < L && cold_tensors.count("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
    }
    ExpertSrc expert_src(int il, bool cold) const {
        const std::string P = "blk." + std::to_string(il) + ".";
        std::pair<const strata::GgufFile*, const strata::TensorInfo*> g, u, d;
        if (cold) {
            g = cold_tensors.at(P + "ffn_gate_exps.weight");
            u = cold_tensors.at(P + "ffn_up_exps.weight");
            d = cold_tensors.at(P + "ffn_down_exps.weight");
        } else {
            g.second = T(P + "ffn_gate_exps.weight", &g.first);
            u.second = T(P + "ffn_up_exps.weight", &u.first);
            d.second = T(P + "ffn_down_exps.weight", &d.first);
        }
        if (g.second->type != u.second->type) throw std::runtime_error("hy4f: gate/up expert types differ in " + P);
        ExpertSrc e;
        e.gate = g.first->tensor_data(*g.second);
        e.up = u.first->tensor_data(*u.second);
        e.down = d.first->tensor_data(*d.second);
        e.gu_type = (int) g.second->type;
        e.dn_type = (int) d.second->type;
        e.ne0 = (int64_t) g.second->shape[0], e.per = (int64_t) g.second->shape[1];
        e.ne0d = (int64_t) d.second->shape[0], e.perd = (int64_t) d.second->shape[1];
        e.rb = row_bytes(e.gu_type, e.ne0), e.eb = e.rb * (size_t) e.per;
        e.rbd = row_bytes(e.dn_type, e.ne0d), e.ebd = e.rbd * (size_t) e.perd;
        e.gu_bytes = 2 * e.eb;
        const size_t bg = row_bytes(e.gu_type, block_elems(e.gu_type)), bd = row_bytes(e.dn_type, block_elems(e.dn_type));
        const size_t unit = bg / std::gcd(bg, bd) * bd;
        e.stride = (e.gu_bytes + e.ebd + unit - 1) / unit * unit;
        return e;
    }
    static void fill_record(const ExpertSrc& e, size_t expert, uint8_t* rec) {
        std::memcpy(rec, e.gate + e.eb * expert, e.eb);
        std::memcpy(rec + e.eb, e.up + e.eb * expert, e.eb);
        std::memcpy(rec + 2 * e.eb, e.down + e.ebd * expert, e.ebd);
        std::memset(rec + 2 * e.eb + e.ebd, 0, e.stride - 2 * e.eb - e.ebd);
    }

    // HY4F_EXPERT_VRAM_MB=<MiB per GPU>: the expert offload.  Per MoE layer the experts are renumbered by heat (the
    // decode picks of HY4F_EXPERT_PROFILE, a HY4F_ROUTE_STATS file; the router rows move with them) and dealt
    // round-robin to the shards; each shard keeps its hottest ones in VRAM up to the budget.  The others stay in
    // pinned host memory: the decode kernels read them in place, the prefill fetches them into a staging buffer.
    // With HY4F_HOT_VRAM_MB and HY4F_EXPERT_GGUF the layers are hybrid: the hottest experts in the base quantization
    // up to that part of the budget, the next ones and the host ones in HY4F_EXPERT_GGUF's.
    void plan_offload() {
        if (const char* p = std::getenv("HY4F_ROUTE_STATS")) route_stats = p;
        const char* mb = std::getenv("HY4F_EXPERT_VRAM_MB");
        if (!mb) {
            if (!cold_tensors.empty()) throw std::runtime_error("hy4f: HY4F_HOT_VRAM_MB needs HY4F_EXPERT_VRAM_MB");
            return;
        }
        if (std::getenv("HY4F_PACK_Q3")) throw std::runtime_error("hy4f: HY4F_PACK_Q3 does not combine with the expert offload");
        offload = true;
        off_auto = !std::strcmp(mb, "auto");
        off_budget = off_auto ? 0 : (size_t) std::atoll(mb) << 20;
        const char* rs = std::getenv("HY4F_VRAM_RESERVE_MB");
        off_reserve = (size_t) (rs ? std::atoll(rs) : 2048) << 20;
        hybrid = !cold_tensors.empty();
        if (hybrid) hot_budget = (size_t) std::atoll(std::getenv("HY4F_HOT_VRAM_MB")) << 20;
        const char* sm = std::getenv("HY4F_OFFLOAD_STAGE_MB");
        off_stage = (size_t) (sm ? std::atoll(sm) : 256) << 20;
        if (const char* a = std::getenv("HY4F_EXPERT_ADAPT")) adapt = std::atoi(a) != 0;
        if (const char* a = std::getenv("HY4F_ADAPT_HORIZON")) adapt_horizon = std::max(1.0, std::atof(a));
        if (const char* a = std::getenv("HY4F_ADAPT_PF_WEIGHT")) adapt_pf_weight = std::atof(a);
        if (const char* a = std::getenv("HY4F_ADAPT_EVERY")) adapt_every = std::max(1, std::atoi(a));
        if (const char* a = std::getenv("HY4F_ADAPT_MARGIN")) adapt_margin = std::atof(a);
        if (const char* a = std::getenv("HY4F_ADAPT_STEP_SWAPS")) adapt_step_swaps = std::atoi(a);
        if (const char* a = std::getenv("HY4F_ADAPT_PF_SWAPS")) adapt_pf_swaps = std::atoi(a);
        const int nl = n_exp / S, last = L + (has_mtp ? 1 : 0);
        for (int g = 0; g <= S; ++g) ebound[g] = g * nl;
        mtp_ebound = ebound;
        nl_max = nl;
        std::map<int, std::vector<double>> heat;
        if (const char* pf = std::getenv("HY4F_EXPERT_PROFILE")) {
            std::FILE* f = std::fopen(pf, "r");
            if (!f) throw std::runtime_error(std::string("hy4f: cannot read ") + pf);
            char kind = 0;
            int il = 0, e = 0;
            double c = 0;
            while (std::fscanf(f, " %c %d %d %lf", &kind, &il, &e, &c) == 4)
                if (kind == 'D' && il >= dense_lead && il < last && e >= 0 && e < n_exp) {
                    auto& v = heat[il];
                    if (v.empty()) v.assign(n_exp, 0.0);
                    v[e] += c;
                }
            std::fclose(f);
        }
        using Item = OffItem;
        off_items.assign(S, {});
        auto& items = off_items;
        auto& stride = off_stride;
        auto& cstride = off_cstride;
        off_profile = !heat.empty();
        profile_seed = heat;
        off_res.assign(S, {});
        off_hot.assign(S, {});
        for (int il = dense_lead; il < last; ++il) {
            std::vector<double> c(n_exp, 1.0);
            if (heat.count(il)) c = heat[il];
            double sum = 0;
            for (double v : c) sum += v;
            std::vector<int> order(n_exp);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c[a] > c[b]; });
            auto& perm = off_perm[il];
            perm.resize(n_exp);
            auto& score = off_score[il];
            score.assign(n_exp, 0.0);
            for (int g = 0; g < S; ++g) {
                off_res[g][il] = 0;
                off_hot[g][il] = 0;
                for (int l = 0; l < nl; ++l) {
                    perm[g * nl + l] = order[l * S + g];
                    // the profile counts as 256 tokens of history
                    if (heat.count(il) && sum > 0) score[g * nl + l] = c[order[l * S + g]] / sum * n_used * 256.0;
                    items[g].push_back({sum > 0 ? c[order[l * S + g]] / sum : 0.0, il, l});
                }
            }
            stride[il] = expert_src(il, false).stride;
            if (hybrid_layer(il)) cstride[il] = expert_src(il, true).stride;
        }
        for (int g = 0; g < S; ++g) {
            std::stable_sort(items[g].begin(), items[g].end(), [](const Item& a, const Item& b) {
                return a.p != b.p ? a.p > b.p : a.l != b.l ? a.l < b.l : a.il < b.il;
            });
            if (!off_auto) plan_residency(g, off_budget);
        }
    }

    // shard g's VRAM residents within `budget` bytes: the hybrid's hot tier first, then the next hottest experts
    void plan_residency(int g, size_t budget) {
        const auto& items = off_items[g];
        auto& stride = off_stride;
        auto& cstride = off_cstride;
        size_t used = 0;
        std::map<int, bool> closed_hot, closed;
        double hit = 0, hot = 0, all = 0;
        int kept = 0, kept_hot = 0;
        if (hybrid)
            for (const OffItem& it : items) {
                if (!hybrid_layer(it.il) || closed_hot[it.il]) continue;
                if (used + stride[it.il] + 4096 > std::min(hot_budget, budget)) {
                    closed_hot[it.il] = true;
                    continue;
                }
                used += stride[it.il];
                ++off_hot[g][it.il];
                ++off_res[g][it.il];
                ++kept_hot;
                hot += it.p;
            }
        const size_t hot_used = used;
        for (const OffItem& it : items) {
            if (it.il < L) all += it.p;
            const bool hyb = hybrid_layer(it.il);
            if ((hyb && it.l < off_hot[g][it.il]) || closed[it.il]) continue;
            const size_t cost = hyb ? cstride[it.il] : stride[it.il];
            if (used + cost + 4096 > budget) {
                closed[it.il] = true;
                continue;
            }
            used += cost;
            ++off_res[g][it.il];
            ++kept;
            if (it.il < L) hit += it.p;
        }
        if (o.verbose) {
            if (hybrid)
                std::fprintf(stderr, "hy4f: offload: shard %d keeps %d hot (%.2f GiB) + %d cold (%.2f GiB) of %zu experts in VRAM, "
                                     "expected picks hot %.3f, in VRAM %.3f%s\n",
                             g, kept_hot, hot_used / 1073741824.0, kept, (used - hot_used) / 1073741824.0, items.size(),
                             all > 0 ? hot / all : 0.0, all > 0 ? (hot + hit) / all : 0.0, !off_profile ? " (no profile)" : "");
            else
                std::fprintf(stderr, "hy4f: offload: shard %d keeps %d of %zu experts in VRAM (%.2f GiB), expected hits %.3f%s\n",
                             g, kept, items.size(), used / 1073741824.0, all > 0 ? hit / all : 0.0,
                             !off_profile ? " (no profile)" : "");
        }
    }

    // The expert host store: anonymous memory (HY4F_HOST_HUGE=1: on transparent huge pages), registered with CUDA once
    // it is filled.  (Huge pages are off by default: with the RAM full of page cache the kernel compacts for minutes.)
    static uint8_t* host_store_alloc(size_t bytes) {
        const size_t huge = size_t(2) << 20, len = (bytes + huge - 1) / huge * huge + huge;
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("hy4f: mmap of the expert host store failed");
        uint8_t* a = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p) + huge - 1) / huge * huge);
        const char* e = std::getenv("HY4F_HOST_HUGE");
        if (e && std::atoi(e)) madvise(a, len - (size_t) (a - static_cast<uint8_t*>(p)), MADV_HUGEPAGE);
        return a;
    }
    static void host_store_register(uint8_t* p, size_t bytes) {
        HY_CK(cudaHostRegister(p, (bytes + 4095) / 4096 * 4096, cudaHostRegisterPortable | cudaHostRegisterMapped));
    }

    void load_offload_experts(Shard& s, int il, const std::string& P) {
        const ExpertSrc base = expert_src(il, false);
        OffLayer ol;
        ol.stride = base.stride;
        ol.gu_bytes = base.gu_bytes;
        ol.n = n_exp / S;
        ol.n_res = off_res[s.g].at(il);
        ol.hybrid = hybrid_layer(il);
        ol.n_hot = ol.hybrid ? off_hot[s.g].at(il) : 0;
        ol.base_type = tuned_expert_type(base.gu_type) && tuned_expert_type(base.dn_type);
        const size_t stride = ol.stride;
        const auto& perm = off_perm.at(il);
        // the records of the shard's locals [l0, l1) of `e` into dst
        auto fill_all = [&](const ExpertSrc& e, uint8_t* dst, int l0, int l1) {
            const int workers = 8;
            std::vector<std::thread> th;
            for (int w = 0; w < workers; ++w)
                th.emplace_back([&, w] {
                    for (int l = l0 + w; l < l1; l += workers)
                        fill_record(e, (size_t) perm[(size_t) s.g * ol.n + l], dst + (size_t) (l - l0) * e.stride);
                });
            for (auto& t : th) t.join();
        };
        uint8_t* base_ptr = nullptr;
        if (ol.hybrid) {
            if (ol.n_hot) {
                ol.vram = static_cast<uint8_t*>(walloc((size_t) ol.n_hot * stride));
                std::vector<uint8_t> h((size_t) ol.n_hot * stride);
                fill_all(base, h.data(), 0, ol.n_hot);
                HY_CK(cudaMemcpy(ol.vram, h.data(), h.size(), cudaMemcpyHostToDevice));
            }
            base_ptr = ol.vram;
            const ExpertSrc cold = expert_src(il, true);
            ol.cold_gu = cold.gu_type, ol.cold_dn = cold.dn_type;
            ol.cstride = cold.stride, ol.cgu_bytes = cold.gu_bytes;
            const size_t host_bytes = (size_t) ol.n * cold.stride + 4096;
            ol.chost = host_store_alloc(host_bytes);
            std::memset(ol.chost + (size_t) ol.n * cold.stride, 0, 4096);
            fill_all(cold, ol.chost, 0, ol.n);
            host_store_register(ol.chost, host_bytes);
            const int nb = ol.n_res - ol.n_hot;
            if (nb) {
                ol.cvram = static_cast<uint8_t*>(walloc((size_t) nb * cold.stride));
                HY_CK(cudaMemcpy(ol.cvram, ol.chost + (size_t) ol.n_hot * cold.stride, (size_t) nb * cold.stride,
                                 cudaMemcpyHostToDevice));
            }
            ol.chome.resize(ol.n);
            std::iota(ol.chome.begin(), ol.chome.end(), 0);
            std::vector<unsigned long long> ptr(ol.n);
            for (int l = 0; l < ol.n; ++l)
                ptr[l] = (unsigned long long) (l >= ol.n_hot && l < ol.n_res ? ol.cvram + (size_t) (l - ol.n_hot) * cold.stride
                                                                              : ol.chost + (size_t) l * cold.stride);
            HY_CK(cudaMalloc(&ol.cptr, sizeof(unsigned long long) * ol.n));
            HY_CK(cudaMemcpy(ol.cptr, ptr.data(), sizeof(unsigned long long) * ol.n, cudaMemcpyHostToDevice));
        } else {
            if (ol.n_res) ol.vram = static_cast<uint8_t*>(walloc((size_t) ol.n_res * stride));
            const size_t host_bytes = (size_t) ol.n * stride + stride + 4096;
            uint8_t* block = host_store_alloc(host_bytes);
            size_t shift = 0;
            if (ol.vram)
                shift = (reinterpret_cast<uintptr_t>(ol.vram) % stride + stride - reinterpret_cast<uintptr_t>(block) % stride) % stride;
            ol.host = block + shift;
            std::memset(ol.host + (size_t) ol.n * stride, 0, host_bytes - shift - (size_t) ol.n * stride);
            fill_all(base, ol.host, 0, ol.n);
            host_store_register(block, host_bytes);
            if (ol.n_res) HY_CK(cudaMemcpy(ol.vram, ol.host, (size_t) ol.n_res * stride, cudaMemcpyHostToDevice));
            uintptr_t origin = reinterpret_cast<uintptr_t>(ol.host);
            if (ol.vram) origin = std::min(origin, reinterpret_cast<uintptr_t>(ol.vram));
            if (scratch_cache) {
                const uintptr_t buffer = reinterpret_cast<uintptr_t>(bsh.front().dbounds);
                const size_t pad = (origin % stride + stride - buffer % stride) % stride;
                origin = std::min(origin, buffer + pad);
                ol.scratch_index.assign(ol.n, -1);
                HY_CK(cudaMalloc(&ol.device_scratch_index, ol.n * sizeof(int)));
                HY_CK(cudaMemset(ol.device_scratch_index, 0xff, ol.n * sizeof(int)));
            }
            ol.record_base = origin;
            base_ptr = reinterpret_cast<uint8_t*>(origin);
            const auto av = ol.vram ? (reinterpret_cast<uintptr_t>(ol.vram) - origin) / stride : 0;
            const auto ah = (reinterpret_cast<uintptr_t>(ol.host) - origin) / stride;
            if (av + ol.n >= INT_MAX || ah + ol.n >= INT_MAX) throw std::runtime_error("hy4f: the expert stores are too far apart");
            ol.av = (int) av;
            ol.ah = (int) ah;
            ol.home.resize(ol.n);
            std::iota(ol.home.begin(), ol.home.end(), 0);
            std::vector<int> vm(ol.n);
            for (int l = 0; l < ol.n; ++l) vm[l] = l < ol.n_res ? l + ol.av : l + ol.ah;
            HY_CK(cudaMalloc(&ol.vmap, sizeof(int) * ol.n));
            HY_CK(cudaMemcpy(ol.vmap, vm.data(), sizeof(int) * ol.n, cudaMemcpyHostToDevice));
        }
        DW w;
        w.type = base.gu_type;
        w.ne[0] = base.ne0;
        w.ne[1] = 2 * base.per;
        w.ne[2] = ol.n;
        w.rb = base.rb;
        w.d = base_ptr;
        w.stride = stride;
        s.w[P + "ffn_gu_exps"] = w;
        DW wd;
        wd.type = base.dn_type;
        wd.ne[0] = base.ne0d;
        wd.ne[1] = base.perd;
        wd.ne[2] = ol.n;
        wd.rb = base.rbd;
        wd.d = base_ptr ? base_ptr + ol.gu_bytes : nullptr;
        wd.stride = stride;
        s.w[P + "ffn_down_exps.weight"] = wd;
        s.off[il] = ol;
    }

    // a one-dimensional F32 tensor with its entries in the offload's expert order
    void up_permuted(Shard& s, const std::string& name, const std::vector<int>* perm) {
        if (!perm) {
            up_full(s, name);
            return;
        }
        const strata::GgufFile* file = nullptr;
        const auto* tensor = T(name, &file);
        if (tensor->type != T_F32 || tensor->shape.size() != 1 || tensor->shape[0] != perm->size())
            throw std::runtime_error("hy4f: cannot reorder " + name);
        const float* src = reinterpret_cast<const float*>(file->tensor_data(*tensor));
        std::vector<float> h(perm->size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = src[(*perm)[i]];
        DW w;
        w.type = T_F32;
        w.ne[0] = (int64_t) h.size();
        w.rb = row_bytes(T_F32, w.ne[0]);
        w.d = walloc(h.size() * sizeof(float));
        HY_CK(cudaMemcpy(w.d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        s.w[name] = w;
    }

    // the routing counts since the last call go to the cumulative statistics and the adaptation scores
    void harvest_routes() {
        if (sh.empty() || !sh[0].rstat) return;
        const int last = L + (has_mtp ? 1 : 0);
        const size_t all = (size_t) (L + 1) * n_exp;
        if (cum_dec.empty()) cum_dec.assign(all, 0), cum_rows.assign(all, 0), cum_touch.assign(all, 0);
        std::vector<unsigned> h((size_t) (L + 1) * nl_max);
        std::map<int, std::vector<double>> add;
        double tokens = 0;
        for (auto& s : sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaDeviceSynchronize());
            HY_CK(cudaMemcpy(h.data(), s.rstat, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
            HY_CK(cudaMemset(s.rstat, 0, h.size() * sizeof(unsigned)));
            for (int il = dense_lead; il < last; ++il) {
                const auto& own = expert_bounds(il);
                auto& a = add[il];
                if (a.empty()) a.assign(n_exp, 0.0);
                for (int l = 0; l < own[s.g + 1] - own[s.g]; ++l) {
                    const int id = own[s.g] + l, e = offload ? off_perm.at(il)[id] : id;
                    const size_t i = (size_t) il * nl_max + l, k = (size_t) il * n_exp + e;
                    cum_dec[k] += h[i];
                    cum_rows[k] += s.pf_rows[i];
                    cum_touch[k] += s.pf_touch[i];
                    a[id] += (double) h[i] + adapt_pf_weight * (double) s.pf_rows[i];
                    if (il == dense_lead) tokens += ((double) h[i] + adapt_pf_weight * (double) s.pf_rows[i]) / n_used;
                    if (offload && il < L) {
                        const OffLayer& ol = s.off.at(il);
                        const bool permanent = l < ol.n_res;
                        const bool res = permanent || (scratch_active && ol.scratch_index[l] >= 0);
                        st_picks += h[i], st_pf += s.pf_touch[i], win_picks += h[i];
                        if (res) st_hits += h[i], win_hits += h[i];
                        if (permanent) st_pf_hit += s.pf_touch[i];
                        if (ol.hybrid ? l < ol.n_hot : (ol.base_type && res)) st_hot += h[i], win_hot += h[i];
                    }
                    s.pf_rows[i] = 0, s.pf_touch[i] = 0;
                }
            }
        }
        if (!adapt) return;
        const double decay = std::exp(-tokens / adapt_horizon);
        for (auto& [il, score] : off_score) {
            const auto& a = add[il];
            for (int id = 0; id < n_exp; ++id) score[id] = score[id] * decay + a[id];
        }
    }

    // HY4F_EXPERT_ADAPT.  A swap exchanges the expert ids of two locals of one layer and shard, `hi` in the better
    // tier: its VRAM slot takes the other expert's weights, the router rows and biases swap on every shard, the
    // host-record indices swap (no host record moves).
    struct Swap {
        double gain;
        int il, g, hi, lo;
    };
    // pass 0: the best tier (the hybrid's hot one / the VRAM set) against everything below; pass 1: the hybrid's
    // cold VRAM tier against the host
    void plan_swaps(std::vector<Swap>& plan, int pass) const {
        const int nl = n_exp / S, last = L + (has_mtp ? 1 : 0);
        std::vector<int> res, host;
        for (int il = dense_lead; il < last; ++il) {
            const auto& score = off_score.at(il);
            for (int g = 0; g < S; ++g) {
                const OffLayer& ol = sh[g].off.at(il);
                if (pass == 1 && !ol.hybrid) continue;
                const int a0 = pass == 0 ? 0 : ol.n_hot, a1 = pass == 0 && ol.hybrid ? ol.n_hot : ol.n_res;
                res.clear(), host.clear();
                for (int l = a0; l < a1; ++l) res.push_back(l);
                for (int l = a1; l < nl; ++l) host.push_back(l);
                const int base = g * nl;
                std::sort(res.begin(), res.end(), [&](int a, int b) { return score[base + a] < score[base + b]; });
                std::sort(host.begin(), host.end(), [&](int a, int b) { return score[base + a] > score[base + b]; });
                const bool hot_tier = pass == 0 && ol.hybrid;   // (a quality tier: a wider hysteresis)
                for (size_t i = 0; i < res.size() && i < host.size(); ++i) {
                    if (score[base + host[i]] <= (hot_tier ? 1.5 : 1.25) * score[base + res[i]] + (hot_tier ? adapt_margin + 5.0 : adapt_margin)) break;
                    plan.push_back({score[base + host[i]] - score[base + res[i]], il, g, res[i], host[i]});
                }
            }
        }
    }
    void do_swap(const Swap& sw) {
        const int nl = n_exp / S, idh = sw.g * nl + sw.hi, idl = sw.g * nl + sw.lo;
        const std::string P = "blk." + std::to_string(sw.il) + ".";
        Shard& s = sh[sw.g];
        OffLayer& ol = s.off.at(sw.il);
        auto& perm = off_perm.at(sw.il);
        HY_CK(cudaSetDevice(s.dev));
        // sw.lo receives the old hot expert. If it also has a scratch slot,
        // keep that slot synchronized before replacing the permanent copy.
        if (scratch_active && ol.scratch_index[sw.lo] >= 0) {
            void* dest = reinterpret_cast<void*>(ol.record_base + size_t(ol.scratch_index[sw.lo]) * ol.stride);
            HY_CK(cudaMemcpy(dest, ol.vram + size_t(sw.hi)*ol.stride, ol.stride, cudaMemcpyDeviceToDevice));
        }
        if (ol.hybrid) {
            const size_t cs = ol.cstride;
            if (sw.hi < ol.n_hot) {   // the hot slot takes the base weights of lo's expert, read from the GGUF
                if (swap_pinned_bytes < ol.stride) {
                    if (swap_pinned) HY_CK(cudaFreeHost(swap_pinned));
                    swap_pinned_bytes = ol.stride + (1 << 20);
                    HY_CK(cudaHostAlloc((void**) &swap_pinned, swap_pinned_bytes, cudaHostAllocPortable));
                }
                fill_record(expert_src(sw.il, false), (size_t) perm[idl], swap_pinned);
                HY_CK(cudaMemcpy(ol.vram + (size_t) sw.hi * ol.stride, swap_pinned, ol.stride, cudaMemcpyHostToDevice));
                if (sw.lo < ol.n_res)   // lo's cold VRAM slot: the cold weights of hi's expert
                    HY_CK(cudaMemcpy(ol.cvram + (size_t) (sw.lo - ol.n_hot) * cs, ol.chost + (size_t) ol.chome[sw.hi] * cs, cs,
                                     cudaMemcpyHostToDevice));
            } else
                HY_CK(cudaMemcpy(ol.cvram + (size_t) (sw.hi - ol.n_hot) * cs, ol.chost + (size_t) ol.chome[sw.lo] * cs, cs,
                                 cudaMemcpyHostToDevice));
            std::swap(ol.chome[sw.hi], ol.chome[sw.lo]);
            if (sw.lo >= ol.n_res) {
                const unsigned long long ptr = (unsigned long long) (ol.chost + (size_t) ol.chome[sw.lo] * cs);
                HY_CK(cudaMemcpy(ol.cptr + sw.lo, &ptr, sizeof(ptr), cudaMemcpyHostToDevice));
            }
        } else {
            HY_CK(cudaMemcpy(ol.vram + (size_t) sw.hi * ol.stride, ol.host + (size_t) ol.home[sw.lo] * ol.stride, ol.stride,
                             cudaMemcpyHostToDevice));
            std::swap(ol.home[sw.hi], ol.home[sw.lo]);
            const int v = ol.home[sw.lo] + ol.ah;
            HY_CK(cudaMemcpy(ol.vmap + sw.lo, &v, sizeof(int), cudaMemcpyHostToDevice));
        }
        for (const char* name : {"ffn_gate_inp.weight", "exp_probs_b.bias"}) {
            const bool bias = name[0] == 'e';
            const size_t rb = bias ? sizeof(float) : sh[0].w.at(P + name).rb;
            swap_rows.resize(2 * rb);
            HY_CK(cudaSetDevice(sh[0].dev));
            uint8_t* w0 = static_cast<uint8_t*>(sh[0].w.at(P + name).d);
            HY_CK(cudaMemcpy(swap_rows.data(), w0 + rb * idh, rb, cudaMemcpyDeviceToHost));
            HY_CK(cudaMemcpy(swap_rows.data() + rb, w0 + rb * idl, rb, cudaMemcpyDeviceToHost));
            for (auto& t : sh) {
                HY_CK(cudaSetDevice(t.dev));
                uint8_t* w = static_cast<uint8_t*>(t.w.at(P + name).d);
                HY_CK(cudaMemcpy(w + rb * idh, swap_rows.data() + rb, rb, cudaMemcpyHostToDevice));
                HY_CK(cudaMemcpy(w + rb * idl, swap_rows.data(), rb, cudaMemcpyHostToDevice));
            }
        }
        std::swap(perm[idh], perm[idl]);
        std::swap(off_score.at(sw.il)[idh], off_score.at(sw.il)[idl]);
    }
    // Only between steps: nothing may be running on the devices.
    void rebalance(int max_swaps) {
        if (!adapt || !offload || max_swaps <= 0) return;
        win_picks = win_hits = win_hot = 0;
        harvest_routes();
        const auto t0 = std::chrono::steady_clock::now();
        size_t wanted = 0, done = 0;
        std::vector<Swap> plan;
        for (int pass = 0; pass < 2; ++pass) {
            plan.clear();
            plan_swaps(plan, pass);
            wanted += plan.size();
            std::sort(plan.begin(), plan.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            size_t left = (size_t) max_swaps - done;
            if (pass == 0 && hybrid) left = std::min<size_t>(left, std::max(16, max_swaps / 8));   // (hot swaps read the GGUF)
            if (plan.size() > left) plan.resize(left);
            for (const Swap& sw : plan) do_swap(sw);
            done += plan.size();
        }
        st_swaps += done;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        st_swap_s += dt;
        if (std::getenv("HY4F_ADAPT_LOG"))
            std::fprintf(stderr, "hy4f: adapt: window picks %llu in VRAM %.3f hot %.3f; %zu swaps of %zu wanted, %.1f ms\n",
                         (unsigned long long) win_picks, win_picks ? (double) win_hits / win_picks : 0.0,
                         win_picks ? (double) win_hot / win_picks : 0.0, done, wanted, dt * 1e3);
    }

    void dump_route_stats() {
        if (sh.empty() || !sh[0].rstat) return;
        try {
            harvest_routes();
        } catch (const std::exception&) {
            return;
        }
        const int last = L + (has_mtp ? 1 : 0);
        if (offload)
            std::fprintf(stderr, "hy4f: offload: decode picks %llu, in VRAM %.3f, hot %.3f; prefill expert uses %llu, in VRAM %.3f; "
                                 "%llu swaps in %.2f s\n",
                         (unsigned long long) st_picks, st_picks ? (double) st_hits / st_picks : 0.0,
                         st_picks ? (double) st_hot / st_picks : 0.0, (unsigned long long) st_pf,
                         st_pf ? (double) st_pf_hit / st_pf : 0.0, (unsigned long long) st_swaps, st_swap_s);
        if (offload && o.verbose)
            for (auto& s : sh) {
                size_t fr = 0, tot = 0;
                if (cudaSetDevice(s.dev) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess)
                    std::fprintf(stderr, "hy4f: GPU %d free %.2f GiB at exit\n", s.dev, fr / 1073741824.0);
            }
        for (const auto& ln : lanes)
            std::fprintf(stderr, "hy4f: CPU lane %d: %llu requests, %llu expert groups, %llu slots, %.2f s busy (%.3f ms a request)\n",
                         ln->g, (unsigned long long) ln->n_req, (unsigned long long) ln->n_groups,
                         (unsigned long long) ln->n_slots, ln->busy, ln->n_req ? 1e3 * ln->busy / ln->n_req : 0.0),
            std::fprintf(stderr, "hy4f: CPU lane %d stages, ms a request: prepare %.3f, gate/up %.3f, quantize h %.3f, down %.3f\n", ln->g,
                         ln->n_req ? 1e3 * ln->t_prep / ln->n_req : 0.0, ln->n_req ? 1e3 * ln->t_gu / ln->n_req : 0.0,
                         ln->n_req ? 1e3 * ln->t_qh / ln->n_req : 0.0, ln->n_req ? 1e3 * ln->t_dn / ln->n_req : 0.0);
        if (route_stats.empty()) return;
        std::FILE* f = std::fopen(route_stats.c_str(), "w");
        if (!f) return;
        for (int il = dense_lead; il < last; ++il)
            for (int e = 0; e < n_exp; ++e) {
                const size_t k = (size_t) il * n_exp + e;
                // (half of the profile this run started from stays in: one file as HY4F_EXPERT_PROFILE and
                // HY4F_ROUTE_STATS carries the routing history over restarts, the older runs fading)
                const auto seed = profile_seed.find(il);
                const unsigned long long d = cum_dec[k] + (seed == profile_seed.end() ? 0ull : (unsigned long long) (0.5 * seed->second[e]));
                if (d) std::fprintf(f, "D %d %d %llu\n", il, e, d);
                if (cum_rows[k]) std::fprintf(f, "P %d %d %llu\n", il, e, (unsigned long long) cum_rows[k]);
                if (cum_touch[k]) std::fprintf(f, "T %d %d %llu\n", il, e, (unsigned long long) cum_touch[k]);
            }
        std::fclose(f);
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
        if (cert_head) up_draft_head(s,T_Q8_0,true);
        if (has_mtp && draft_head_type && !cert_mtp && !(cert_head && draft_head_type==T_Q8_0))
            up_draft_head(s,draft_head_type,false);
        const char* project_env = std::getenv("HY4F_Q8_KV_PROJ");
        const int project_mask = project_env ? std::atoi(project_env) : 0;
        const bool pack_k = (project_mask & 1) && nope == 192 && kvl == 512;
        const bool pack_v = (project_mask & 2) && kvl == 512 && vh == 256;
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
                up_rows(s, P + "attn_k_b.weight", (int64_t) h0 * kvl, (int64_t) (h0 + Hl) * kvl, pack_k);
            } else {
                up_concat(s, P + "qproj", {{P + "attn_q_b.weight", 0, -1}, {P + "indexer.attn_q_b.weight", 0, -1}});
                up_rows(s, P + "attn_k_b.weight", 0, -1, pack_k);
                up_concat(s, P + "ivg", {{P + "indexer_compressor_value.weight", 0, -1},
                                         {P + "indexer_compressor_gate.weight", 0, -1}});
                for (const char* n : {"indexer.k_norm.weight", "indexer.proj.weight", "indexer_compressor_ape.weight",
                                      "indexer_compressor_norm.weight"})
                    up_full(s, P + n);
            }
            up_rows(s, P + "attn_v_b.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh, pack_v);
            up_cols(s, P + "attn_output.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh);
            if (il < dense_lead) {
                const int64_t f0 = (int64_t) g * n_ff / S, f1 = (int64_t) (g + 1) * n_ff / S;
                up_concat(s, P + "dense_gu", {{P + "ffn_gate.weight", f0, f1}, {P + "ffn_up.weight", f0, f1}});
                up_cols(s, P + "ffn_down.weight", f0, f1);
            } else {
                const std::vector<int>* perm = offload ? &off_perm.at(il) : nullptr;
                up_router(s, P + "ffn_gate_inp.weight", perm);
                up_permuted(s, P + "exp_probs_b.bias", perm);
                up_full(s, P + "ffn_routed_up.weight");
                const auto& ownership = expert_bounds(il);
                const int64_t e0 = ownership[g], e1 = ownership[g + 1];
                if (offload) s.off_pending.push_back(il);
                else {   // gate | up of each expert as one [2 n_ff, latent] matrix (MMQ run_glu's layout; one decode launch)
                    const strata::GgufFile *fg = nullptr, *fu = nullptr;
                    const strata::TensorInfo* tg = T(P + "ffn_gate_exps.weight", &fg);
                    const strata::TensorInfo* tu = T(P + "ffn_up_exps.weight", &fu);
                    if (tg->type != tu->type) throw std::runtime_error("hy4f: gate/up expert types differ in " + P);
                    if (!tuned_expert_type((int) tg->type))
                        throw std::runtime_error("hy4f: experts of type " + std::to_string((int) tg->type) +
                                                 " need HY4F_EXPERT_VRAM_MB (their kernels read the offload records)");
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
                if (replicate_prefix())
                    HY_CK(cudaMalloc(&s.prefix_kv[il], (size_t)top_k * KV_RB));
                s.ring[il] = dalloc((size_t) IRING * 2 * IK);
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
        HY_CK(cudaMalloc(&s.evirt, sizeof(int) * n_used));
        if (!route_stats.empty() || adapt) {
            const size_t n = (size_t) (L + 1) * nl_max;
            HY_CK(cudaMalloc(&s.rstat, sizeof(unsigned) * n));
            HY_CK(cudaMemset(s.rstat, 0, sizeof(unsigned) * n));
            s.pf_rows.assign(n, 0);
            s.pf_touch.assign(n, 0);
        }
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
        if (offload) {   // the experts last: the auto budget is what the rest left free
            if (off_auto) {
                size_t fr = 0, tot = 0;
                HY_CK(cudaMemGetInfo(&fr, &tot));
                const size_t budget = fr > off_reserve ? fr - off_reserve : 0;
                if (o.verbose)
                    std::fprintf(stderr, "hy4f: offload: shard %d free %.2f GiB, expert budget %zu MiB\n", g, fr / 1073741824.0,
                                 budget >> 20);
                plan_residency(g, budget);
            }
            const char* lane_mode = std::getenv("HY4F_CPU_LANE");
            scratch_cache = S == 1 && !hybrid && lane_mode && std::atoi(lane_mode)
                && single_prefill_mode() && std::getenv("HY4F_SINGLE_WORKSPACE_CACHE");
            if (scratch_cache) batch_setup(o.prefill_chunk > 0 ? o.prefill_chunk : 2048);
            for (int il : s.off_pending) load_offload_experts(s, il, "blk." + std::to_string(il) + ".");
            s.off_pending.clear();
        }
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
        case T_ROUTER_PACKED:
            kd_router_gemv<<<dim3(nb(n_out, 4), 1), 128, 0, s.st>>>(reinterpret_cast<const uint16_t*>(d), x, y, n_in, n_out);
            break;
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
    void router(Shard& s, const float* lg, const float* bias, int T, int* ids, float* w, cudaStream_t st = nullptr) {
        if (!st) st = s.st;
        if (n_exp == 512 && !std::getenv("HY4F_OLD_ROUTER"))
            kd_router<512><<<T, 512, 0, st>>>(lg, bias, n_used, w_scale, ids, w);
        else
            kn::router_sigmoid(lg, bias, T, n_exp, n_used, w_scale, true, ids, w, st);
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
            if (kb.packed_projection)
                k_project_eager<192><<<nb(Hl * kvl), 256, 0, s.st>>>((const uint8_t*)kb.d, s.qp, nullptr, s.qabs, Hl, QH);
            else if (kb.type == T_Q8_0) k_absorb_q8<<<nb(Hl * kvl), 256, 0, s.st>>>((const uint8_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hl);
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
            if (kb.packed_projection)
                k_project_eager<192><<<nb(Hd * kvl), 256, 0, s.st>>>((const uint8_t*)kb.d, s.qp, nullptr, s.qabs, Hd, QH);
            else if (kb.type == T_Q8_0) k_absorb_q8<<<nb(Hd * kvl), 256, 0, s.st>>>((const uint8_t*) kb.d, s.qp, QH, s.qabs, nope, kvl, Hd);
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
        if (vb.packed_projection)
            k_project_eager<512><<<nb(Hl * vh * 32), 256, 0, s.st>>>((const uint8_t*)vb.d, s.o, s.xp + og, s.a, Hl, kvl);
        else if (vb.type == T_Q8_0)
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
        const auto& ownership = expert_bounds(il);
        const int e0 = ownership[s.g], e1 = ownership[s.g + 1], fl = n_ffx / S;
        k_local_ids<<<1, 32, 0, s.st>>>(s.eids, s.eloc, n_used, e0, e1);
        const int* eidx = s.eloc;
        if (offload) {
            const OffLayer& ol = s.off.at(il);
            if (ol.hybrid) throw std::runtime_error("hy4f: the hybrid experts need the CUDA graphs' step");
            k_virt_ids<<<1, 32, 0, s.st>>>(s.eloc, s.evirt, n_used, ol.vmap);
            eidx = s.evirt;
        }
        mmvq(s, P + "xmoe", s.xq, s.xm);
        k_q8<<<nb(latent), 256, 0, s.st>>>(s.xm, latent, s.xlq);
        auto ids_mmvq = [&](const DW& w, const void* xq, bool shared, float* y, int n_in, int n_out) {
            const size_t eb = expert_stride(w);
            if (w.type == T_Q8_0) {
                expert_q8::launch<false>(w.d, eb, eidx, n_used, nullptr, nullptr, nullptr, xq, shared, y, n_in, n_out,
                                         nb(n_out, 4), s.st);
                HY_CK(cudaGetLastError());
            }
            else if (w.type == T_Q3_K) kn::native_q3_k_mmvq_ids_loop(w.d, eb, eidx, n_used, xq, shared, y, n_in, n_out, s.st);
            else if (w.type == T_Q4_K) kn::native_q4_k_mmvq_ids_loop(w.d, eb, eidx, n_used, xq, shared, y, n_in, n_out, s.st);
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
        dump_route_stats();
        lane_stop();
        for (auto& s : sh) {
            cudaSetDevice(s.dev);
            for (auto& [il, layer] : s.off)
                if (layer.device_scratch_index) cudaFree(layer.device_scratch_index);
        }
        if (pool) {
            {
                std::lock_guard<std::mutex> lk(pool->m);
                pool->quit = true;
            }
            pool->cv.notify_all();
            for (auto& t : pool->th) t.join();
        }
        for (size_t i = 0; i < dsh.size(); ++i) {
            cudaSetDevice(sh[i].dev);
            auto& d = dsh[i];
            for (auto& graph : d.gv) graph.clear();
            for (auto& graph : d.gm) graph.clear();
            d.gc.clear();
            for (auto& graph : d.mtp_heads) graph.clear();
            for (auto& pair : d.verify_heads) for (auto& graph : pair) graph.clear();
            if (d.engram_side) cudaStreamDestroy(d.engram_side);
            if (d.engram_fork) cudaEventDestroy(d.engram_fork);
            if (d.engram_ready) cudaEventDestroy(d.engram_ready);
            if (d.engram_q) cudaFree(d.engram_q);
            if (d.cert.input) cudaFree(d.cert.input);
            if (d.cert.lower) cudaFree(d.cert.lower);
            if (d.cert.upper) cudaFree(d.cert.upper);
            if (d.cert.count) cudaFree(d.cert.count);
            if (d.cert.ids) cudaFree(d.cert.ids);
            if (d.cert.rank_hist) cudaFree(d.cert.rank_hist);
            if (d.cert.rank_cut) cudaFree(d.cert.rank_cut);
            if (d.packed_query) cudaFree(d.packed_query);
            if (d.index_cache) cudaFree(d.index_cache);
            if (d.sel_compact) cudaFree(d.sel_compact);
            if (d.rank_tiles) cudaFree(d.rank_tiles);
            if (d.rank_final) cudaFree(d.rank_final);
        }
        if (rank_host) cudaFreeHost(rank_host);
        discard_full_storage();
        for (size_t g = 0; g < bsh.size(); ++g) {
            auto& b = bsh[g];
            cudaSetDevice(sh[g].dev);
            if (b.routing_ready) cudaEventDestroy(b.routing_ready);
            if (!b.ag_compute) continue;
            for (auto event : b.ag_slice) if (event) cudaEventDestroy(event);
            if (b.ag_complete) cudaEventDestroy(b.ag_complete);
            cudaStreamDestroy(b.ag_compute);
        }
    }

#include "hy4f_batch.inc"
#include "hy4f_partial.inc"
#include "hy4f_dec.inc"
#include "hy4f_lane.inc"
#include "hy4f_dfly.inc"

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
           ", max context " + std::to_string(d_->o.max_context) + (d_->has_mtp ? ", MTP" : "") +
           (d_->df_nl ? ", DFly2 (" + std::to_string(d_->df_nl) + " layers)" : "");
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
        d.load_draft_experts(std::getenv("HY4F_MTP_EXPERT_GGUF"), false);
        d.load_draft_experts(std::getenv("HY4F_MTP_DOWN_GGUF"), true);
        d.load_trunk_experts(std::getenv("HY4F_EXPERT_GGUF"));
        d.load_engram();
        HY_CK(cudaHostAlloc((void**) &d.hio, sizeof(Io), cudaHostAllocPortable));
        HY_CK(cudaHostAlloc((void**) &d.hlogits, sizeof(float) * d.V, cudaHostAllocPortable));
        {
            const strata::GgufFile* f = nullptr;
            const strata::TensorInfo* t = d.find("blk." + std::to_string(d.L) + ".nextn.eh_proj.weight", &f);
            const strata::TensorInfo* te = d.find("blk." + std::to_string(d.L) + ".ffn_gate_exps.weight", &f);
            if (const char* bits = std::getenv("HY4F_MTP_HEAD_Q")) {
                if (!std::strcmp(bits, "8")) d.draft_head_type = T_Q8_0;
                else if (!std::strcmp(bits, "4")) d.draft_head_type = T_Q4_K;
                else throw std::runtime_error("hy4f: HY4F_MTP_HEAD_Q expects 8 or 4");
            }
            const bool eligible = d.compact_io && d.E == 3072;
            const bool both = std::getenv("HY4F_CERT_HEAD") != nullptr;
            d.cert_target = eligible && (both || std::getenv("HY4F_CERT_TARGET"));
            d.cert_mtp = eligible && (both || std::getenv("HY4F_CERT_MTP"));
            const char* ranked=std::getenv("HY4F_CERT_RANKED");
            d.cert_ranked = eligible && d.rank_geometry() && ranked && !std::strcmp(ranked,"1");
            d.cert_head = d.cert_target || d.cert_mtp || d.cert_ranked;
            d.has_mtp = t && te && (int) te->type != 30 /*BF16 experts: requantize first (scripts/requant_mtp_q3k.py)*/ &&
                        (int) t->type == 8 && !std::getenv("HY4F_NO_MTP") && (o.dfly.empty() || std::getenv("HY4F_KEEP_MTP"));
        }
        if (!o.dfly.empty()) {
            d.files.emplace_back(new strata::GgufFile(o.dfly));
            d.load_dfly_meta();
        }
        d.split_experts(devs);
        d.sh.resize(d.S);
        for (int g = 0; g < d.S; ++g) {
            d.sh[g].g = g;
            d.sh[g].dev = devs[g];
        }
        if (d.S > 1 && std::getenv("HY4F_PARALLEL_LOAD")) {
            // Materialize the only lazy host-side weight before rank workers read it.
            d.emb_host();
            d.launch_parallel([&](Shard& s) { d.load_shard(s); });
        } else {
            for (auto& s : d.sh) d.load_shard(s);
        }
        if (d.df_nl) for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            d.load_dfly_shard(s);
        }
        if (d.S > 1) d.setup_peers();
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaDeviceSynchronize());
        }
        d.lane_start();
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

bool Model::warm_full_verify(std::string& err) {
    try {
        Impl& d=*d_;
        if (!d.graphs || !d.ensure_full_storage()) return true;
        for (int T=1; T<=Impl::DT; ++T) d.capture_full_verify(T);
        return true;
    } catch (const std::exception& e) {
        err=e.what();
        return false;
    }
}

bool Model::prefill(const int32_t* toks, int n, int64_t pos, float* logits, int chunk, std::string& err,
                    const std::vector<EmbdSpan>* spans) {
    try {
        Impl& d = *d_;
        if (pos > d.pos_next) throw std::runtime_error("hy4f: prefill out of order");
        d.packed_index_groups = -1;
        if (pos < prefix_pos_) prefix_pos_=-1;
        if (pos + n > d.o.max_context) throw std::runtime_error("hy4f: context full");
        if (d.sparse_limit) {
            const int end = (int)(pos+n);
            if (end > d.sparse_limit) d.sparse_valid = false;
            d.sparse_high = std::max(d.sparse_high, std::min(end,d.sparse_limit));
        }
        d.truncate_full_mirrors((int)pos);
        const auto t_start = std::chrono::steady_clock::now();
        const char* short_limit = d.scratch_cache ? std::getenv("HY4F_SINGLE_SHORT_VERIFY") : nullptr;
        if (short_limit && n > 0 && n <= std::atoi(short_limit) && d.has_mtp && !d.dsh.empty()
            && (!spans || spans->empty())) {
            // A short text turn uses the decode graphs and keeps the cached experts in place.
            // At the boundary, MTP pairs the previous trunk hidden state with the new token.
            Shard& s = d.sh.front();
            auto& b = d.bsh.front();
            auto& ds = d.dsh.front();
            HY_CK(cudaSetDevice(s.dev));
            if (pos > 0) {
                HY_CK(cudaMemcpyAsync(ds.h,b.hprev,sizeof(float)*d.E,cudaMemcpyDeviceToDevice,s.st));
                if (!mtp(toks,1,pos-1,nullptr,err)) return false;
            }
            for (int t = 0; t < n; t += Impl::DT) {
                const int T = std::min(Impl::DT, n-t);
                const bool last = t+T == n;
                const int pairs = last ? T-1 : T;
                // Use the native output staging to avoid a host allocation for T full rows.
                d.hio_v->pos = int(pos)+t;
                d.hio_v->n = T;
                std::copy_n(toks+t,T,d.hio_v->toks);
                d.dec_launch(false,T,false,last && logits);
                if (last && logits) std::copy_n(d.hlogT + size_t(T-1)*d.V,d.V,logits);
                if (pairs && !mtp(toks+t+1,pairs,pos+t,nullptr,err)) return false;
                if (last) {
                    HY_CK(cudaMemcpyAsync(b.hprev,ds.h+size_t(T-1)*d.E,sizeof(float)*d.E,cudaMemcpyDeviceToDevice,s.st));
                    HY_CK(cudaMemcpyAsync(ds.h,b.hprev,sizeof(float)*d.E,cudaMemcpyDeviceToDevice,s.st));
                    HY_CK(cudaStreamSynchronize(s.st));
                }
            }
            d.pos_next = pos+n;
            d.full_prefill_warmed = true;
            d.rebalance(d.adapt_pf_swaps);
            if (logits && !d.o.dump_dir.empty()) {
                if (auto* f = std::fopen((d.o.dump_dir+"/logits.f32").c_str(),"ab")) {
                    std::fwrite(logits,sizeof(float),d.V,f);
                    std::fclose(f);
                }
            }
            if (d.o.verbose) {
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_start).count();
                std::fprintf(stderr,"hy4f: graph prefill %d tokens: %.1f ms (%.1f tok/s)\n",n,ms,1000.*n/ms);
            }
            return true;
        }
        d.release_decode_scratch();
        d.change_q3_layout(false);
        if (chunk <= 0) chunk = 1024;
        // two micro-batches (overlap one half's exchanges with the other's compute) unless HY4F_MB=1
        // (experimental, off by default: measured slower - MMQ / attention fill the SMs, the SM-based exchanges cannot run
        // beside them; a real overlap needs DMA transfers)
        const bool mb = d.S > 1 && chunk >= 512 && std::getenv("HY4F_MB") && std::atoi(std::getenv("HY4F_MB")) == 2;
        if (d.sparse_limit && mb) d.sparse_valid = false;
        const int tmax = mb ? chunk / 2 : chunk;
        if (d.bsh.empty() || d.bsh[0].Tmax < tmax) {
            if (!d.bsh.empty()) throw std::runtime_error("hy4f: prefill chunk grew");
            if (mb) d.attn_tile = 256, d.score_tile = 32, d.cap_num = 5, d.cap_den = 4;   // (two halves' buffers in the VRAM of one)
            d.batch_setup(tmax);
            if (mb) d.mb_setup(tmax);
        }
        const bool build_full = d.full_requested && !mb && d.sparse_limit
            && std::max<int64_t>(d.full_request_end, pos+n) > d.sparse_limit;
        int progressive_end = d.full_ready ? d.full_prefix[0] : 0;
        d.full_batch_writes = false;
        for (int c0 = 0; c0 < n; c0 += chunk) {
            const int T = std::min(chunk, n - c0);
            const bool last = c0 + T == n;
            const int p0 = (int) (pos + c0);
            const bool can_fill = build_full && p0 >= progressive_end && p0-progressive_end <= 4096
                && (d.full_ready || d.full_prefill_warmed || c0 > 0);
            if (can_fill && c0 > 0 && (!d.full_ready || p0 != progressive_end))
                d.launch_parallel([&](Shard& s) { HY_CK(cudaStreamSynchronize(s.st)); });
            if (can_fill && d.ensure_full_storage()) {
                if (p0 != progressive_end) d.restore_full_mirrors(p0);
                d.full_batch_writes = true;
                progressive_end = p0 + T;
            } else {
                d.full_batch_writes = false;
            }
            std::vector<Impl::Span> cs;   // the spans' pieces inside this chunk
            if (spans)
                for (const EmbdSpan& e : *spans) {
                    const int a = std::max(e.offset, c0), z = std::min(e.offset + e.n, c0 + T);
                    if (a < z) cs.push_back({a - c0, z - a, e.rows + (size_t) (a - e.offset) * d.E});
                }
            d.launch_parallel([&](Shard& s) {
                Impl::BShard& b = d.bsh[s.g];
                if (!d.mbs.empty() && T > b.Tmax) {
                    Impl::MbState& M = d.mbs[s.g];
                    const int TA = T / 2;
                    HY_CK(cudaMemcpyAsync(b.toks, toks + c0, sizeof(int) * TA, cudaMemcpyHostToDevice, b.st));
                    HY_CK(cudaMemcpyAsync(M.B.toks, toks + c0 + TA, sizeof(int) * (T - TA), cudaMemcpyHostToDevice, M.B.st));
                    std::vector<Impl::Span> sa, sb;
                    for (const auto& x : cs) {
                        if (x.t0 < TA) sa.push_back({x.t0, std::min(x.n, TA - x.t0), x.rows});
                        if (x.t0 + x.n > TA) {
                            const int a0 = std::max(x.t0, TA);
                            sb.push_back({a0 - TA, x.t0 + x.n - a0, x.rows + (size_t) (a0 - x.t0) * d.E});
                        }
                    }
                    d.b_chunk2(s, b, M, p0, T, last && logits, sa, sb);
                } else {
                    HY_CK(cudaMemcpyAsync(b.toks, toks + c0, sizeof(int) * T, cudaMemcpyHostToDevice, s.st));
                    d.b_chunk(s, b, p0, T, last && logits, cs);
                }
            });
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaStreamSynchronize(s.st));
        }
        HY_CK(cudaGetLastError());
        d.full_prefill_warmed = true;
        if (d.full_batch_writes) std::fill_n(d.full_prefix,d.full_count,progressive_end);
        d.full_batch_writes = false;
        d.pos_next = pos + n;
        d.rebalance(d.adapt_pf_swaps);
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
        if (pos < prefix_pos_) prefix_pos_=-1;
        if (pos + T > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_v->pos = (int) pos;
        d.hio_v->n = T;
        for (int t = 0; t < T; ++t) d.hio_v->toks[t] = toks[t];
        d.dec_launch(false, T, false, logits != nullptr);
        d.pos_next = pos + T;
        if (d.adapt && ++d.adapt_calls % d.adapt_every == 0) d.rebalance(d.adapt_step_swaps);
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

bool Model::mtp(const int32_t* next_toks, int T, int64_t mpos, int32_t* draft, std::string& err, float* logits) {
    try {
        Impl& d = *d_;
        if (!d.has_mtp || d.dsh.empty()) throw std::runtime_error("hy4f: no MTP layer (or no CUDA graphs)");
        if (T < 1 || T > Impl::DT) throw std::runtime_error("hy4f: mtp takes 1.." + std::to_string(Impl::DT) + " tokens");
        if (mpos + T > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_m->pos = (int) mpos;
        d.hio_m->n = T;
        d.hio_m->last_pos = (int) mpos + T - 1;
        for (int t = 0; t < T; ++t) d.hio_m->toks[t] = next_toks[t];
        d.dec_launch(true, T, false, logits != nullptr);
        d.m_next = mpos + T;
        float bv = -INFINITY;
        int bi = 0;
        for (int g = 0; g < d.S; ++g)
            if (d.mtp_max(g) > bv) bv = d.mtp_max(g), bi = d.hmargi[g];
        if (draft) *draft = bi;
        if (logits) std::memcpy(logits, d.hmlog, sizeof(float) * d.V);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
bool Model::mtp_chain(int32_t tok, int64_t mpos, int32_t* draft, std::string& err, float* logits) {
    try {
        Impl& d = *d_;
        if (!d.has_mtp || d.dsh.empty()) throw std::runtime_error("hy4f: no MTP layer (or no CUDA graphs)");
        if (mpos + 1 > d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio_m->pos = (int) mpos;
        d.hio_m->n = 1;
        d.hio_m->toks[0] = tok;
        d.dec_launch(true, 1, true, logits != nullptr);
        float bv = -INFINITY;
        int bi = 0;
        for (int g = 0; g < d.S; ++g)
            if (d.mtp_max(g) > bv) bv = d.mtp_max(g), bi = d.hmargi[g];
        if (draft) *draft = bi;
        if (logits) std::memcpy(logits, d.hmlog, sizeof(float) * d.V);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
void Model::set_greedy_draft_head(bool enabled) { d_->greedy_draft_head = enabled; }
void Model::set_partial_prompt_end(int pos) { d_->full_request_end = pos; }

bool Model::set_ranked_output(bool enabled) {
    d_->rank_active=enabled && d_->rank_host;
    return d_->rank_active;
}
bool Model::ranked_row(int row, int k, float inv_temp, std::vector<HeadValue>& values, float* fallback) {
    Impl& d=*d_;
    const int n=d.V/d.S;
    bool ambiguous=!(inv_temp>0.f && std::isfinite(inv_temp));
    const HeadValue* ranks=d.rank_host+(size_t)row*d.S*head_candidates;
    for (int g=0; g<d.S; ++g) {
        const auto* v=ranks+g*head_candidates;
        // Scaling raw logits can create a tie at the truncation boundary.
        // The full sampler breaks such ties by vocabulary ID.
        ambiguous |= v[0].id<0 || !std::isfinite(v[0].logit*inv_temp) ||
                     v[k-1].logit*inv_temp==v[k].logit*inv_temp;
    }
    if (ambiguous) {
        const unsigned bit=1u<<row;
        d.launch_parallel([&](Shard& s) {
            auto& ds=d.dsh[s.g];
            auto* logits=ds.logits+(size_t)row*n;
            if (d.rank_pruned_rows & bit) {
                const auto* w=static_cast<const uint16_t*>(s.w.at("output.weight").d);
                // Verify preserves every normalized head input in h; MTP preserves its final input in mh.
                const float* x=d.rank_from_mtp ? ds.mh : ds.h+(size_t)row*d.E;
                kd_bf16_gemv<1><<<nb(n*32),256,0,s.st>>>(w,x,d.E,n,logits);
            }
            HY_CK(cudaMemcpyAsync(fallback+(size_t)s.g*n,logits,sizeof(float)*n,cudaMemcpyDeviceToHost,s.st));
            HY_CK(cudaStreamSynchronize(s.st));
        });
        d.rank_pruned_rows &= ~bit;
        return false;
    }
    values.clear();
    for (int g=0; g<d.S; ++g) {
        const auto* v=ranks+g*head_candidates;
        values.insert(values.end(),v,v+k);
    }
    return true;
}

float Model::draft_margin(int token) const {
    const Impl& d = *d_;
    const int rows = d.V / d.S, owner = token / rows;
    float runner_up = -INFINITY;
    for (int g = 0; g < d.S; ++g) if (g != owner) runner_up = std::max(runner_up, d.mtp_max(g));
    if (d.compact_io && token == d.hmargi[owner])
        return d.mtp_max(owner) - std::max(runner_up, d.hmargv[2 * owner + 1]);
    // An alternative-token query needs the original host reduction as well:
    // std::max_element's comparison order is observable for nonfinite logits.
    if (d.certified_mtp_active) {
        const auto& s=d.sh[owner];
        const auto& ds=d.dsh[owner];
        HY_CK(cudaSetDevice(s.dev));
        const auto* w=static_cast<const uint16_t*>(s.w.at("output.weight").d);
        kd_bf16_gemv<1><<<nb(rows*32),256,0,s.st>>>(w,ds.mh,d.E,rows,ds.logits);
        HY_CK(cudaStreamSynchronize(s.st));
    }
    if (!d.mtp_full_row) {
        HY_CK(cudaSetDevice(d.sh[owner].dev));
        HY_CK(cudaMemcpy(d.hmlog + (size_t)owner * rows, d.dsh[owner].logits,
                         rows * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const float* logits = d.hmlog;
    const int begin = owner * rows, end = begin + rows;
    if (begin < token) runner_up = std::max(runner_up, *std::max_element(logits + begin, logits + token));
    if (token + 1 < end) runner_up = std::max(runner_up, *std::max_element(logits + token + 1, logits + end));
    return logits[token] - runner_up;
}
bool Model::has_mtp() const { return d_->has_mtp; }
bool Model::has_dfly() const { return d_->df_nl > 0 && !d_->dfs.empty(); }
bool Model::dfly(int src_row, int32_t bonus, int64_t anchor, int n, int32_t* drafts, std::string& err, float* logits) {
    try {
        Impl& d = *d_;
        if (!has_dfly()) throw std::runtime_error("hy4f: no DFly drafter (or no CUDA graphs)");
        if (n < 1 || n > Impl::DF_N) throw std::runtime_error("hy4f: dfly drafts 1..5");
        if (src_row < 0 || src_row >= Impl::DT) throw std::runtime_error("hy4f: dfly source row");
        if (anchor + Impl::DF_B > d.o.max_context) throw std::runtime_error("hy4f: context full");
        if (logits && !d.df_logits) throw std::runtime_error("hy4f: dfly logits need HY4F_DFLY_LOGITS at load");
        d.df_run(src_row, bonus, anchor, n, drafts);
        if (logits) std::memcpy(logits, d.df_hlog, sizeof(float) * n * d.V);
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Model::load_vision(const std::string& mmproj, std::string& err) {
#ifdef HY4F_VISION
    d_->venc.reset(new vision::Encoder());
    vision::Options vo;
    vo.verbose = d_->o.verbose;
    if (!d_->venc->load(mmproj, vo, err)) {
        d_->venc.reset();
        return false;
    }
    return true;
#else
    (void) mmproj;
    err = "hy4f: built without the vision encoder";
    return false;
#endif
}
bool Model::has_vision() const {
#ifdef HY4F_VISION
    return d_->venc != nullptr;
#else
    return false;
#endif
}


bool Model::encode_image(const std::string& path, std::vector<float>& rows, int& n_tokens, std::string& err) {
#ifdef HY4F_VISION
    try {
        vision::Image img;
        if (!vision::load_image(path, img, err)) return false;
        return d_->encode_img(img, rows, n_tokens, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
#else
    (void) path, (void) rows, (void) n_tokens;
    err = "hy4f: built without the vision encoder";
    return false;
#endif
}
bool Model::encode_image_mem(const uint8_t* bytes, size_t n, std::vector<float>& rows, int& n_tokens, std::string& err) {
#ifdef HY4F_VISION
    try {
        vision::Image img;
        if (!vision::load_image_mem(bytes, n, img, err)) return false;
        return d_->encode_img(img, rows, n_tokens, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
#else
    (void) bytes, (void) n, (void) rows, (void) n_tokens;
    err = "hy4f: built without the vision encoder";
    return false;
#endif
}

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
        d.truncate_full_mirrors((int)pos);
        d.packed_index_groups = -1;
        if (pos >= d.o.max_context) throw std::runtime_error("hy4f: context full");
        d.hio->pos = (int) pos;
        d.change_q3_layout(false);
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
// group's positions in row p % IRING (8 rows: a rejected draft never overwrites the open group's rows).  A row a kept position needs is lost when a later position sharing it was written
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
    bool cold_aligned = false;  // host cache provenance for exact 2048-token continuations
    bool aligned_start = false;
    int64_t pos = -1;                                   // the position mark() copied at (-1: none)
    std::vector<int> devices;
    ~Snap() {
        for (size_t i=0; i<devices.size(); ++i) {
            cudaSetDevice(devices[i]);
            for (const auto& item : rows[i]) cudaFree(item.second);
        }
    }
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
    return {(void*) s.ring[il], (size_t) IRING * 2 * IK * sizeof(float)};
}
}  // namespace

struct Model::PrefixSnap {
    bool cold_aligned = false;
    std::vector<uint8_t*> buffers;
    ~PrefixSnap() { for (auto* p : buffers) if (p) cudaFreeHost(p); }
};

void Model::transfer_prefix(bool save) {
    Impl& d = *d_;
    const int layers = d.L + int(d.has_mtp);
    if (!prefix_) {
        auto snap = std::make_shared<PrefixSnap>();
        snap->buffers.resize(d.S);
        for (auto& s : d.sh) {
            size_t bytes = d.has_mtp ? sizeof(float)*d.E : 0;
            for (int il=0;il<layers;++il)
                bytes += serve_ring(s,il,d.swa[il]!=0,d.win,d.kpool,d.IK).second;
            HY_CK(cudaHostAlloc((void**)&snap->buffers[s.g],bytes,cudaHostAllocPortable));
        }
        prefix_=std::move(snap);
    }
    d.launch_parallel([&](Shard& s) {
        uint8_t* host=prefix_->buffers[s.g];
        auto copy=[&](void* gpu,size_t bytes) {
            if(save) HY_CK(cudaMemcpyAsync(host,gpu,bytes,cudaMemcpyDeviceToHost,s.st));
            else HY_CK(cudaMemcpyAsync(gpu,host,bytes,cudaMemcpyHostToDevice,s.st));
            host += bytes;
        };
        for(int il=0;il<layers;++il) {
            auto row=serve_ring(s,il,d.swa[il]!=0,d.win,d.kpool,d.IK);
            copy(row.first,row.second);
        }
        if(d.has_mtp) {
            auto* prev=d.bsh[s.g].hprev;
            copy(prev,sizeof(float)*d.E);
            if(!save && !d.dsh.empty())
                HY_CK(cudaMemcpyAsync(d.dsh[s.g].h,prev,sizeof(float)*d.E,cudaMemcpyDeviceToDevice,s.st));
        }
        HY_CK(cudaStreamSynchronize(s.st));
    });
}

struct Model::HostPrefix {
    struct Buffer { uint8_t* ptr = nullptr; size_t capacity = 0; };
    struct Region { void* device; size_t bytes; };
    std::vector<Buffer> buffers;
    int64_t pos = 0, capacity_rows = 0, begin = 0;
    int parent = -1;
    uint64_t generation = 0, parent_generation = 0;
    bool delta_capacity = false;
    explicit HostPrefix(int shards) : buffers(shards) {}
    ~HostPrefix() { for (auto& b : buffers) if (b.ptr) cudaFreeHost(b.ptr); }

    static std::vector<Region> regions(Impl& d, Shard& s, int64_t pos, int64_t begin = 0) {
        std::vector<Region> regions;
        for (int il = 0; il < d.L + int(d.has_mtp); ++il) {
            auto ring = serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK);
            regions.push_back({ring.first, ring.second});
            if (d.swa[il]) continue;
            // Trunk positions [0,pos); the MTP prefills pairs [0,pos-1).
            const size_t rows = size_t(pos - (il == d.L));
            const size_t first = begin ? size_t(begin - (il == d.L)) : 0;
            const size_t owned = (rows + d.S - 1 - s.g) / d.S;
            const size_t owned_before = (first + d.S - 1 - s.g) / d.S;
            regions.push_back({s.kv.at(il) + owned_before * KV_RB, (owned - owned_before) * KV_RB});
            const size_t pools_before = first / d.kpool;
            regions.push_back({s.pooled.at(il) + pools_before * d.IK,
                               (rows / d.kpool - pools_before) * d.IK * sizeof(__half)});
            auto replica = s.prefix_kv.find(il);
            if (!begin && replica != s.prefix_kv.end())
                regions.push_back({replica->second, std::min(rows, size_t(d.top_k)) * KV_RB});
        }
        regions.push_back({s.hist + begin, size_t(pos - begin) * sizeof(int)});
        if (d.has_mtp) regions.push_back({d.bsh[s.g].hprev, size_t(d.E) * sizeof(float)});
        return regions;
    }

    void reserve(Impl& d, int i, int64_t rows, int64_t begin = 0) {
        const int64_t count = rows - begin;
        if (count <= capacity_rows && delta_capacity == bool(begin)) return;
        capacity_rows = 0;  // A failed grow can leave different capacities on different shards.
        std::vector<double> alloc_ms(d.S);
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        auto report = [&](bool ok) {
            size_t bytes = 0;
            for (const auto& b : buffers) bytes += b.capacity;
            std::fprintf(stderr, "maya hy4f serve: host prefix reserve slot=%d rows=%lld bytes=%zu "
                         "cuda_alloc_max_ms=%.1f grow_wall_ms=%.1f ok=%d\n", i, (long long)rows, bytes,
                         *std::max_element(alloc_ms.begin(), alloc_ms.end()),
                         std::chrono::duration<double, std::milli>(clock::now() - start).count(), int(ok));
        };
        try {
            d.launch_parallel([&](Shard& s) {
                size_t bytes = 0;
                for (const auto& r : regions(d, s, rows, begin)) bytes += r.bytes;
                auto& b = buffers[s.g];
                if (bytes <= b.capacity) return;
                if (b.ptr) HY_CK(cudaFreeHost(b.ptr));
                b = {};
                uint8_t* ptr = nullptr;
                const auto t = clock::now();
                const auto rc = cudaHostAlloc(reinterpret_cast<void**>(&ptr), bytes, cudaHostAllocPortable);
                alloc_ms[s.g] = std::chrono::duration<double, std::milli>(clock::now() - t).count();
                if (rc != cudaSuccess) {
                    cudaGetLastError();  // Recoverable reservation failure on this launcher thread.
                    HY_CK(rc);
                }
                b = {ptr, bytes};
            });
        } catch (...) { report(false); throw; }
        capacity_rows = count;
        delta_capacity = bool(begin);
        report(true);
    }
};

bool Model::preallocate_host_prefixes(int slots, int64_t short_rows, int64_t deep_rows, std::string& err, int64_t fine_rows) {
    try {
        if (!host_prefixes_.empty() || (slots != 2 && slots != 3) || d_->bsh.empty() ||
            short_rows < 0 || deep_rows < 0 || short_rows >= max_context() || deep_rows >= max_context() ||
            fine_rows < 0 || fine_rows > 6144 || (fine_rows && (short_rows != 32768 || short_rows + fine_rows >= max_context())))
            throw std::runtime_error("invalid host prefix reservation");
        // Publish the whole reservation only when it fits. On failure the temporaries unpin
        // every successful allocation, including other shards of the incomplete slot.
        std::vector<std::shared_ptr<HostPrefix>> pending((deep_rows || fine_rows) ? 2 * slots : slots);
        if (fine_rows) pending.resize(2 * slots + 2);
        auto reserve = [&](int index, int64_t end, int64_t begin = 0) {
            if (!end) return;
            auto slot = std::make_shared<HostPrefix>(d_->S);
            slot->reserve(*d_, index, end, begin);
            pending[index] = std::move(slot);
        };
        for (int i = 0; i < slots; ++i) reserve(i, short_rows);
        for (int i = 0; i < slots; ++i) reserve(slots + i, deep_rows);
        if (fine_rows) {
            try {
                for (int i = 0; i < 2; ++i) reserve(2 * slots + i, short_rows + fine_rows, short_rows);
            } catch (const std::exception& e) {
                // Keep the established anchor reservations. Optional delta buffers can grow on demand.
                pending.resize(2 * slots);
                std::fprintf(stderr, "maya hy4f serve: fine host prealloc on-demand (%s)\n", e.what());
            }
        }
        host_prefixes_.swap(pending);
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

void Model::transfer_host_prefix(HostPrefix& slot, int64_t pos, bool save, int64_t begin) {
    Impl& d = *d_;
    d.launch_parallel([&](Shard& s) {
        auto regions = HostPrefix::regions(d, s, pos, begin);
        auto& buffer = slot.buffers[s.g];
        size_t bytes = 0;
        for (const auto& r : regions) bytes += r.bytes;
        if (!buffer.ptr || buffer.capacity < bytes) throw std::runtime_error("host prefix buffer incomplete");
        size_t offset = 0;
        try {
            for (const auto& r : regions) {
                if (save) HY_CK(cudaMemcpyAsync(buffer.ptr + offset, r.device, r.bytes, cudaMemcpyDeviceToHost, s.st));
                else HY_CK(cudaMemcpyAsync(r.device, buffer.ptr + offset, r.bytes, cudaMemcpyHostToDevice, s.st));
                offset += r.bytes;
            }
            if (!save && d.has_mtp && !d.dsh.empty())
                HY_CK(cudaMemcpyAsync(d.dsh[s.g].h, d.bsh[s.g].hprev, sizeof(float) * d.E, cudaMemcpyDeviceToDevice, s.st));
            HY_CK(cudaStreamSynchronize(s.st));
        } catch (...) {
            cudaStreamSynchronize(s.st);
            throw;
        }
    });
}

bool Model::save_host_prefix(int i, std::string& err, double* d2h_ms, int parent) {
    try {
        if (i < 0 || i > 8 || d_->pos_next < 4096 || d_->pos_next >= d_->o.max_context || d_->pos_next % 2048 || d_->bsh.empty())
            throw std::runtime_error("invalid host prefix checkpoint");
        int64_t begin = 0;
        uint64_t parent_generation = 0;
        if (parent >= 0) {
            if (parent == i || size_t(parent) >= host_prefixes_.size() || !host_prefixes_[parent])
                throw std::runtime_error("host prefix delta has no anchor");
            const auto& anchor = *host_prefixes_[parent];
            begin = anchor.pos;
            if (anchor.parent >= 0 || begin < 32768 || begin % 8192 ||
                d_->pos_next <= begin || d_->pos_next - begin > 6144)
                throw std::runtime_error("invalid host prefix delta boundary");
            parent_generation = anchor.generation;
        }
        if (host_prefixes_.size() <= size_t(i)) host_prefixes_.resize(i + 1);
        auto& slot = host_prefixes_[i];
        if (!slot) slot = std::make_shared<HostPrefix>(d_->S);
        slot->pos = 0;
        ++slot->generation;
        slot->parent = parent;
        slot->begin = begin;
        slot->parent_generation = parent_generation;
        slot->reserve(*d_, i, d_->pos_next, begin);
        const auto start = std::chrono::steady_clock::now();
        transfer_host_prefix(*slot, d_->pos_next, true, begin);
        if (d2h_ms)
            *d2h_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        slot->pos = d_->pos_next;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

bool Model::load_host_prefix(int i, int64_t pos, std::string& err) {
    try {
        if (i < 0 || size_t(i) >= host_prefixes_.size() || !host_prefixes_[i] || host_prefixes_[i]->pos != pos || pos <= 0)
            throw std::runtime_error("host prefix missing");
        const auto& slot = host_prefixes_[i];
        if (slot->parent >= 0) {
            const auto& anchor = host_prefixes_.at(slot->parent);
            if (!anchor || !anchor->pos || anchor->pos != slot->begin || anchor->generation != slot->parent_generation)
                throw std::runtime_error("host prefix delta anchor expired");
        }
        // A different branch owns the live rings now. Invalidate both old marks before the first write.
        prefix_pos_ = -1;
        if (snap_) snap_->pos = -1;
        d_->pos_next = hw_ = 0;
        d_->truncate_full_mirrors(0);
        // The host checkpoint contains the canonical rows and short dense
        // prefix. Borrowed sparse stripes belong to the previous branch.
        if (d_->sparse_limit) {
            d_->sparse_valid = false;
            d_->sparse_high = 0;
        }
        d_->packed_index_groups = -1;
        if (slot->parent >= 0)
            transfer_host_prefix(*host_prefixes_[slot->parent], slot->begin, false);
        transfer_host_prefix(*slot, pos, false, slot->begin);
        d_->pos_next = hw_ = pos;
        d_->sparse_high = (int)std::min<int64_t>(pos,d_->sparse_limit);
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

bool Model::mark_prefix(std::string& err, bool cold_aligned) {
    prefix_pos_=-1;
    try {
        if(d_->bsh.empty() || d_->pos_next<=0) return true;
        transfer_prefix(true);
        prefix_pos_=d_->pos_next;
        prefix_->cold_aligned = cold_aligned;
        return true;
    } catch(const std::exception& e) {
        err=e.what();
        return false;
    }
}

bool Model::mark(std::string& err, bool aligned_start, bool cold_aligned) {
    try {
        Impl& d = *d_;
        const int Lm = d.L + (d.has_mtp ? 1 : 0);
        if (!snap_) {
            snap_ = std::make_shared<Snap>();
            snap_->rows.resize(d.S);
            snap_->devices.reserve(d.S);
            for (auto& s : d.sh) {
                HY_CK(cudaSetDevice(s.dev));
                snap_->devices.push_back(s.dev);
                if (d.has_mtp && !d.bsh.empty() && std::getenv("HY4F_PREFIX_TURN_MARK")) {
                    void* hidden = nullptr;
                    HY_CK(cudaMalloc(&hidden, sizeof(float) * d.E));
                    snap_->rows[s.g][Lm] = hidden;
                }
                for (int il = 0; il < Lm; ++il) {
                    void* p = nullptr;
                    HY_CK(cudaMalloc(&p, serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK).second));
                    snap_->rows[s.g][il] = p;
                }
            }
        }
        for (auto& s : d.sh) {
            HY_CK(cudaSetDevice(s.dev));
            const auto hidden = snap_->rows[s.g].find(Lm);
            if (hidden != snap_->rows[s.g].end())
                HY_CK(cudaMemcpyAsync(hidden->second, d.bsh[s.g].hprev, sizeof(float)*d.E, cudaMemcpyDeviceToDevice, s.st));
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
        snap_->aligned_start = aligned_start;
        snap_->cold_aligned = cold_aligned;
        return true;
    } catch (const std::exception& e) {
        if (snap_) snap_->pos = -1;
        err = e.what();
        return false;
    }
}

int64_t Model::rewind(int64_t pos, std::string& err, bool restore_prompt, bool* cold_aligned) {
    if (cold_aligned) *cold_aligned = false;
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
        if (pos > 0 && ((restore_prompt && pos == m) || !keeps(pos, hw_))) {
            const int64_t down = has_dsa ? pos - pos % d.kpool : 0;
            const bool can_round = down > 0 && keeps(down, hw_);
            const bool earlier_turn = std::getenv("HY4F_PREFIX_TURN_MARK") && m > 0 && m < pos &&
                                      !can_round && m >= prefix_pos_ && snap_->aligned_start;
            if (earlier_turn || (m > 0 && (pos == m || pos == m - 1) && keeps(pos, m))) {
                if (earlier_turn) pos = m;
                for (auto& s : d.sh) {
                    HY_CK(cudaSetDevice(s.dev));
                    const auto hidden = snap_->rows[s.g].find(Lm);
                    if (pos == m && hidden != snap_->rows[s.g].end()) {
                        HY_CK(cudaMemcpyAsync(d.bsh[s.g].hprev, hidden->second, sizeof(float)*d.E, cudaMemcpyDeviceToDevice, s.st));
                        if (!d.dsh.empty())
                            HY_CK(cudaMemcpyAsync(d.dsh[s.g].h, hidden->second, sizeof(float)*d.E, cudaMemcpyDeviceToDevice, s.st));
                    }
                    for (int il = 0; il < Lm; ++il) {
                        const auto r = serve_ring(s, il, d.swa[il] != 0, d.win, d.kpool, d.IK);
                        HY_CK(cudaMemcpyAsync(r.first, snap_->rows[s.g][il], r.second, cudaMemcpyDeviceToDevice, s.st));
                    }
                }
                for (auto& s : d.sh) {
                    HY_CK(cudaSetDevice(s.dev));
                    HY_CK(cudaStreamSynchronize(s.st));
                }
                if (cold_aligned && pos == m && snap_->cold_aligned &&
                    (!d.has_mtp || snap_->rows[0].count(Lm)))
                    *cold_aligned = true;
                hw_ = m;                                // (the rings as they were at m)
            } else if (can_round) {
                pos = down;                             // (the indexer group's start: < kpool tokens read again)
            } else if (prefix_pos_ > 0 && prefix_pos_ <= pos) {
                transfer_prefix(false);
                pos=prefix_pos_;
                hw_=pos;
                if (cold_aligned) *cold_aligned = prefix_->cold_aligned;
            } else {
                pos = 0;
            }
        }
        if (snap_ && pos < m) snap_->pos = -1;          // (positions before the copy's are written anew)
        if (pos < prefix_pos_) prefix_pos_=-1;
        if (pos == 0) hw_ = 0;
        d.pos_next = pos;
        d.truncate_full_mirrors((int)pos);
        if (d.sparse_limit) {
            d.sparse_high = (int)std::min<int64_t>(pos,d.sparse_limit);
            if (pos == 0) d.sparse_valid = true;
        }
        return pos;
    } catch (const std::exception& e) {
        err = e.what();
        return -1;
    }
}

}  // namespace strata::hy4f
