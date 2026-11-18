// src/core/hy4f_geometry.cpp - the Hy4 flash port: the hy_v4_flash geometry reader and its tensor-shape check.
//
// Tencent Hy4 flash (tencent/Hy4-flash, open weights 18.11.2026) as llama.cpp 2cbca389f + AngelSlim's
// hy4-flash-spark.patch writes it (llm_arch "hy_v4_flash", src/models/hy-v4-flash.cpp).  The model is GLM-5.3-Flash's
// sibling more than DeepSeek's: MLA with a 64-dim rope tail on every layer (one latent geometry), 12 DSA layers with a
// k-pool indexer and 30 sliding-window layers (window 512, twice the heads, their own rope base), hc 4 streams with
// pre/post weights only, sandwich norms, a learnable per-head sink, LatentMoE (512 experts top-16 at width 1536 behind
// a shared down/up projection), sigmoid routing with the exp_probs_b bias, and Engram n-gram tables on 4 layers.
//
// Fail-closed like glm_dsa_geometry.cpp: every key the forward depends on is required and every value this port does
// not serve is refused with the key named.  Shapes in ggml ne order (dim 0 fastest).
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_model.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

namespace {

std::string dims(const std::vector<uint64_t>& s) {
    std::string r = "{";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "}";
}

std::vector<uint64_t> trimmed(std::vector<uint64_t> s) {
    while (s.size() > 1 && s.back() == 1) s.pop_back();
    return s;
}

}  // namespace

bool from_gguf_hy4_flash(const strata::GgufFile& g, Glm5Geometry& out_ref, std::string& err) {
    const std::string p = "hy_v4_flash.";
    Glm5Geometry out;
    std::map<std::string, strata::MetaValue> whole;   // arrays the reader sampled, read again whole
    const auto get = [&](const std::string& key) -> const strata::MetaValue* {
        const strata::MetaValue* v = g.get(key);
        if (v == nullptr || v->type != strata::MetaType::ARRAY || v->items.size() == v->count) return v;
        auto it = whole.find(key);
        if (it == whole.end()) {
            strata::MetaValue f;
            if (!g.get_full(key, f)) return v;
            it = whole.emplace(key, std::move(f)).first;
        }
        return &it->second;
    };
    out.arch = GlmArch::Hy4Flash;
    bool ok = true;
    const auto fail = [&](const std::string& what) {
        if (ok) err = "hy-v4-flash: " + what;
        ok = false;
        return false;
    };
    const auto num = [&](const std::string& key, double& v) {
        const strata::MetaValue* m = get(p + key);
        if (m == nullptr) return fail("required key " + p + key + " is missing");
        if (!m->is_num()) return fail(p + key + " is not a number");
        v = m->num();
        return true;
    };
    const auto cnt = [&](const std::string& key, int64_t lo, int64_t& v) {
        double d = 0;
        if (!num(key, d)) return false;
        if (d != std::floor(d) || d < (double) lo || d > 1e12)
            return fail(p + key + " = " + std::to_string(d) + " is not a valid count (>= " + std::to_string(lo) + ")");
        v = (int64_t) d;
        return true;
    };
    // a numeric array of exactly n items (n < 0: any length), read whole
    const auto arr = [&](const std::string& key, int64_t n, std::vector<double>& v) {
        const strata::MetaValue* m = get(p + key);
        if (m == nullptr) return fail("required key " + p + key + " is missing");
        if (m->type != strata::MetaType::ARRAY) return fail(p + key + " is not an array");
        if (m->items.size() != m->count) return fail(p + key + " was truncated by the reader");
        if (n >= 0 && (int64_t) m->count != n)
            return fail(p + key + " has " + std::to_string(m->count) + " entries, expected " + std::to_string(n));
        v.clear();
        for (const auto& it : m->items) {
            if (!it.is_num()) return fail(p + key + " holds a non-number");
            v.push_back(it.type == strata::MetaType::BOOL ? (double) (it.u != 0) : it.num());
        }
        return true;
    };

    int64_t block_count = 0, n_embd = 0, n_ff = 0, n_ctx = 0, hkv = 0, q_lora = 0, kv_lora = 0, key_len = 0,
            val_len = 0, key_mla = 0, val_mla = 0, n_rot = 0, n_expert = 0, n_used = 0, n_shared = 0, n_ff_exp = 0,
            dense_lead = 0, gating = 0, idx_heads = 0, idx_key = 0, idx_top_k = 0, kpool = 0, n_swa = 0, latent = 0,
            nextn = 0, hc = 0, n_vocab = 0;
    double rms_eps = 0, theta = 0, theta_swa = 0, w_scale = 0, hc_eps = 0, hc_mag = 0, w_norm = 0, tail = 0;
    cnt("block_count", 1, block_count);
    cnt("embedding_length", 1, n_embd);
    cnt("feed_forward_length", 1, n_ff);
    cnt("context_length", 1, n_ctx);
    cnt("attention.head_count_kv", 1, hkv);
    cnt("attention.q_lora_rank", 1, q_lora);
    cnt("attention.kv_lora_rank", 1, kv_lora);
    cnt("attention.key_length", 1, key_len);
    cnt("attention.value_length", 1, val_len);
    cnt("attention.key_length_mla", 1, key_mla);
    cnt("attention.value_length_mla", 1, val_mla);
    num("attention.layer_norm_rms_epsilon", rms_eps);
    cnt("rope.dimension_count", 1, n_rot);
    num("rope.freq_base", theta);
    num("rope.freq_base_swa", theta_swa);
    cnt("expert_count", 1, n_expert);
    cnt("expert_used_count", 1, n_used);
    cnt("expert_shared_count", 0, n_shared);
    cnt("expert_feed_forward_length", 1, n_ff_exp);
    cnt("leading_dense_block_count", 0, dense_lead);
    num("expert_weights_scale", w_scale);
    num("expert_weights_norm", w_norm);
    cnt("expert_gating_func", 0, gating);
    cnt("expert_latent_length", 1, latent);
    cnt("attention.indexer.head_count", 1, idx_heads);
    cnt("attention.indexer.key_length", 1, idx_key);
    cnt("attention.indexer.top_k", 1, idx_top_k);
    cnt("attention.indexer.kpool", 1, kpool);
    num("attention.indexer.kpool_select_tail", tail);
    cnt("attention.sliding_window", 1, n_swa);
    cnt("nextn_predict_layers", 0, nextn);
    cnt("hyper_connection.count", 1, hc);
    num("hyper_connection.epsilon", hc_eps);
    num("hyper_connection.magnitude", hc_mag);
    cnt("vocab_size", 1, n_vocab);
    if (!ok) return false;

    // the trunk and the NextN block (block_count 43 = 42 + 1)
    if (nextn > 1) return fail("nextn_predict_layers " + std::to_string(nextn) + ": one NextN block is served");
    out.nextn = (int) nextn;
    out.n_layers = (int) (block_count - nextn);
    if (out.n_layers < 2) return fail("block_count " + std::to_string(block_count) + " leaves no trunk");
    const int n_all = out.n_layers + out.nextn;
    out.n_embd = (int) n_embd;
    out.n_vocab = (int) n_vocab;
    out.n_ctx_train = n_ctx;
    out.norm_eps = (float) rms_eps;

    // MLA driven as MQA: one latent row [c_kv | k_pe] per token, absorbed k_b / v_b
    if (hkv != 1) return fail("attention.head_count_kv " + std::to_string(hkv) + " != 1 (MLA is one latent row)");
    out.q_lora = (int) q_lora;
    out.kv_lora = (int) kv_lora;
    out.qk_rope = (int) n_rot;
    out.qk_nope = (int) (key_mla - n_rot);
    out.v_head = (int) val_mla;
    if (out.qk_nope <= 0) return fail("attention.key_length_mla <= rope.dimension_count (no nope part)");
    if (key_len != kv_lora + n_rot) return fail("attention.key_length != kv_lora_rank + rope (the latent row)");
    if (val_len != kv_lora) return fail("attention.value_length != kv_lora_rank");
    if (n_rot % 2 != 0) return fail("rope.dimension_count is odd");
    if (!(theta > 0) || !(theta_swa > 0)) return fail("rope.freq_base / rope.freq_base_swa must be > 0");
    out.rope_theta = theta;
    out.rope_theta_swa = theta_swa;

    // the hybrid stack: per-layer heads and the sliding-window pattern (43 entries: the NextN block included)
    std::vector<double> heads, swa, itypes, clamp;
    if (!arr("attention.head_count", -1, heads) || !arr("attention.sliding_window_pattern", -1, swa) ||
        !arr("attention.indexer.types", -1, itypes) || !arr("swiglu_clamp_exp", -1, clamp))
        return false;
    for (auto* v : {&heads, &swa, &clamp})
        if ((int) v->size() != n_all && (int) v->size() != out.n_layers)
            return fail("a per-layer array has " + std::to_string(v->size()) + " entries, expected " +
                        std::to_string(out.n_layers) + " or " + std::to_string(n_all));
    out.swa_layer.assign((size_t) out.n_layers, 0);
    out.n_head = 0;
    out.n_head_swa = 0;
    for (int il = 0; il < out.n_layers; ++il) {
        const int h = (int) heads[(size_t) il];
        out.swa_layer[(size_t) il] = swa[(size_t) il] != 0 ? 1 : 0;
        int& slot = out.swa_layer[(size_t) il] ? out.n_head_swa : out.n_head;
        if (slot == 0) slot = h;
        if (slot != h)
            return fail("layer " + std::to_string(il) + " has " + std::to_string(h) + " heads, its kind has " +
                        std::to_string(slot) + " (one head count per attention kind is served)");
    }
    if (out.n_head == 0) return fail("no DSA (full-attention) layer");
    if (out.swa_layer[0]) return fail("layer 0 is a sliding-window layer (the indexer needs a DSA layer first)");
    if (nextn && swa.size() == (size_t) n_all && swa[(size_t) out.n_layers] != 0)
        return fail("the NextN block is marked sliding-window (it is a DSA layer in the reference)");
    if (nextn && heads.size() == (size_t) n_all && (int) heads[(size_t) out.n_layers] != out.n_head)
        return fail("the NextN block's head count differs from the DSA layers'");
    out.n_head_kv.assign((size_t) out.n_layers, 1);
    out.n_swa = (int) n_swa;
    for (size_t i = 0; i < clamp.size(); ++i)
        if (clamp[i] != clamp[0]) return fail("swiglu_clamp_exp differs between layers (one value is served)");
    out.swiglu_exp = (float) clamp[0];                      // routed experts only (the reference)
    out.swiglu_shexp = std::numeric_limits<float>::infinity();   // the shared and dense FFNs are not clamped

    // the indexer: on the DSA layers only, k-pool over a gated projection of the normed key
    out.idx_kind = GlmIdxKind::KPoolProj;
    out.idx_heads = (int) idx_heads;
    out.idx_key = (int) idx_key;
    out.idx_top_k = (int) idx_top_k;
    out.idx_kpool = (int) kpool;
    out.idx_select_tail = tail != 0 ? 1 : 0;
    out.idx_rope = (int) n_rot;
    if (itypes.size() != (size_t) out.n_layers && itypes.size() != (size_t) n_all)
        return fail("attention.indexer.types has " + std::to_string(itypes.size()) + " entries");
    out.indexer_full.assign((size_t) n_all, 1);
    for (int il = 0; il < out.n_layers; ++il) {
        out.indexer_full[(size_t) il] = itypes[(size_t) il] != 0 ? 1 : 0;
        if (out.indexer_full[(size_t) il] == out.swa_layer[(size_t) il])
            return fail("layer " + std::to_string(il) + ": the indexer must be on exactly the DSA layers");
    }
    if (out.idx_top_k % out.idx_kpool != 0) return fail("indexer top_k is not a multiple of kpool");

    // hyper-connections: hc streams, pre/post weights only (the hc_*_fn tensors are [hc*n_embd, 2*hc])
    out.hc = (int) hc;
    out.hc_eps = (float) hc_eps;
    out.hc_magnitude = (float) hc_mag;
    out.hc_identity = true;
    out.sinkhorn_iters = 0;
    out.sandwich_norm = true;
    out.attn_sinks = true;

    // MoE: sigmoid + exp_probs_b for the selection, normalised weights x scale; LatentMoE at expert_latent_length
    out.n_expert = (int) n_expert;
    out.n_exp_used = (int) n_used;
    out.n_shared = (int) n_shared;
    out.n_ff_exp = (int) n_ff_exp;
    out.n_ff_dense = (int) n_ff;
    out.dense_lead = (int) dense_lead;
    out.w_scale = (float) w_scale;
    out.norm_w = w_norm != 0 ? 1 : 0;
    out.n_expert_latent = (int) latent;
    if (gating != 2) return fail("expert_gating_func " + std::to_string(gating) + " is not sigmoid (2)");
    if (n_used > n_expert) return fail("expert_used_count > expert_count");
    if (dense_lead >= out.n_layers) return fail("leading_dense_block_count leaves no MoE layer");

    // Engram (llama.cpp "ple"): modules on ple.layers, multi-head hashes of 2..ngram_size-grams into one table
    std::vector<double> layers, mult, offs, vsz, tmap;
    if (!arr("ple.layers", -1, layers) || !arr("ple.layer_multipliers", -1, mult) || !arr("ple.head_offsets", -1, offs) ||
        !arr("ple.head_vocab_sizes", -1, vsz))
        return false;
    int64_t ngram = 0, eheads = 0, edim = 0, pad = 0;
    if (!cnt("ple.ngram_size", 2, ngram) || !cnt("ple.heads_per_ngram", 1, eheads) ||
        !cnt("embedding_length_per_layer_input", 1, edim) || !cnt("engram.pad_token_id", 0, pad))
        return false;
    if (const strata::MetaValue* m = g.get(p + "engram.token_map"); m != nullptr && m->type == strata::MetaType::ARRAY)
        out.engram_token_map = (int64_t) m->count;   // 120832 entries: counted, read by the forward when it exists
    else
        return fail("required key " + p + "engram.token_map is missing");
    const size_t mods = layers.size();
    const size_t heads_per_mod = (size_t) ((ngram - 1) * eheads);
    if (mods == 0) return fail("ple.layers is empty");
    if (offs.size() != mods * heads_per_mod || vsz.size() != offs.size())
        return fail("ple.head_offsets / head_vocab_sizes have " + std::to_string(offs.size()) + " entries, expected " +
                    std::to_string(mods * heads_per_mod) + " (modules x (ngram-1) x heads)");
    if (mult.size() % mods != 0) return fail("ple.layer_multipliers is not a whole number per module");
    for (double l : layers) {
        if (l < 0 || l >= out.n_layers) return fail("ple.layers holds a layer outside the trunk");
        out.engram_layers.push_back((int) l);
    }
    for (double v : mult) out.engram_mult.push_back((uint64_t) v);
    for (double v : offs) out.engram_offsets.push_back((uint64_t) v);
    for (double v : vsz) {
        if (v < 1) return fail("ple.head_vocab_sizes holds a 0 bucket count");
        out.engram_vocab.push_back((uint64_t) v);
    }
    out.engram_ngram = (int) ngram;
    out.engram_heads = (int) eheads;
    out.engram_dim = (int) edim;
    out.engram_pad = (int) pad;
    out.engram_rows = (int64_t) (out.engram_offsets.back() + out.engram_vocab.back());

    out_ref = out;
    return true;
}

bool hy4f_geometry_check(Glm5Geometry& g, const std::function<const strata::TensorInfo*(const std::string&)>& find,
                         int l0, int l1, std::string& err) {
    if (l1 <= 0) l1 = g.n_layers;
    const uint64_t E = (uint64_t) g.n_embd, V = (uint64_t) g.n_vocab, HC = (uint64_t) g.hc;
    const uint64_t QL = (uint64_t) g.q_lora, KL = (uint64_t) g.kv_lora, NOPE = (uint64_t) g.qk_nope;
    const uint64_t VH = (uint64_t) g.v_head, IH = (uint64_t) g.idx_heads, IK = (uint64_t) g.idx_key;
    const uint64_t NX = (uint64_t) g.n_expert, FX = (uint64_t) g.n_ff_exp, LT = (uint64_t) g.n_expert_latent;
    const uint64_t FS = (uint64_t) g.n_ff_exp * (uint64_t) g.n_shared, FD = (uint64_t) g.n_ff_dense;
    const uint64_t KP = (uint64_t) g.idx_kpool;
    const auto need = [&](const std::string& name, std::vector<uint64_t> want) -> bool {
        const strata::TensorInfo* t = find(name);
        if (t == nullptr) {
            err = "hy-v4-flash: tensor " + name + " is missing (expected " + dims(want) + ")";
            return false;
        }
        if (trimmed(t->shape) != trimmed(want)) {
            err = "hy-v4-flash: tensor " + name + " has shape " + dims(t->shape) + ", expected " + dims(want);
            return false;
        }
        return true;
    };
    if (!need("token_embd.weight", {E, V}) || !need("output_norm.weight", {E}) || !need("output.weight", {E, V}) ||
        !need("output_hc_fn.weight", {HC * E, HC}) || !need("output_hc_base.weight", {HC}) ||
        !need("output_hc_scale.weight", {1}))
        return false;
    if (!g.engram_layers.empty() &&
        !need("per_layer_token_embd.weight", {(uint64_t) g.engram_dim, (uint64_t) g.engram_rows}))
        return false;

    std::vector<int> layers;
    for (int il = l0; il < l1; ++il) layers.push_back(il);
    if (l1 == g.n_layers)
        for (int il = g.n_layers; il < g.n_layers + g.nextn; ++il) layers.push_back(il);
    for (int il : layers) {
        const std::string P = "blk." + std::to_string(il) + ".";
        const bool mtp = il >= g.n_layers;
        if (mtp && find(P + "nextn.eh_proj.weight") == nullptr) {
            std::fprintf(stderr, "hy-v4-flash: WARNING no NextN block (blk.%d) - no speculative decode\n", il);
            continue;
        }
        const uint64_t H = (uint64_t) g.heads(il), QH = (uint64_t) g.qk_head();
        if (!need(P + "attn_norm.weight", {E}) || !need(P + "attn_q_a_norm.weight", {QL}) ||
            !need(P + "attn_kv_a_norm.weight", {KL}) || !need(P + "attn_q_a.weight", {E, QL}) ||
            !need(P + "attn_q_b.weight", {QL, H * QH}) || !need(P + "attn_kv_a_mqa.weight", {E, (uint64_t) g.kv_row()}) ||
            !need(P + "attn_k_b.weight", {NOPE, KL, H}) || !need(P + "attn_v_b.weight", {KL, VH, H}) ||
            !need(P + "attn_gate.weight", {E, H * VH}) || !need(P + "attn_output.weight", {H * VH, E}) ||
            !need(P + "attn_sinks.weight", {H}) || !need(P + "post_attention_norm.weight", {E}) ||
            !need(P + "ffn_norm.weight", {E}) || !need(P + "post_ffw_norm.weight", {E}))
            return false;
        if (!mtp)
            for (const char* k : {"hc_attn", "hc_ffn"})
                if (!need(P + k + "_fn.weight", {HC * E, 2 * HC}) || !need(P + k + "_base.weight", {2 * HC}) ||
                    !need(P + k + "_scale.weight", {2}))
                    return false;
        if (g.idx_full(il)) {
            if (!need(P + "indexer.attn_k.weight", {E, IK}) || !need(P + "indexer.attn_q_b.weight", {QL, IH * IK}) ||
                !need(P + "indexer.k_norm.weight", {IK}) || !need(P + "indexer.proj.weight", {E, IH}) ||
                !need(P + "indexer_compressor_ape.weight", {IK, KP}) ||
                !need(P + "indexer_compressor_gate.weight", {IK, IK}) ||
                !need(P + "indexer_compressor_norm.weight", {IK}) ||
                !need(P + "indexer_compressor_value.weight", {IK, IK}))
                return false;
        } else if (find(P + "indexer.attn_k.weight") != nullptr) {
            err = "hy-v4-flash: sliding-window layer " + std::to_string(il) + " carries indexer tensors";
            return false;
        }
        if (il < g.dense_lead) {
            if (!need(P + "ffn_gate.weight", {E, FD}) || !need(P + "ffn_up.weight", {E, FD}) ||
                !need(P + "ffn_down.weight", {FD, E}))
                return false;
        } else {
            if (!need(P + "ffn_gate_inp.weight", {E, NX}) || !need(P + "exp_probs_b.bias", {NX}) ||
                !need(P + "ffn_routed_down.weight", {E, LT}) || !need(P + "ffn_routed_up.weight", {LT, E}) ||
                !need(P + "ffn_gate_exps.weight", {LT, FX, NX}) || !need(P + "ffn_up_exps.weight", {LT, FX, NX}) ||
                !need(P + "ffn_down_exps.weight", {FX, LT, NX}))
                return false;
            if (g.n_shared > 0 && (!need(P + "ffn_gate_shexp.weight", {E, FS}) ||
                                   !need(P + "ffn_up_shexp.weight", {E, FS}) || !need(P + "ffn_down_shexp.weight", {FS, E})))
                return false;
        }
        bool engram = false;
        for (int el : g.engram_layers) engram |= el == il;
        if (engram) {
            const uint64_t ED = (uint64_t) g.engram_dim, NH = (uint64_t) ((g.engram_ngram - 1) * g.engram_heads);
            if (!need(P + "engram_lookup_norm.weight", {ED, NH}) || !need(P + "engram_kv.weight", {ED * NH, E * (HC + 1)}) ||
                !need(P + "engram_kv_b.weight", {E, HC + 1}) || !need(P + "engram_gate.weight", {E, HC}) ||
                !need(P + "engram_hc_scale.weight", {E, HC}) || !need(P + "engram_pre_gamma.weight", {E}))
                return false;
        }
        if (mtp && (!need(P + "nextn.eh_proj.weight", {2 * E, E}) || !need(P + "nextn.enorm.weight", {E}) ||
                    !need(P + "nextn.hnorm.weight", {E}) || !need(P + "nextn.shared_head_norm.weight", {E})))
            return false;
    }
    return true;
}

std::string hy4f_geometry_summary(const Glm5Geometry& g) {
    int swa = 0, full = 0;
    for (int il = 0; il < g.n_layers; ++il) (g.is_swa(il) ? swa : full) += 1;
    char buf[768];
    std::snprintf(buf, sizeof buf,
                  "%d layers + %d NextN, n_embd %d, vocab %d; %d DSA layers (%d heads, theta %.0f, indexer %dx%d top %d "
                  "kpool %d) + %d SWA layers (%d heads, window %d, theta %.0f); MLA q_lora %d kv_lora %d nope %d rope %d "
                  "v %d; hc %d (identity comb); MoE %d x %d top-%d + %d shared, latent %d, dense lead %d (ff %d), scale "
                  "%.3f; Engram on %zu layers, %d heads x %d dims, %lld rows",
                  g.n_layers, g.nextn, g.n_embd, g.n_vocab, full, g.n_head, g.rope_theta, g.idx_heads, g.idx_key,
                  g.idx_top_k, g.idx_kpool, swa, g.n_head_swa, g.n_swa, g.rope_theta_swa, g.q_lora, g.kv_lora,
                  g.qk_nope, g.qk_rope, g.v_head, g.hc, g.n_expert, g.n_ff_exp, g.n_exp_used, g.n_shared,
                  g.n_expert_latent, g.dense_lead, g.n_ff_dense, g.w_scale, g.engram_layers.size(),
                  (g.engram_ngram - 1) * g.engram_heads, g.engram_dim, (long long) g.engram_rows);
    return buf;
}

}  // namespace strata::core
