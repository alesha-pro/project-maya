// src/hy4f/hy4f_dec.inc - maya-hy4f: the multi-token decode step (T <= DT tokens: the speculative verify) and the MTP
// draft layer (blk.L, "nextn"), both captured as one CUDA graph per shard per T.  Included inside Model::Impl.
//
// Verify(T): the trunk over tokens toks[0..T) at positions pos .. pos+T-1 (writes their KV; a rejected tail is simply
// overwritten later: every cache is position-indexed, the SWA ring has RING_PAD spare rows), logits of every row and
// the rows' final hidden states h (hc head + output norm) kept on the device for the MTP.
// MTP(T): the draft layer over (h[t], next token[t]) at positions mpos .. mpos+T-1 (its own DSA KV), the draft token =
// argmax of the head over the last row.  llama.cpp hy-v4-flash.cpp graph_mtp: concat(RMS(emb) enorm, RMS(h) hnorm) ->
// eh_proj -> an ordinary sparse layer with a plain residual -> shared_head_norm -> the trunk's output head.

static constexpr int DT = 4;

struct IoT {   // a multi-token step's inputs (pinned host -> device at the head of each graph)
    int pos;
    int n;
    int toks[8];
};

struct DShard {
    IoT *dio_v = nullptr, *dio_m = nullptr;
    float *X = nullptr, *x = nullptr, *h = nullptr, *post = nullptr, *hcp = nullptr, *hcs = nullptr, *ones = nullptr;
    float *xp = nullptr, *qr = nullptr, *qp = nullptr, *qabs = nullptr, *o = nullptr, *part = nullptr, *a = nullptr,
          *ao = nullptr, *ikn = nullptr, *ivg = nullptr, *iw = nullptr, *score = nullptr;
    int *cells = nullptr, *eids = nullptr, *eloc = nullptr;
    float *lg = nullptr, *ew = nullptr, *xm = nullptr, *ga16 = nullptr, *dn16 = nullptr, *moe = nullptr, *mu = nullptr,
          *shh = nullptr, *sh = nullptr, *ga = nullptr, *ffo = nullptr;
    int64_t* eg_ids = nullptr;
    float *eg_r = nullptr, *eg_kv = nullptr, *logits = nullptr, *argv = nullptr;
    int* argi = nullptr;
    float *me = nullptr, *cat = nullptr, *mr = nullptr, *mh = nullptr;
    Q81 *xq = nullptr, *xlq = nullptr, *q16 = nullptr;
    float* ar_buf[2] = {nullptr, nullptr};
    float* cb_buf[2] = {nullptr, nullptr};
    int *ar_flag = nullptr, *cb_flag = nullptr, *ar_ctr = nullptr, *cb_ctr = nullptr;
    Peers ar{}, cb{};
    cudaGraphExec_t gv[DT + 1] = {}, gm[DT + 1] = {};
};
std::vector<DShard> dsh;
bool has_mtp = false;
IoT *hio_v = nullptr, *hio_m = nullptr;
float* hlogT = nullptr;    // [DT][V] pinned
float* hargv = nullptr;    // [S][DT] pinned (per shard max)
int* hargi = nullptr;
float* hmargv = nullptr;   // [S] (MTP draft)
int* hmargi = nullptr;
int64_t m_next = 0;        // next MTP position

void dec_alloc(Shard& s, DShard& d) {
    HY_CK(cudaSetDevice(s.dev));
    const int K = n_used, QH = nope + rope, Hmax = std::max(Hd, Hs), ncol = (eg_ngram - 1) * eg_heads;
    const int64_t C = o.max_context;
    const size_t xpl = ((size_t) q_lora + KV_ROW + (size_t) Hmax / S * vh + IK + 3) / 4 * 4;
    const size_t qpl = std::max((size_t) Hd * QH + (size_t) IH * IK, (size_t) Hs / S * QH);
    const size_t xml = (size_t) latent + 2 * n_ffx / S;
    const int Ha = std::max(Hd, Hs / S), Ho = S > 1 ? Hmax / S : Hmax;
    auto qa = [&](size_t n) {
        Q81* p = nullptr;
        HY_CK(cudaMalloc(&p, sizeof(Q81) * ((n + 31) / 32)));
        return p;
    };
    auto ia = [&](size_t n) {
        int* p = nullptr;
        HY_CK(cudaMalloc(&p, sizeof(int) * n));
        HY_CK(cudaMemset(p, 0, sizeof(int) * n));
        return p;
    };
    HY_CK(cudaMalloc(&d.dio_v, sizeof(IoT)));
    HY_CK(cudaMalloc(&d.dio_m, sizeof(IoT)));
    HY_CK(cudaMemset(d.dio_v, 0, sizeof(IoT)));
    HY_CK(cudaMemset(d.dio_m, 0, sizeof(IoT)));
    d.X = dalloc((size_t) DT * hc * E);
    d.x = dalloc((size_t) DT * E);
    d.h = dalloc((size_t) DT * E);
    d.post = dalloc((size_t) DT * 8);
    d.hcp = dalloc((size_t) DT * 8 * HC_SPLIT);
    d.hcs = dalloc((size_t) DT * HC_SPLIT);
    {
        std::vector<float> one((size_t) DT * 8, 1.0f);
        d.ones = dalloc(one.size());
        HY_CK(cudaMemcpy(d.ones, one.data(), sizeof(float) * one.size(), cudaMemcpyHostToDevice));
    }
    d.xp = dalloc((size_t) DT * xpl);
    d.qr = dalloc((size_t) DT * q_lora);
    d.qp = dalloc((size_t) DT * qpl);
    d.qabs = dalloc((size_t) DT * Ha * kvl);
    d.o = dalloc((size_t) DT * Ho * kvl);
    d.part = dalloc((size_t) DT * Hd * CBW());
    d.a = dalloc((size_t) DT * Hmax / S * vh);
    d.ao = dalloc((size_t) DT * E);
    d.ikn = dalloc((size_t) DT * IK);
    d.ivg = dalloc((size_t) DT * 2 * IK);
    d.iw = dalloc((size_t) DT * IH);
    d.score = dalloc((size_t) DT * (C / kpool + 1));
    d.cells = ia((size_t) DT * n_cap);
    d.lg = dalloc((size_t) DT * n_exp);
    d.ew = dalloc((size_t) DT * K);
    d.eids = ia((size_t) DT * K);
    d.eloc = ia((size_t) DT * K);
    d.xm = dalloc((size_t) DT * xml);
    d.ga16 = dalloc((size_t) DT * K * 2 * n_ffx);
    d.dn16 = dalloc((size_t) DT * K * latent);
    d.moe = dalloc((size_t) DT * latent);
    d.mu = dalloc((size_t) DT * E);
    d.shh = dalloc((size_t) DT * std::max(n_ffx / S, n_ff / S));
    d.sh = dalloc((size_t) DT * E);
    d.ga = dalloc((size_t) DT * 2 * n_ff / S);
    d.ffo = dalloc((size_t) DT * E);
    HY_CK(cudaMalloc(&d.eg_ids, sizeof(int64_t) * DT * ncol));
    d.eg_r = dalloc((size_t) DT * ncol * eg_D);
    d.eg_kv = dalloc((size_t) DT * (hc + 1) * E);
    d.logits = dalloc((size_t) DT * (V / S));
    d.argv = dalloc(DT);
    d.argi = ia(DT);
    d.me = dalloc((size_t) DT * E);
    d.cat = dalloc((size_t) DT * 2 * E);
    d.mr = dalloc((size_t) DT * E);
    d.mh = dalloc((size_t) DT * E);
    d.xq = qa((size_t) DT * std::max({2 * E, Hmax / S * vh, n_ff / S, ncol * eg_D, latent}));
    d.xlq = qa((size_t) DT * K * latent);
    d.q16 = qa((size_t) DT * K * n_ffx);
    for (int p = 0; p < 2; ++p) {
        d.ar_buf[p] = dalloc((size_t) S * DT * E);
        d.cb_buf[p] = dalloc((size_t) S * DT * (Hd / S) * CBW());
    }
    d.ar_flag = ia(4);
    d.cb_flag = ia(4);
    d.ar_ctr = ia(1);
    d.cb_ctr = ia(1);
}

void dec_mmvq(Shard& s, const std::string& n, const Q81* xq, float* y, int T) {
    const DW& w = s.w.at(n);
    if (w.type != T_Q8_0) throw std::runtime_error("hy4f: " + n + " is not Q8_0");
    kn::native_q8_0_mmvq(w.d, xq, y, (int) w.ne[0], (int) (w.ne[1] * w.ne[2]), T, s.st);
}
int rows_of(Shard& s, const std::string& n) {
    const DW& w = s.w.at(n);
    return (int) (w.ne[1] * w.ne[2]);
}

void d_ar(Shard& s, DShard& d, const float* x1, const float* x2, const std::string& norm, int T, bool mtp) {
    kd_ar_post<<<1, 1024, sizeof(float) * T * E, s.st>>>(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, f32(s, norm), eps,
                                                          mtp ? d.mr : d.X, mtp ? d.ones : d.post, mtp ? 1 : hc);
}
void d_norm_in(Shard& s, DShard& d, const std::string& P, const std::string& k, const std::string& norm, int n_mix,
               float mag, int T) {
    kb_hc_mix<<<dim3(HC_SPLIT, T), 256, 0, s.st>>>(d.X, f32(s, P + k + "_fn.weight"), hc * E, n_mix, d.hcp, d.hcs);
    kb_hc_apply_norm<<<T, 1024, 0, s.st>>>(d.X, d.hcp, d.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                           f32(s, norm), E, hc, n_mix, eps, hc_eps, mag, d.x, d.post);
}

void d_engram(Shard& s, DShard& d, int e, int il, const int* pp, int T) {
    const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
    const std::string P = "blk." + std::to_string(il) + ".";
    kb_engram_hash<<<T, 32, 0, s.st>>>(s.hist, pp, s.eg_map, (int64_t) eg_map.size(), eg_pad, s.eg_mult + (size_t) e * ng,
                                       s.eg_vsz + (size_t) e * ncol, s.eg_off + (size_t) e * ncol, ng, nh, d.eg_ids);
    kb_engram_rows<<<dim3(ncol, T), 128, 0, s.st>>>(eg_table, d.eg_ids, f32(s, P + "engram_lookup_norm.weight"), d.eg_r,
                                                    eg_D, ncol, 1.1920929e-07f);
    k_q8<<<nb(T * ncol * eg_D), 256, 0, s.st>>>(d.eg_r, T * ncol * eg_D, d.xq);
    dec_mmvq(s, P + "engram_kv.weight", d.xq, d.eg_kv, T);
    kb_engram_mix<<<T, 1024, 0, s.st>>>(d.X, d.X, d.eg_kv, f32(s, P + "engram_kv_b.weight"), f32(s, P + "engram_gate.weight"),
                                        f32(s, P + "engram_hc_scale.weight"), f32(s, P + "engram_pre_gamma.weight"), E, hc,
                                        1.1920929e-07f);
}

void d_attention(Shard& s, DShard& d, int il, const int* pp, int T, bool mtp) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const bool sw = swa[il] != 0;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const int xpc = rows_of(s, P + "xproj"), qpc = rows_of(s, P + "qproj");
    const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
    if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "attn_norm.weight"), E, eps, d.x, E);
    else d_norm_in(s, d, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
    k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    dec_mmvq(s, P + "xproj", d.xq, d.xp, T);
    kb_rms<<<T, 1024, 0, s.st>>>(d.xp + oq, xpc, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, d.qr, q_lora);
    kb_kv_prep<<<T, 576, 0, s.st>>>(d.xp + okv, xpc, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th, s.kv[il], pp,
                                    sw ? 3 : 2, S, s.g, win + RING_PAD);
    k_q8<<<nb(T * q_lora), 256, 0, s.st>>>(d.qr, T * q_lora, d.xq);
    dec_mmvq(s, P + "qproj", d.xq, d.qp, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const int Ha = sw ? Hl : Hd;
    if (kb.type == T_Q8_0)
        kb_absorb_q8<<<dim3(nb(Ha * kvl), T), 256, 0, s.st>>>((const uint8_t*) kb.d, d.qp, qpc, QH, d.qabs, nope, kvl, Ha);
    else
        kb_absorb_bf16<<<dim3(nb(Ha * kvl), T), 256, 0, s.st>>>((const uint16_t*) kb.d, d.qp, qpc, QH, d.qabs, nope, kvl, Ha);
    if (sw) {
        kb_attn2<<<dim3(Hl, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, nullptr, 0, pp, win, 1, 0,
                                                f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, d.o, nullptr, Hl);
    } else {
        float* iq = d.qp + (size_t) Hd * QH;
        kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, s.st>>>(iq, qpc, IH, IK, IK - rope, rope, pp, th);
        kb_rms<<<T, 128, 0, s.st>>>(d.xp + oik, xpc, f32(s, P + "indexer.k_norm.weight"), IK, eps, d.ikn, IK);
        const DW& ivw = s.w.at(P + "ivg");
        kb_q8_gemv<<<dim3(nb(2 * IK * 32), T), 256, 0, s.st>>>((const uint8_t*) ivw.d, d.ikn, IK, d.ivg, 2 * IK, IK, 2 * IK);
        kb_pool_groups<<<T / kpool + 1, 128, 0, s.st>>>(d.ivg, s.ring[il], pp, T, f32(s, P + "indexer_compressor_ape.weight"),
                                                        f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK,
                                                        kpool, rope, eps, th);
        kb_ring_chunk<<<kpool, 256, 0, s.st>>>(d.ivg, s.ring[il], T, pp, IK, kpool);
        kb_f32_gemv<<<dim3(nb(IH * 32), T), 256, 0, s.st>>>(f32(s, P + "indexer.proj.weight"), d.x, E, d.iw, IH, E, IH,
                                                            1.0f / std::sqrt((float) (IK * IH)));
        const int ld_sc = (int) (o.max_context / kpool + 1);
        kb_score<<<dim3(nb(ld_sc * 32), T), 256, 0, s.st>>>(iq, qpc, s.pooled[il], d.iw, IH, IK, IH, kpool, top_k, pp, 0,
                                                            d.score, ld_sc);
        kb_select_kpool<1024><<<T, 1024, 0, s.st>>>(d.score, ld_sc, pp, 0, kpool, top_k, top_k / kpool - tail, n_cap, d.cells);
        if (S > 1) {
            kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp, win,
                                                    S, s.g, nullptr, scale, nullptr, d.part, Hd);
            kd_combine<<<1, 1024, 0, s.st>>>(d.part, T, Hd, kvl, CBW(), s.g * Hl, Hl, f32(s, P + "attn_sinks.weight"), d.cb,
                                             s.g, S, d.cb_ctr, d.o);
        } else {
            kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp, win,
                                                    1, 0, f32(s, P + "attn_sinks.weight"), scale, d.o, nullptr, Hd);
        }
    }
    if (vb.type == T_Q8_0)
        kb_vb_gate_q8<<<dim3(nb(Hl * vh * 32), T), 256, 0, s.st>>>((const uint8_t*) vb.d, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl);
    else
        kb_vb_gate_bf16<<<dim3(nb(Hl * vh * 32), T), 256, 0, s.st>>>((const uint16_t*) vb.d, d.o, d.xp + og, xpc, d.a, kvl, vh,
                                                                     Hl);
    k_q8<<<nb(T * Hl * vh), 256, 0, s.st>>>(d.a, T * Hl * vh, d.xq);
    dec_mmvq(s, P + "attn_output.weight", d.xq, d.ao, T);
    d_ar(s, d, d.ao, nullptr, P + "post_attention_norm.weight", T, mtp);
}

void d_ffn(Shard& s, DShard& d, int il, int T, bool mtp) {
    const std::string P = "blk." + std::to_string(il) + ".";
    if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, d.x, E);
    else d_norm_in(s, d, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag, T);
    k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    if (il < dense_lead) {
        const int fl = n_ff / S;
        dec_mmvq(s, P + "dense_gu", d.xq, d.ga, T);
        kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(d.ga, 2 * fl, fl, T, d.shh);
        k_q8<<<nb(T * fl), 256, 0, s.st>>>(d.shh, T * fl, d.xq);
        dec_mmvq(s, P + "ffn_down.weight", d.xq, d.ffo, T);
        d_ar(s, d, d.ffo, nullptr, P + "post_ffw_norm.weight", T, mtp);
        return;
    }
    const int K = n_used, e0 = ebound[s.g], e1 = ebound[s.g + 1], fl = n_ffx / S;
    const int xmc = rows_of(s, P + "xmoe");
    kb_f32_gemv<<<dim3(nb(n_exp * 32), T), 256, 0, s.st>>>(f32(s, P + "ffn_gate_inp.weight"), d.x, E, d.lg, n_exp, E, n_exp,
                                                           1.0f);
    kn::router_sigmoid(d.lg, f32(s, P + "exp_probs_b.bias"), T, n_exp, K, w_scale, true, d.eids, d.ew, s.st);
    kd_local_ids<<<1, 128, 0, s.st>>>(d.eids, d.eloc, T * K, e0, e1);
    dec_mmvq(s, P + "xmoe", d.xq, d.xm, T);
    kd_q8_slots<<<dim3(nb(latent), T * K), 256, 0, s.st>>>(d.xm, xmc, latent, K, d.xlq);
    auto ids_mmvq = [&](const DW& w, const void* xq, float* y, int n_in, int n_out) {
        const size_t eb = w.rb * (size_t) w.ne[1];
        if (w.type == T_Q3_K) kn::native_q3_k_mmvq_ids(w.d, eb, d.eloc, T * K, xq, false, y, n_in, n_out, s.st);
        else if (w.type == T_Q4_K) kn::native_q4_k_mmvq_ids(w.d, eb, d.eloc, T * K, xq, false, y, n_in, n_out, s.st);
        else throw std::runtime_error("hy4f: expert type " + std::to_string(w.type));
    };
    ids_mmvq(s.w.at(P + "ffn_gu_exps"), d.xlq, d.ga16, latent, 2 * n_ffx);
    k_swiglu_gu_q8<<<nb(T * K * n_ffx), 256, 0, s.st>>>(d.ga16, n_ffx, T * K * n_ffx, clampv, d.q16);
    ids_mmvq(s.w.at(P + "ffn_down_exps.weight"), d.q16, d.dn16, n_ffx, latent);
    kd_moe_sum<<<dim3(nb(latent), T), 256, 0, s.st>>>(d.moe, d.dn16, d.ew, d.eloc, K, latent);
    k_q8<<<nb(T * latent), 256, 0, s.st>>>(d.moe, T * latent, d.xq);
    dec_mmvq(s, P + "ffn_routed_up.weight", d.xq, d.mu, T);
    kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(d.xm + latent, xmc, fl, T, d.shh);
    k_q8<<<nb(T * fl), 256, 0, s.st>>>(d.shh, T * fl, d.xq);
    dec_mmvq(s, P + "ffn_down_shexp.weight", d.xq, d.sh, T);
    d_ar(s, d, d.mu, d.sh, P + "post_ffw_norm.weight", T, mtp);
}

void head_rows(Shard& s, DShard& d, const float* x, int T, float* argv, int* argi) {
    const DW& w = s.w.at("output.weight");
    if (w.type != T_BF16) throw std::runtime_error("hy4f: output.weight is not BF16");
    const int rows = V / S;
    const uint16_t* W = (const uint16_t*) w.d;
    switch (T) {
    case 1: kd_bf16_gemv<1><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    case 2: kd_bf16_gemv<2><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    case 3: kd_bf16_gemv<3><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    default: kd_bf16_gemv<4><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    }
    kd_argmax<<<T, 1024, 0, s.st>>>(d.logits, rows, s.g * rows, argv, argi);
}

void verify_ops(Shard& s, DShard& d, int T) {
    HY_CK(cudaMemcpyAsync(d.dio_v, hio_v, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &d.dio_v->pos;
    const DW& emb = s.w.at("token_embd.weight");
    kb_embed<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, d.dio_v->toks, pp, d.X, d.X, E, hc, s.hist);
    for (int il = 0; il < L; ++il) {
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) d_engram(s, d, (int) e, il, pp, T);
        d_attention(s, d, il, pp, T, false);
        d_ffn(s, d, il, T, false);
    }
    d_norm_in(s, d, "", "output_hc", "output_norm.weight", hc, 0.0f, T);
    HY_CK(cudaMemcpyAsync(d.h, d.x, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s.st));
    head_rows(s, d, d.x, T, d.argv, d.argi);
    const int rows = V / S;
    HY_CK(cudaMemcpy2DAsync(hlogT + (size_t) s.g * rows, sizeof(float) * V, d.logits, sizeof(float) * rows,
                            sizeof(float) * rows, T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargv + (size_t) s.g * DT, d.argv, sizeof(float) * T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargi + (size_t) s.g * DT, d.argi, sizeof(int) * T, cudaMemcpyDeviceToHost, s.st));
}

void mtp_ops(Shard& s, DShard& d, int T) {
    HY_CK(cudaMemcpyAsync(d.dio_m, hio_m, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &d.dio_m->pos;
    const std::string P = "blk." + std::to_string(L) + ".";
    const DW& emb = s.w.at("token_embd.weight");
    kd_embed_rows<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, d.dio_m->toks, E, d.me);
    kb_rms<<<T, 1024, 0, s.st>>>(d.me, E, f32(s, P + "nextn.enorm.weight"), E, eps, d.cat, 2 * E);
    kb_rms<<<T, 1024, 0, s.st>>>(d.h, E, f32(s, P + "nextn.hnorm.weight"), E, eps, d.cat + E, 2 * E);
    k_q8<<<nb(T * 2 * E), 256, 0, s.st>>>(d.cat, T * 2 * E, d.xq);
    dec_mmvq(s, P + "nextn.eh_proj.weight", d.xq, d.mr, T);
    d_attention(s, d, L, pp, T, true);
    d_ffn(s, d, L, T, true);
    kb_rms<<<1, 1024, 0, s.st>>>(d.mr + (size_t) (T - 1) * E, E, f32(s, P + "nextn.shared_head_norm.weight"), E, eps, d.mh, E);
    head_rows(s, d, d.mh, 1, d.argv + DT - 1, d.argi + DT - 1);
    HY_CK(cudaMemcpyAsync(hmargv + s.g, d.argv + DT - 1, sizeof(float), cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hmargi + s.g, d.argi + DT - 1, sizeof(int), cudaMemcpyDeviceToHost, s.st));
}

void dec_setup() {
    HY_CK(cudaHostAlloc((void**) &hio_v, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hio_m, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hlogT, sizeof(float) * DT * V, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargv, sizeof(float) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargi, sizeof(int) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargv, sizeof(float) * S, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargi, sizeof(int) * S, cudaHostAllocPortable));
    std::memset(hio_v, 0, sizeof(IoT));
    std::memset(hio_m, 0, sizeof(IoT));
    dsh.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) dec_alloc(sh[i], dsh[i]);
    Peers ar{}, cb{};
    for (int j = 0; j < S; ++j) {
        ar.buf[j][0] = dsh[j].ar_buf[0];
        ar.buf[j][1] = dsh[j].ar_buf[1];
        ar.flag[j] = dsh[j].ar_flag;
        cb.buf[j][0] = dsh[j].cb_buf[0];
        cb.buf[j][1] = dsh[j].cb_buf[1];
        cb.flag[j] = dsh[j].cb_flag;
    }
    for (auto& d : dsh) d.ar = ar, d.cb = cb;
    kn::native_mmvq_set_multi_exact(false);   // (one weight read for the T columns; set before the captures)
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaFuncSetAttribute(kd_ar_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) (sizeof(float) * DT * E)));
        if (std::getenv("HY4F_DEC_EAGER")) {   // (debug: one eager verify per T with a check after it)
            for (int T = 1; T <= DT; ++T) {
                hio_v->pos = 0;
                hio_v->n = T;
                launch_parallel([&](Shard& q) { verify_ops(q, dsh[q.g], T); HY_CK(cudaStreamSynchronize(q.st)); });
                HY_CK(cudaGetLastError());
            }
        }
        DShard& d = dsh[s.g];
        for (int T = 1; T <= DT; ++T) {
            cudaGraph_t gr = nullptr;
            HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
            verify_ops(s, d, T);
            HY_CK(cudaStreamEndCapture(s.st, &gr));
            HY_CK(cudaGraphInstantiate(&d.gv[T], gr, 0));
            HY_CK(cudaGraphDestroy(gr));
            if (has_mtp) {
                HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                mtp_ops(s, d, T);
                HY_CK(cudaStreamEndCapture(s.st, &gr));
                HY_CK(cudaGraphInstantiate(&d.gm[T], gr, 0));
                HY_CK(cudaGraphDestroy(gr));
            }
        }
        HY_CK(cudaDeviceSynchronize());
    }
}

void dec_launch(bool mtp, int T) {
    launch_parallel([&](Shard& s) { HY_CK(cudaGraphLaunch(mtp ? dsh[s.g].gm[T] : dsh[s.g].gv[T], s.st)); });
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaStreamSynchronize(s.st));
    }
    HY_CK(cudaGetLastError());
}
