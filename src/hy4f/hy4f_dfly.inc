// src/hy4f/hy4f_dfly.inc - maya-hy4f: the DFly2 drafter of Hy4 flash (tencent/Hy4-flash model.dfly_layers.*, exported
// by scripts/dfly_to_gguf.py).  Included inside Model::Impl after hy4f_dec.inc.
//
// The math (vLLM PR #68296 vllm/models/hy_v4/nvidia/dfly*.py, SGLang PR #51159; docs/DFLY2-RESEARCH.md):
//  * a block of B = 6 rows at positions anchor .. anchor+5: [bonus, mask x 5] (mask = token 120009); anchor = the
//    position of the bonus token (the token after the last accepted one), the "source row" = the verify row of the last
//    accepted token;
//  * row 0 = emb(bonus) + gate * eh_proj([RMS(emb(bonus)) enorm, RMS(h) hnorm]), h = the target's final normed hidden
//    of the source row (our d.h); rows 1..5 = emb(mask);
//  * 5 decoder layers, single residual stream: h += post_proj_norm(attn(input_norm(h))); h += mlp(ffn_norm(h)) (dense
//    SiLU MLP 9216, no post-FFN norm).  Attention = the target's DSA layer form (MLA absorbed, 32 heads, learnable sink,
//    sigmoid output gate) with the drafter's own projections, over the latent KV cache of target layer tl[i]
//    (25/29/33/37/41): the block's own latent rows are written there at anchor .. anchor+5 (the next verify rewrites
//    them) and each row attends the target's top-k cells of the source row at that layer (snapshotted by the verify)
//    plus the block rows up to itself;
//  * z = final_norm(h); then sequentially for k = 0..4: corrected = z[k+1] + down(silu(g) * u), [g|u] = gu([RMS(z[k+1])
//    hidden_norm, RMS(emb(prev)) embed_norm]), prev = bonus, then the previous draft; draft k = argmax(lm_head(corrected)).
// The block (+ correction 0) is one CUDA graph per shard, each later correction step another (the argmax over the
// vocabulary slices goes through the host: 5 small syncs per draft block).

static constexpr int DF_B = 6, DF_N = 5;
int df_nl = 0, df_mask = 120009, df_ncd = 0, df_ff = 0;
std::vector<int> df_tl;   // the target layer each draft layer reads (KV + top-k)
IoT *hio_d = nullptr, *hio_c = nullptr;
float* df_hargv = nullptr;   // [S][DF_N] pinned
int* df_hargi = nullptr;
float* df_hlog = nullptr;    // [DF_N][V] pinned: the corrected rows' logits (when df_logits)
bool df_logits = false;

void load_dfly_meta() {
    const strata::GgufFile& f = *files.back();
    const strata::MetaValue* a = f.get("general.architecture");
    if (!a || a->s != "hy_v4_flash_dfly") throw std::runtime_error("hy4f: not a DFly GGUF (scripts/dfly_to_gguf.py)");
    df_nl = (int) f.get("dfly.layer_count")->u;
    df_mask = (int) f.get("dfly.mask_token_id")->u;
    if ((int) f.get("dfly.block_size")->u != DF_B) throw std::runtime_error("hy4f: DFly block size is not 6");
    strata::MetaValue v;
    if (!f.get_full("dfly.target_layers", v)) throw std::runtime_error("hy4f: DFly GGUF without target layers");
    df_tl.clear();
    for (const auto& it : v.items) df_tl.push_back((int) it.u);
    if ((int) df_tl.size() != df_nl) throw std::runtime_error("hy4f: DFly target layers / layer count");
    df_slot.assign(L, -1);
    for (int j = 0; j < df_nl; ++j) {
        if (df_tl[j] < 0 || df_tl[j] >= L || swa[df_tl[j]]) throw std::runtime_error("hy4f: DFly target layer is not a DSA layer");
        df_slot[df_tl[j]] = j;
    }
    const strata::GgufFile* ff = nullptr;
    df_ff = (int) T("dfly.0.ffn_gate.weight", &ff)->shape[1];
    if (df_ff % S) throw std::runtime_error("hy4f: DFly FFN width does not split");
    df_ncd = n_cap + 8;
}

void load_dfly_shard(Shard& s) {
    HY_CK(cudaSetDevice(s.dev));
    const int g = s.g, Hl = Hd / S, h0 = g * Hl;
    for (int i = 0; i < df_nl; ++i) {
        const std::string P = "dfly." + std::to_string(i) + ".";
        for (const char* n : {"attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                              "post_attention_norm.weight", "attn_sinks.weight"})
            up_full(s, P + n);
        up_concat(s, P + "xproj", {{P + "attn_q_a.weight", 0, -1}, {P + "attn_kv_a_mqa.weight", 0, -1},
                                   {P + "attn_gate.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh}});
        up_concat(s, P + "qproj", {{P + "attn_q_b.weight", 0, -1}});
        up_full(s, P + "attn_k_b.weight");
        up_rows(s, P + "attn_v_b.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh);
        up_cols(s, P + "attn_output.weight", (int64_t) h0 * vh, (int64_t) (h0 + Hl) * vh);
        const int64_t f0 = (int64_t) g * df_ff / S, f1 = (int64_t) (g + 1) * df_ff / S;
        up_concat(s, P + "dense_gu", {{P + "ffn_gate.weight", f0, f1}, {P + "ffn_up.weight", f0, f1}});
        up_cols(s, P + "ffn_down.weight", f0, f1);
    }
    for (const char* n : {"dfly.fuse.enorm.weight", "dfly.fuse.hnorm.weight", "dfly.fuse.gate.weight", "dfly.fuse.eh_proj.weight",
                          "dfly.corr.hidden_norm.weight", "dfly.corr.embed_norm.weight", "dfly.corr.down.weight",
                          "dfly.final_norm.weight"})
        up_full(s, n);
    up_concat(s, "dfly.corr.gu", {{"dfly.corr.gate.weight", 0, -1}, {"dfly.corr.up.weight", 0, -1}});
}

void df_alloc(Shard& s, DFShard& f) {
    HY_CK(cudaSetDevice(s.dev));
    auto ia = [&](size_t n) {
        int* p = nullptr;
        HY_CK(cudaMalloc(&p, sizeof(int) * n));
        HY_CK(cudaMemset(p, 0xff, sizeof(int) * n));   // (-1: no cell)
        return p;
    };
    HY_CK(cudaMalloc(&f.dio, sizeof(IoT)));
    HY_CK(cudaMalloc(&f.cio, sizeof(IoT)));
    HY_CK(cudaMemset(f.dio, 0, sizeof(IoT)));
    HY_CK(cudaMemset(f.cio, 0, sizeof(IoT)));
    const int ffl = df_ff / S;
    f.snap = ia((size_t) DF_N * DT * n_cap);
    f.dc = ia((size_t) DF_B * df_ncd);
    f.cat = dalloc(2 * (size_t) E);
    f.fz = dalloc(E);
    f.z = dalloc((size_t) DF_B * E);
    f.ga = dalloc((size_t) DF_B * 2 * ffl);
    f.sh = dalloc((size_t) DF_B * ffl);
    f.pe = dalloc(E);
    f.cc = dalloc(2 * (size_t) E);
    f.cgu = dalloc(2 * (size_t) E);
    f.ch = dalloc(E);
    f.cd = dalloc(E);
    f.corr = dalloc(E);
    f.argv = dalloc(DF_N);
    f.argi = ia(DF_N);
}

template <int TT> void absorb_q8(Shard& s, DShard& d, const uint8_t* kw, int qpc, int QH, int Ha) {
    kd_absorb<TT><<<nb(Ha * kvl * 32), 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha);
}
template <int TT> void vb_gate_q8(Shard& s, DShard& d, const uint8_t* vw, int og, int xpc, int Hl) {
    kd_vb_gate<TT><<<nb(Hl * vh * 32), 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl);
}

void df_attention(Shard& s, DShard& d, DFShard& f, int i, const int* pp, int T) {
    const std::string P = "dfly." + std::to_string(i) + ".";
    const int Hl = Hd / S, QH = nope + rope, tl = df_tl[i];
    const int xpc = rows_of(s, P + "xproj"), qpc = rows_of(s, P + "qproj");
    const int okv = q_lora, og = q_lora + KV_ROW;
    kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "attn_norm.weight"), E, eps, d.x, E);
    k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    dec_mmvq(s, P + "xproj", d.xq, d.xp, T);
    kb_rms<<<T, 1024, 0, s.st>>>(d.xp, xpc, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, d.qr, q_lora);
    kb_kv_prep<<<T, 576, 0, s.st>>>(d.xp + okv, xpc, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, theta, s.kv[tl], pp,
                                    2, S, s.g, win + RING_PAD);
    k_q8<<<nb(T * q_lora), 256, 0, s.st>>>(d.qr, T * q_lora, d.xq);
    dec_mmvq(s, P + "qproj", d.xq, d.qp, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    if (kb.type != T_Q8_0 || vb.type != T_Q8_0) throw std::runtime_error("hy4f: DFly k_b / v_b are not Q8_0");
    absorb_q8<DF_B>(s, d, (const uint8_t*) kb.d, qpc, QH, Hd);
    kd_df_cells<<<T, 256, 0, s.st>>>(f.snap + (size_t) i * DT * n_cap, n_cap, (const int*) f.dio, f.dc, df_ncd);
    const float scale = 1.0f / std::sqrt((float) QH);
    const float* sinks = f32(s, P + "attn_sinks.weight");
    kd_mqa_dec<32><<<dim3(NSP, T), 256, mqd_smem<32>(), s.st>>>(d.qabs, d.qp, qpc, QH, nope, theta, s.kv[tl], f.dc, df_ncd, pp,
                                                               win, S, s.g, scale, d.spm, d.spl, d.spacc);
    if (S > 1) {
        kd_mqa_merge<<<dim3(Hd, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hd, nullptr, nullptr, d.part, CBW());
        if (push_seq)
            kd_push_seq<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl, d.cb, s.g, S,
                                                          d.cb_ctr, d.arrive + 8);
        else
            for (int r = 1; r < S; ++r)
                kd_push1<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl, d.cb, s.g,
                                                           (s.g + r) % S, d.cb_ctr, d.arrive + 8);
        kd_cb_tail<<<T * Hl, 256, 0, s.st>>>(d.part, T, Hd, kvl, CBW(), s.g * Hl, Hl, sinks, d.cb, s.g, S, d.cb_ctr,
                                             d.arrive + 12, d.o);
    } else {
        kd_mqa_merge<<<dim3(Hd, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hd, sinks, d.o, nullptr, 0);
    }
    vb_gate_q8<DF_B>(s, d, (const uint8_t*) vb.d, og, xpc, Hl);
    k_q8<<<nb(T * Hl * vh), 256, 0, s.st>>>(d.a, T * Hl * vh, d.xq);
    dec_mmvq(s, P + "attn_output.weight", d.xq, d.ao, T);
    d_ar(s, d, d.ao, nullptr, P + "post_attention_norm.weight", T, true);
}

void df_ffn(Shard& s, DShard& d, DFShard& f, int i, int T) {
    const std::string P = "dfly." + std::to_string(i) + ".";
    const int fl = df_ff / S;
    kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, d.x, E);
    k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    dec_mmvq(s, P + "dense_gu", d.xq, f.ga, T);
    kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(f.ga, 2 * fl, fl, T, f.sh);
    k_q8<<<nb(T * fl), 256, 0, s.st>>>(f.sh, T * fl, d.xq);
    dec_mmvq(s, P + "ffn_down.weight", d.xq, d.ffo, T);
    d_ar(s, d, d.ffo, nullptr, "", T, true);   // (no post-FFN norm in the drafter)
}

// correction step k: prev token = (own_io ? cio : dio)->toks[0]; draft k -> df_hargv / df_hargi [g][k]
void df_corr_ops(Shard& s, DShard& d, DFShard& f, int k, bool own_io) {
    if (own_io) HY_CK(cudaMemcpyAsync(f.cio, hio_c, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const DW& emb = s.w.at("token_embd.weight");
    kd_embed_rows<<<dim3(nb(E), 1), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, own_io ? f.cio->toks : f.dio->toks, E, f.pe);
    const float* zk = f.z + (size_t) (k + 1) * E;
    kd_df_rms2<<<1, 1024, 0, s.st>>>(zk, f32(s, "dfly.corr.hidden_norm.weight"), f.pe, nullptr, f32(s, "dfly.corr.embed_norm.weight"),
                                     E, eps, f.cc);
    k_q8<<<nb(2 * E), 256, 0, s.st>>>(f.cc, 2 * E, d.xq);
    dec_mmvq(s, "dfly.corr.gu", d.xq, f.cgu, 1);
    kb_swiglu_rows<<<dim3(nb(E), 1), 256, 0, s.st>>>(f.cgu, 2 * E, E, 1, f.ch);
    k_q8<<<nb(E), 256, 0, s.st>>>(f.ch, E, d.xq);
    dec_mmvq(s, "dfly.corr.down.weight", d.xq, f.cd, 1);
    kd_add<<<nb(E), 256, 0, s.st>>>(zk, f.cd, E, f.corr);
    head_rows(s, d, std::getenv("HY4F_DFLY_NOCORR") ? zk : f.corr, 1, f.argv + k, f.argi + k);   // (ablation: no correction)
    HY_CK(cudaMemcpyAsync(df_hargv + (size_t) s.g * DF_N + k, f.argv + k, sizeof(float), cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(df_hargi + (size_t) s.g * DF_N + k, f.argi + k, sizeof(int), cudaMemcpyDeviceToHost, s.st));
    if (df_logits)
        HY_CK(cudaMemcpyAsync(df_hlog + (size_t) k * V + (size_t) s.g * (V / S), d.logits, sizeof(float) * (V / S),
                              cudaMemcpyDeviceToHost, s.st));
}

void df_block_ops(Shard& s, DShard& d, DFShard& f) {
    HY_CK(cudaMemcpyAsync(f.dio, hio_d, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &f.dio->pos;
    const DW& emb = s.w.at("token_embd.weight");
    kd_embed_rows<<<dim3(nb(E), DF_B), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, f.dio->toks, E, d.me);
    kd_df_rms2<<<1, 1024, 0, s.st>>>(d.me, f32(s, "dfly.fuse.enorm.weight"), d.h, (const int*) f.dio,
                                     f32(s, "dfly.fuse.hnorm.weight"), E, eps, f.cat);
    k_q8<<<nb(2 * E), 256, 0, s.st>>>(f.cat, 2 * E, d.xq);
    dec_mmvq(s, "dfly.fuse.eh_proj.weight", d.xq, f.fz, 1);
    kd_df_init<<<nb(DF_B * E), 256, 0, s.st>>>(d.me, f.fz, f32(s, "dfly.fuse.gate.weight"), E, DF_B, d.mr);
    for (int i = 0; i < df_nl; ++i) {
        df_attention(s, d, f, i, pp, DF_B);
        df_ffn(s, d, f, i, DF_B);
    }
    kb_rms<<<DF_B, 1024, 0, s.st>>>(d.mr, E, f32(s, "dfly.final_norm.weight"), E, eps, f.z, E);
    df_corr_ops(s, d, f, 0, false);
}

void df_setup() {
    if (DT < DF_B) throw std::runtime_error("hy4f: the DFly drafter needs the -DHY4F_DT6 build (verify steps of 6 tokens)");
    HY_CK(cudaHostAlloc((void**) &hio_d, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hio_c, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &df_hargv, sizeof(float) * S * DF_N, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &df_hargi, sizeof(int) * S * DF_N, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &df_hlog, sizeof(float) * DF_N * V, cudaHostAllocPortable));
    std::memset(hio_d, 0, sizeof(IoT));
    std::memset(hio_c, 0, sizeof(IoT));
    df_logits = std::getenv("HY4F_DFLY_LOGITS") != nullptr;
    dfs.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) df_alloc(sh[i], dfs[i]);
}

void df_capture(Shard& s) {
    DShard& d = dsh[s.g];
    DFShard& f = dfs[s.g];
    cudaGraph_t gr = nullptr;
    HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
    df_block_ops(s, d, f);
    HY_CK(cudaStreamEndCapture(s.st, &gr));
    HY_CK(cudaGraphInstantiate(&f.gb, gr, 0));
    HY_CK(cudaGraphDestroy(gr));
    for (int k = 1; k < DF_N; ++k) {
        HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
        df_corr_ops(s, d, f, k, true);
        HY_CK(cudaStreamEndCapture(s.st, &gr));
        HY_CK(cudaGraphInstantiate(&f.gk[k], gr, 0));
        HY_CK(cudaGraphDestroy(gr));
    }
}

int df_pick(int k) {
    float bv = -INFINITY;
    int bi = 0;
    for (int g = 0; g < S; ++g) {
        const float v = df_hargv[g * DF_N + k];
        if (v > bv) bv = v, bi = df_hargi[g * DF_N + k];
    }
    return bi;
}

// n (1..5) drafts after `bonus` at `anchor`, the source row `row` of the last verify; drafts[k] = argmax of step k
void df_run(int row, int bonus, int64_t anchor, int n, int* drafts) {
    hio_d->pos = (int) anchor;
    hio_d->n = row;
    hio_d->toks[0] = bonus;
    for (int t = 1; t < DF_B; ++t) hio_d->toks[t] = df_mask;
    auto run = [&](int k) {
        launch_parallel([&](Shard& s) { HY_CK(cudaGraphLaunch(k == 0 ? dfs[s.g].gb : dfs[s.g].gk[k], s.st)); });
        for (auto& s : sh) {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaStreamSynchronize(s.st));
        }
        HY_CK(cudaGetLastError());
    };
    run(0);
    drafts[0] = df_pick(0);
    for (int k = 1; k < n; ++k) {
        hio_c->toks[0] = drafts[k - 1];
        run(k);
        drafts[k] = df_pick(k);
    }
}
