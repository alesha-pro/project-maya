// src/hy4f/hy4f_batch.inc - maya-hy4f: the batched forward (a chunk of T tokens), included inside Model::Impl.
// The prompt path (and later the speculative verify): projections and experts through llama.cpp's MMQ (int8 tensor
// cores, prefill::mmq), F32 matrices through cuBLAS, attention by the batched kernels of hy4f_bkern.cuh, the
// exchanges as eager push / wait kernels with a host step counter.  Launched eagerly from the per-shard threads.

struct BShard {
    int Tmax = 0;
    strata::prefill::mmq::Context* mmq = nullptr;
    cublasHandle_t blas = nullptr;
    int *dbounds = nullptr, *iota = nullptr, *toks = nullptr;
    void *xq = nullptr, *xq2 = nullptr;
    float *X = nullptr, *R = nullptr, *x = nullptr, *post = nullptr, *hcp = nullptr, *hcs = nullptr;
    float *xp = nullptr, *qr = nullptr, *qp = nullptr, *qabs = nullptr, *o = nullptr, *part = nullptr, *a = nullptr,
          *ao = nullptr, *ikn = nullptr, *ivg = nullptr, *iw = nullptr, *score = nullptr;
    uint8_t* chunk = nullptr;
    int* cells = nullptr;
    float *lg = nullptr, *ew = nullptr, *xm = nullptr, *gu = nullptr, *dn = nullptr, *moe = nullptr, *mu = nullptr,
          *shh = nullptr, *shd = nullptr, *ffo = nullptr;
    int *eids = nullptr, *mb = nullptr, *mb2 = nullptr, *mcnt = nullptr, *rows_tok = nullptr, *row_of = nullptr;
    int* hmb = nullptr;   // pinned copy of the bounds (the MMQ grid and the rows are host values)
    int64_t cap = 0;      // rows of the expert buffers
    float *eg_r = nullptr, *eg_kv = nullptr;
    int64_t* eg_ids = nullptr;
    float* ar_buf[2] = {nullptr, nullptr};   // reduce-scatter receive [src][ts][E] bf16
    float* ag_buf[2] = {nullptr, nullptr};   // all-gather receive [T][E] fp16
    __half* yn = nullptr;                    // this shard's normalized slice [ts][E] fp16
    int* ag_flag = nullptr;
    Peers ag{};
    float* cb_buf[2] = {nullptr, nullptr};
    int *ar_flag = nullptr, *cb_flag = nullptr, *ctr = nullptr;   // ctr: [2][4] arrive counters (all-reduce, combine)
    float* wdq = nullptr;   // a dequantized k_b / v_b (F32) for the batched products
    Peers ar{}, cb{};
    int step = 0;
    int* hb = nullptr;   // pinned {0, T}
};
std::vector<BShard> bsh;
int score_tile = 64;
int CBW() const { return kvl + 4; }   // a combine partial's row: kvl accumulators, M, L, 2 pad (float4 / WMMA aligned)
size_t xp_ld = 0, qp_ld = 0, xm_ld = 0;

#define HY_BL(x)                                                                                                        \
    do {                                                                                                                \
        cublasStatus_t s_ = (x);                                                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x) + ": cublas " + std::to_string((int) s_)); \
    } while (0)

void batch_alloc(Shard& s, BShard& b, int Tmax) {
    HY_CK(cudaSetDevice(s.dev));
    b.Tmax = Tmax;
    const int T = Tmax, K = n_used, QH = nope + rope, Hmax = std::max(Hd, Hs);
    const int64_t C = o.max_context;
    b.mmq = new strata::prefill::mmq::Context();
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<16>()));
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<32>()));
    HY_BL(cublasCreate(&b.blas));
    HY_BL(cublasSetStream(b.blas, s.st));
    HY_BL(cublasSetMathMode(b.blas, CUBLAS_DEFAULT_MATH));
    HY_CK(cudaMalloc(&b.dbounds, sizeof(int) * 2));
    HY_CK(cudaMalloc(&b.iota, sizeof(int) * ((size_t) T * K + strata::prefill::mmq::IDS_PAD)));
    strata::prefill::mmq::iota(b.iota, (int64_t) T * K + strata::prefill::mmq::IDS_PAD, s.st);
    HY_CK(cudaMalloc(&b.toks, sizeof(int) * T));
    HY_CK(cudaHostAlloc((void**) &b.hb, sizeof(int) * 2, cudaHostAllocPortable));
    const size_t q8a = std::max({strata::prefill::mmq::q8_bytes(std::min((int64_t) T * K, std::max<int64_t>(2 * (int64_t) T * K / S, 4 * T)), std::max(latent, n_ffx)),
                                 strata::prefill::mmq::q8_bytes(T, std::max({E, Hmax / S * vh, n_ff / S, 4096})),
                                 strata::prefill::mmq::q8_bytes(T, eg_D * (eg_ngram - 1) * eg_heads)});
    HY_CK(cudaMalloc(&b.xq, q8a));
    HY_CK(cudaMalloc(&b.xq2, q8a));
    b.X = dalloc((size_t) T * hc * E);
    b.R = dalloc((size_t) T * hc * E);
    b.x = dalloc((size_t) T * E);
    b.post = dalloc((size_t) T * 8);
    b.hcp = dalloc((size_t) T * 8 * HC_SPLIT);
    b.hcs = dalloc((size_t) T * HC_SPLIT);
    xp_ld = ((size_t) q_lora + KV_ROW + (size_t) Hmax / S * vh + IK + 3) / 4 * 4;
    qp_ld = std::max((size_t) Hd * QH + (size_t) IH * IK, (size_t) Hs / S * QH);
    xm_ld = (size_t) latent + 2 * n_ffx / S;
    b.xp = dalloc((size_t) T * xp_ld);
    b.qr = dalloc((size_t) T * q_lora);
    b.qp = dalloc((size_t) T * qp_ld);
    b.qabs = dalloc((size_t) T * Hmax * kvl);
    b.o = dalloc((size_t) T * Hmax * kvl);
    b.part = dalloc((size_t) T * Hd * CBW());
    b.a = dalloc((size_t) T * Hmax / S * vh);
    b.ao = dalloc((size_t) T * E);
    b.ikn = dalloc((size_t) T * IK);
    b.ivg = dalloc((size_t) T * 2 * IK);
    b.iw = dalloc((size_t) T * IH);
    b.score = dalloc((size_t) score_tile * (C / kpool + 1));
    HY_CK(cudaMalloc(&b.chunk, (size_t) T * KV_RB));
    HY_CK(cudaMalloc(&b.cells, sizeof(int) * (size_t) T * n_cap));
    b.lg = dalloc((size_t) T * n_exp);
    b.ew = dalloc((size_t) T * K);
    HY_CK(cudaMalloc(&b.eids, sizeof(int) * (size_t) T * K));
    HY_CK(cudaMalloc(&b.mb, sizeof(int) * (n_exp / S + 1)));
    HY_CK(cudaMalloc(&b.mb2, sizeof(int) * (n_exp / S + 1)));
    HY_CK(cudaMalloc(&b.mcnt, sizeof(int) * (n_exp / S + 1)));
    HY_CK(cudaHostAlloc((void**) &b.hmb, sizeof(int) * (n_exp / S + 1), cudaHostAllocPortable));
    b.cap = std::min((int64_t) T * K, std::max<int64_t>(2 * (int64_t) T * K / S, 4 * T));
    HY_CK(cudaMalloc(&b.rows_tok, sizeof(int) * ((size_t) T * K + strata::prefill::mmq::IDS_PAD)));
    HY_CK(cudaMemset(b.rows_tok, 0, sizeof(int) * ((size_t) T * K + strata::prefill::mmq::IDS_PAD)));
    HY_CK(cudaMalloc(&b.row_of, sizeof(int) * (size_t) T * K));
    b.xm = dalloc((size_t) T * xm_ld);
    b.gu = dalloc((size_t) b.cap * 2 * n_ffx);
    b.dn = dalloc((size_t) b.cap * latent);
    b.moe = dalloc((size_t) T * latent);
    b.mu = dalloc((size_t) T * E);
    b.shh = dalloc((size_t) T * std::max(n_ffx / S, n_ff / S));
    b.shd = dalloc((size_t) T * std::max((size_t) E, (size_t) 2 * n_ff / S));
    b.ffo = dalloc((size_t) T * E);
    const int ncol = (eg_ngram - 1) * eg_heads;
    b.eg_r = dalloc((size_t) T * ncol * eg_D);
    b.eg_kv = dalloc((size_t) T * (hc + 1) * E);
    HY_CK(cudaMalloc(&b.eg_ids, sizeof(int64_t) * (size_t) T * ncol));
    const int ts = (T + S - 1) / S;
    for (int p = 0; p < 2; ++p) {
        b.ar_buf[p] = dalloc((size_t) S * ts * E / 2 + 64);
        b.ag_buf[p] = dalloc((size_t) T * E / 2 + 64);
        b.cb_buf[p] = dalloc((size_t) S * T * (Hd / S) * CBW());
    }
    HY_CK(cudaMalloc(&b.yn, sizeof(__half) * (size_t) ts * E));
    HY_CK(cudaMalloc(&b.ag_flag, sizeof(int) * 4));
    HY_CK(cudaMemset(b.ag_flag, 0, sizeof(int) * 4));
    HY_CK(cudaMalloc(&b.ar_flag, sizeof(int) * 4));
    HY_CK(cudaMemset(b.ar_flag, 0, sizeof(int) * 4));
    HY_CK(cudaMalloc(&b.cb_flag, sizeof(int) * 4));
    HY_CK(cudaMemset(b.cb_flag, 0, sizeof(int) * 4));
    HY_CK(cudaMalloc(&b.ctr, sizeof(int) * 12));
    HY_CK(cudaMemset(b.ctr, 0, sizeof(int) * 12));
    b.wdq = dalloc(std::max((size_t) Hd * kvl * nope, (size_t) Hmax / S * vh * kvl));
}

void batch_setup(int Tmax) {
    bsh.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) batch_alloc(sh[i], bsh[i], Tmax);
    Peers ar{}, cb{}, ag{};
    for (int j = 0; j < S; ++j) {
        ar.buf[j][0] = bsh[j].ar_buf[0];
        ar.buf[j][1] = bsh[j].ar_buf[1];
        ar.flag[j] = bsh[j].ar_flag;
        ag.buf[j][0] = bsh[j].ag_buf[0];
        ag.buf[j][1] = bsh[j].ag_buf[1];
        ag.flag[j] = bsh[j].ag_flag;
        cb.buf[j][0] = bsh[j].cb_buf[0];
        cb.buf[j][1] = bsh[j].cb_buf[1];
        cb.flag[j] = bsh[j].cb_flag;
    }
    for (auto& b : bsh) b.ar = ar, b.cb = cb, b.ag = ag;
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaDeviceSynchronize());
    }
}

// y[T][rows] = x[T][cols] W^T through MMQ (x quantized here); ld_x / ld_y in floats
void blin(Shard& s, BShard& b, const DW& w, const float* x, int ldx, float* y, int ldy, int T) {
    namespace mq = strata::prefill::mmq;
    const int cols = (int) w.ne[0], rows = (int) (w.ne[1] * w.ne[2]);
    if (w.type == T_F32) {
        const float one = 1.0f, zero = 0.0f;
        HY_BL(cublasSgemm(b.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &one, (const float*) w.d, cols, x, ldx, &zero, y,
                          ldy));
        return;
    }
    if (!mq::supported(w.type)) throw std::runtime_error("hy4f: MMQ does not take type " + std::to_string(w.type));
    mq::quantize(x, nullptr, b.xq, w.type, cols, ldx, T, s.st);
    mq::Product p{};
    p.w = w.d;
    p.type = w.type;
    p.w_rows = rows;
    p.w_cols = cols;
    p.expert_bytes = 0;
    p.n = 1;
    p.xq = b.xq;
    p.bounds = b.dbounds;
    p.ids = b.iota;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = ldy;
    b.mmq->run(p, s.st);
}
void blin(Shard& s, BShard& b, const std::string& n, const float* x, int ldx, float* y, int ldy, int T) {
    blin(s, b, s.w.at(n), x, ldx, y, ldy, T);
}

// the payload to every shard, one peer at a time in ring order (this shard last)
void push(Shard& s, BShard& b, const float* src, size_t per, int mode, int Hl, int W, const Peers& P, int* ctr) {
    const int blocks = std::min(nb((int) (per / 4), 512), 160);
    for (int r = 1; r <= S; ++r) {
        const int j = (s.g + r) % S;
        kb_push<<<blocks, 512, 0, s.st>>>(src, per, mode, 0, Hl, W, P, s.g, S, b.step, ctr, j);
    }
}
// y_h = op(A_h) x_h for each head h (cuBLAS strided batched SGEMM, column-major views)
void bgemm(BShard& b, int m, int n, int k, const float* A, int lda, long long sA, const float* B, int ldb, long long sB,
           float* C, int ldc, long long sC, int batch) {
    const float one = 1.0f, zero = 0.0f;
    HY_BL(cublasSgemmStridedBatched(b.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, lda, sA, B, ldb, sB, &zero, C, ldc,
                                    sC, batch));
}

// all-reduce of y [T][E] then RMS * w, X = R + post * y: reduce-scatter (bf16) by token slices, all-gather (fp16)
void b_ar_post(Shard& s, BShard& b, const float* y, const std::string& norm, int T) {
    if (S == 1) {
        kb_ar_post<<<T, 1024, 0, s.st>>>(y, E, b.ar, s.g, S, b.step, f32(s, norm), eps, b.X, b.R, b.post, hc, (size_t) T * E);
        return;
    }
    const int ts = (T + S - 1) / S, blocks = std::min(nb(ts * E / 8, 512), 160);
    ++b.step;
    for (int r = 1; r <= S; ++r)
        kb_rs_push<<<blocks, 512, 0, s.st>>>(y, T, E, ts, b.ar, s.g, b.step, b.ctr, (s.g + r) % S);
    kb_rs_post<<<ts, 1024, 0, s.st>>>(T, E, ts, b.ar, s.g, S, b.step, f32(s, norm), eps, b.yn);
    ++b.step;
    for (int r = 1; r <= S; ++r)
        kb_ag_push<<<blocks, 512, 0, s.st>>>(b.yn, T, E, ts, b.ag, s.g, b.step, b.ctr + 8, (s.g + r) % S);
    kb_ag_post<<<T, 1024, 0, s.st>>>(E, b.ag, s.g, S, b.step, b.X, b.R, b.post, hc);
}

void b_norm_in(Shard& s, BShard& b, const std::string& P, const std::string& k, const std::string& norm, int n_mix,
               float mag, int T, size_t t0 = 0) {
    const float* X = b.X + t0 * hc * E;
    kb_hc_mix<<<dim3(HC_SPLIT, T), 256, 0, s.st>>>(X, f32(s, P + k + "_fn.weight"), hc * E, n_mix, b.hcp, b.hcs);
    kb_hc_apply_norm<<<T, 1024, 0, s.st>>>(X, b.hcp, b.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                           f32(s, norm), E, hc, n_mix, eps, hc_eps, mag, b.x, b.post);
}

void b_engram(Shard& s, BShard& b, int e, int il, int pos0, int T) {
    const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
    const std::string P = "blk." + std::to_string(il) + ".";
    kb_engram_hash<<<T, 32, 0, s.st>>>(s.hist, pos0, s.eg_map, (int64_t) eg_map.size(), eg_pad, s.eg_mult + (size_t) e * ng,
                                       s.eg_vsz + (size_t) e * ncol, s.eg_off + (size_t) e * ncol, ng, nh, b.eg_ids);
    kb_engram_rows<<<dim3(ncol, T), 128, 0, s.st>>>(eg_table, b.eg_ids, f32(s, P + "engram_lookup_norm.weight"), b.eg_r,
                                                    eg_D, ncol, 1.1920929e-07f);
    blin(s, b, P + "engram_kv.weight", b.eg_r, ncol * eg_D, b.eg_kv, (hc + 1) * E, T);
    kb_engram_mix<<<T, 1024, 0, s.st>>>(b.X, b.R, b.eg_kv, f32(s, P + "engram_kv_b.weight"), f32(s, P + "engram_gate.weight"),
                                        f32(s, P + "engram_hc_scale.weight"), f32(s, P + "engram_pre_gamma.weight"), E, hc,
                                        1.1920929e-07f);
}

void b_attention(Shard& s, BShard& b, int il, int pos0, int T) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const bool sw = swa[il] != 0;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
    b_norm_in(s, b, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
    blin(s, b, P + "xproj", b.x, E, b.xp, (int) xp_ld, T);
    kb_rms<<<T, 1024, 0, s.st>>>(b.xp + oq, (int) xp_ld, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, b.qr, q_lora);
    kb_kv_prep<<<T, 576, 0, s.st>>>(b.xp + okv, (int) xp_ld, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th,
                                    sw ? b.chunk : s.kv[il], pos0, sw ? 1 : 2, S, s.g);
    blin(s, b, P + "qproj", b.qr, q_lora, b.qp, (int) qp_ld, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const int Ha = sw ? Hl : Hd;   // heads absorbed / attended on this shard
    // q_abs[t][h] = k_b_h q_nope[t][h]: k_b_h [kvl][nope] row-major
    kb_deq<<<std::min(nb(Ha * kvl * nope), 1024), 256, 0, s.st>>>(kb.d, kb.type, (size_t) Ha * kvl * nope, b.wdq);
    bgemm(b, kvl, T, nope, b.wdq, nope, (long long) kvl * nope, b.qp, (int) qp_ld, QH, b.qabs, Ha * kvl, kvl, Ha);
    if (sw) {
        if (Hl == 16)
            kb_mqa_attn<16><<<T, 256, mqa_smem<16>(), s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk,
                                                            nullptr, 0, pos0, win, 1, 0,
                                                            f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, b.o, nullptr, 0);
        else if (Hl == 32)
            kb_mqa_attn<32><<<T, 256, mqa_smem<32>(), s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk,
                                                            nullptr, 0, pos0, win, 1, 0,
                                                            f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, b.o, nullptr, 0);
        else
            kb_attn2<<<dim3(Hl, T), 256, 0, s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk, nullptr, 0,
                                                    pos0, win, 1, 0, f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, b.o,
                                                    nullptr, Hl);
        kb_ring_write<<<std::min(T, win), 152, 0, s.st>>>(b.chunk, s.kv[il], T, pos0, win);
    } else {
        // the indexer: pooled keys of the groups completed in the chunk, then the per-token selection in tiles
        float* iq = b.qp + (size_t) Hd * QH;
        kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, s.st>>>(iq, (int) qp_ld, IH, IK, IK - rope, rope, pos0, th);
        kb_rms<<<T, 128, 0, s.st>>>(b.xp + oik, (int) xp_ld, f32(s, P + "indexer.k_norm.weight"), IK, eps, b.ikn, IK);
        const DW& ivw = s.w.at(P + "ivg");
        kb_q8_gemv<<<dim3(nb(2 * IK * 32), T), 256, 0, s.st>>>((const uint8_t*) ivw.d, b.ikn, IK, b.ivg, 2 * IK, IK, 2 * IK);
        const int g_first = pos0 / kpool, n_groups = (pos0 + T) / kpool - g_first;
        if (n_groups > 0)
            kb_pool_groups<<<n_groups, 128, 0, s.st>>>(b.ivg, s.ring[il], pos0, g_first,
                                                       f32(s, P + "indexer_compressor_ape.weight"),
                                                       f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK,
                                                       kpool, rope, eps, th);
        kb_ring_chunk<<<kpool, 256, 0, s.st>>>(b.ivg, s.ring[il], T, pos0, IK, kpool);
        blin(s, b, P + "indexer.proj.weight", b.x, E, b.iw, IH, T);
        k_scale<<<nb(T * IH), 256, 0, s.st>>>(b.iw, T * IH, 1.0f / std::sqrt((float) (IK * IH)));
        const int64_t ld_sc = o.max_context / kpool + 1;
        for (int t0 = 0; t0 < T; t0 += score_tile) {
            const int tt = std::min(score_tile, T - t0);
            const int n_pool_max = (pos0 + t0 + tt - 1) / kpool;
            if (pos0 + t0 + tt > top_k && n_pool_max > 0)
                kb_score<<<dim3(nb(n_pool_max * 32), tt), 256, 0, s.st>>>(iq, (int) qp_ld, s.pooled[il], b.iw, IH, IK, IH,
                                                                          kpool, top_k, pos0, t0, b.score, (int) ld_sc);
            kb_select_kpool<1024><<<tt, 1024, 0, s.st>>>(b.score, (int) ld_sc, pos0, t0, kpool, top_k, top_k / kpool - tail,
                                                         n_cap, b.cells);
        }
        if (S > 1) {
            if (Hd == 32 && !std::getenv("HY4F_OLD_ATTN"))
                kb_mqa_attn<32><<<T, 256, mqa_smem<32>(), s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], nullptr,
                                                                b.cells, n_cap, pos0, win, S, s.g, nullptr, scale, nullptr,
                                                                b.part, CBW());
            else
                kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], nullptr, b.cells,
                                                        n_cap, pos0, win, S, s.g, nullptr, scale, nullptr, b.part, Hd);
            ++b.step;
            const size_t per = (size_t) T * Hl * CBW();
            push(s, b, b.part, per, 1, Hl, CBW(), b.cb, b.ctr + 4);
            kb_combine<<<T, 512, 0, s.st>>>(kvl, CBW(), s.g * Hl, Hl, f32(s, P + "attn_sinks.weight"), b.cb, s.g, S, b.step, per,
                                            b.o);
        } else {
            kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], nullptr, b.cells, n_cap,
                                                    pos0, win, 1, 0, f32(s, P + "attn_sinks.weight"), scale, b.o, nullptr, Hd);
        }
    }
    // a[t][h] = v_b_h o[t][h] (v_b_h [vh][kvl] row-major), then the sigmoid gate
    kb_deq<<<std::min(nb(Hl * vh * kvl), 1024), 256, 0, s.st>>>(vb.d, vb.type, (size_t) Hl * vh * kvl, b.wdq);
    bgemm(b, vh, T, kvl, b.wdq, kvl, (long long) vh * kvl, b.o, Hl * kvl, kvl, b.a, Hl * vh, vh, Hl);
    kb_gate_mul<<<dim3(nb(Hl * vh), T), 256, 0, s.st>>>(b.a, Hl * vh, b.xp + og, (int) xp_ld, T);
    blin(s, b, P + "attn_output.weight", b.a, Hl * vh, b.ao, E, T);
    b_ar_post(s, b, b.ao, P + "post_attention_norm.weight", T);
}

void b_ffn(Shard& s, BShard& b, int il, int T) {
    namespace mq = strata::prefill::mmq;
    const std::string P = "blk." + std::to_string(il) + ".";
    b_norm_in(s, b, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag, T);
    if (il < dense_lead) {
        const int fl = n_ff / S;
        blin(s, b, P + "dense_gu", b.x, E, b.shd, 2 * fl, T);
        kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(b.shd, 2 * fl, fl, T, b.shh);
        blin(s, b, P + "ffn_down.weight", b.shh, fl, b.ffo, E, T);
        b_ar_post(s, b, b.ffo, P + "post_ffw_norm.weight", T);
        return;
    }
    const int K = n_used, e0 = s.g * n_exp / S, nl = n_exp / S, fl = n_ffx / S;
    blin(s, b, P + "ffn_gate_inp.weight", b.x, E, b.lg, n_exp, T);
    kn::router_sigmoid(b.lg, f32(s, P + "exp_probs_b.bias"), T, n_exp, K, w_scale, true, b.eids, b.ew, s.st);
    kb_moe_count<<<nl, 256, 0, s.st>>>(b.eids, T * K, e0, b.mcnt);
    kb_moe_scan<<<1, std::max(nl, 32), 0, s.st>>>(b.mcnt, nl, b.mb);
    kb_moe_place<<<nl, 256, 0, s.st>>>(b.eids, T * K, K, e0, nl, b.mb, b.rows_tok, b.row_of);
    HY_CK(cudaMemcpyAsync(b.hmb, b.mb, sizeof(int) * (nl + 1), cudaMemcpyDeviceToHost, s.st));
    blin(s, b, P + "xmoe", b.x, E, b.xm, (int) xm_ld, T);
    HY_CK(cudaStreamSynchronize(s.st));   // (the xmoe product queued behind it keeps the GPU busy meanwhile)
    const DW& wgu = s.w.at(P + "ffn_gu_exps");
    const DW& wd = s.w.at(P + "ffn_down_exps.weight");
    // the experts in ranges of at most cap rows (one range unless the routing piles onto this shard)
    for (int ea = 0; ea < nl;) {
        int eb = ea;
        int64_t mx = 0;
        while (eb < nl && (eb == ea || b.hmb[eb + 1] - b.hmb[ea] <= b.cap)) {
            mx = std::max<int64_t>(mx, b.hmb[eb + 1] - b.hmb[eb]);
            ++eb;
        }
        const int ra = b.hmb[ea], rb = b.hmb[eb];
        const int64_t rows = rb - ra;
        if (rows > b.cap) throw std::runtime_error("hy4f: one expert has more rows than the buffers");
        const int* bounds = b.mb + ea;
        if (ra != 0) {
            kb_shift_bounds<<<nb(eb - ea + 1), 256, 0, s.st>>>(b.mb + ea, b.mb2, eb - ea);
            bounds = b.mb2;
        }
        if (rows > 0) {
            mq::quantize(b.xm, b.rows_tok + ra, b.xq, wgu.type, latent, (int64_t) xm_ld, rows, s.st);
            mq::Product p{};
            p.w = static_cast<const uint8_t*>(wgu.d) + (size_t) ea * wgu.rb * (size_t) wgu.ne[1];
            p.type = wgu.type;
            p.w_rows = 2 * n_ffx;
            p.w_cols = latent;
            p.expert_bytes = wgu.rb * (size_t) wgu.ne[1];
            p.n = eb - ea;
            p.xq = b.xq;
            p.bounds = bounds;
            p.ids = b.iota;
            p.total_rows = rows;
            p.max_rows = mx;
            p.dst = b.gu;
            p.ld_dst = 2 * n_ffx;
            b.mmq->run_glu(p, clampv, s.st);
            mq::quantize(b.gu, nullptr, b.xq2, wd.type, n_ffx, 2 * n_ffx, rows, s.st);
            p.w = static_cast<const uint8_t*>(wd.d) + (size_t) ea * wd.rb * (size_t) wd.ne[1];
            p.type = wd.type;
            p.w_rows = latent;
            p.w_cols = n_ffx;
            p.expert_bytes = wd.rb * (size_t) wd.ne[1];
            p.xq = b.xq2;
            p.dst = b.dn;
            p.ld_dst = latent;
            b.mmq->run(p, s.st);
        }
        kb_moe_combine<<<dim3(nb(latent), T), 256, 0, s.st>>>(b.dn, latent, b.row_of, b.ew, K, b.moe, latent, ra, rb, ea > 0);
        ea = eb;
    }
    blin(s, b, P + "ffn_routed_up.weight", b.moe, latent, b.mu, E, T);
    kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(b.xm + latent, (int) xm_ld, fl, T, b.shh);
    blin(s, b, P + "ffn_down_shexp.weight", b.shh, fl, b.shd, E, T);
    kb_add<<<nb(T * E), 256, 0, s.st>>>(b.mu, b.shd, b.ffo, (size_t) T * E);
    b_ar_post(s, b, b.ffo, P + "post_ffw_norm.weight", T);
}

// one chunk on one shard: tokens b.toks (already on the device), positions pos0 ..; logits of the last token
void b_chunk(Shard& s, BShard& b, int pos0, int T, bool want_logits) {
    b.hb[0] = 0;
    b.hb[1] = T;
    HY_CK(cudaMemcpyAsync(b.dbounds, b.hb, sizeof(int) * 2, cudaMemcpyHostToDevice, s.st));
    const DW& emb = s.w.at("token_embd.weight");
    kb_embed<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, b.toks, pos0, b.X, b.R, E, hc, s.hist);
    for (int il = 0; il < L; ++il) {
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) b_engram(s, b, (int) e, il, pos0, T);
        b_attention(s, b, il, pos0, T);
        b_ffn(s, b, il, T);
    }
    if (want_logits) {
        b_norm_in(s, b, "", "output_hc", "output_norm.weight", hc, 0.0f, 1, (size_t) T - 1);
        gemv(s, "output.weight", b.x, s.logits);
        HY_CK(cudaMemcpyAsync(hlogits + (size_t) s.g * (V / S), s.logits, sizeof(float) * (V / S), cudaMemcpyDeviceToHost,
                              s.st));
    }
}
