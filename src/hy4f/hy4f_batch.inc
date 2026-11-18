// src/hy4f/hy4f_batch.inc - maya-hy4f: the batched forward (a chunk of T tokens), included inside Model::Impl.
// The prompt path (and later the speculative verify): projections and experts through llama.cpp's MMQ (int8 tensor
// cores, prefill::mmq), F32 matrices through cuBLAS, attention by the batched kernels of hy4f_bkern.cuh, the
// exchanges as eager push / wait kernels with a host step counter.  Launched eagerly from the per-shard threads.

struct BShard {
    int Tmax = 0;
    cudaStream_t st = nullptr;   // this micro-batch's stream (A: the shard's stream, B: its own)
    // the SWA chunk buffer as this pass sees it (micro-batches share one per layer parity): kv rows written at chunk_off,
    // the attention's positions = *pp_attn + toff + t, the ring written for ring_T rows (0: another pass does it)
    uint8_t* chunk_cur = nullptr;
    int chunk_off = 0, toff = 0, ring_T = 0;
    const int* pp_attn = nullptr;
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
    int* cellsx[2] = {nullptr, nullptr};   // the indexer split's cells (persistent: peers write them), by step parity
    int* cl_flag = nullptr;
    Peers cl{};
    int *ar_flag = nullptr, *cb_flag = nullptr, *ctr = nullptr;   // ctr: [2][4] arrive counters (all-reduce, combine)
    float* wdq = nullptr;   // a dequantized k_b / v_b (F32) for the batched products
    Peers ar{}, cb{};
    int step = 0;
    int* hb = nullptr;   // pinned {0, T}
    int* emask = nullptr;   // [T] 1 at embedding (image) rows: no Engram
    size_t bytes = 0;    // device bytes of the batch buffers
    void* arena_base = nullptr;   // the arena (attention / FFN / Engram phases; the image encoder between calls)
    size_t arena_bytes = 0;
    // the MTP layer over the chunk (plain residual mr, positions at dbounds[3])
    bool mtp = false;
    float *mr = nullptr, *ones = nullptr, *hprev = nullptr, *hold = nullptr;
    const int* pp = nullptr;   // the positions' device pointer of the current pass
};
std::vector<BShard> bsh;
int score_tile = 64;
bool paired = !std::getenv("HY4F_RING_PUSH");   // exchanges in XOR-paired phases (S a power of two)
int peer_of(int g, int r) const { return paired && (S & (S - 1)) == 0 ? (g ^ r) : (g + r) % S; }
int cap_num = 3, cap_den = 2;   // expert rows buffered per shard: cap_num/cap_den x the even share (ranges past it)
int attn_tile = 512;
int mqa_opt = std::getenv("HY4F_PF_MQA") ? std::atoi(std::getenv("HY4F_PF_MQA")) : 0;
int64_t idx_split_from = std::getenv("HY4F_IDX_SPLIT_FROM") ? std::atoll(std::getenv("HY4F_IDX_SPLIT_FROM")) : 16384;   // indexer split by tokens from this depth   // tokens per attention tile (q_abs / o / partials / cells / the combine are tile-sized)
int CBW() const { return kvl + 4; }   // a combine partial's row: kvl accumulators, M, L, 2 pad (float4 / WMMA aligned)
size_t xp_ld = 0, qp_ld = 0, xm_ld = 0;

#define HY_BL(x)                                                                                                        \
    do {                                                                                                                \
        cublasStatus_t s_ = (x);                                                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(#x) + ": cublas " + std::to_string((int) s_)); \
    } while (0)

// one bump allocation, 256-byte aligned pieces
struct Bump {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class TT> TT* take(size_t n) {
        TT* p = reinterpret_cast<TT*>(base + off);
        off += (n * sizeof(TT) + 255) / 256 * 256;
        return p;
    }
};

// the batch buffers: persistent ones, and one arena the attention / FFN / Engram phases of a layer share
void batch_alloc(Shard& s, BShard& b, int Tmax) {
    HY_CK(cudaSetDevice(s.dev));
    b.Tmax = Tmax;
    if (!b.st) b.st = s.st;
    if (S == 1) attn_tile = Tmax;   // (the single-GPU attention falls back to kb_attn2 over the whole chunk)
    const int T = Tmax, K = n_used, QH = nope + rope, Hmax = std::max(Hd, Hs);
    const int64_t C = o.max_context;
    b.mmq = new strata::prefill::mmq::Context();
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<16>()));
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<32>()));
    HY_BL(cublasCreate(&b.blas));
    HY_BL(cublasSetStream(b.blas, b.st));
    HY_BL(cublasSetMathMode(b.blas, CUBLAS_DEFAULT_MATH));
    HY_CK(cudaHostAlloc((void**) &b.hb, sizeof(int) * 4, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &b.hmb, sizeof(int) * (nl_max + 1), cudaHostAllocPortable));
    b.cap = std::min((int64_t) T * K, std::max<int64_t>((int64_t) cap_num * T * K / (cap_den * S), 2 * T));
    xp_ld = ((size_t) q_lora + KV_ROW + (size_t) Hmax / S * vh + IK + 3) / 4 * 4;
    qp_ld = std::max((size_t) Hd * QH + (size_t) IH * IK, (size_t) Hs / S * QH);
    xm_ld = (size_t) latent + 2 * n_ffx / S;
    const size_t q8a = std::max({strata::prefill::mmq::q8_bytes(b.cap, std::max(latent, n_ffx)),
                                 strata::prefill::mmq::q8_bytes(T, std::max({E, Hmax / S * vh, n_ff / S, 4096})),
                                 strata::prefill::mmq::q8_bytes(T, eg_D * (eg_ngram - 1) * eg_heads)});
    const int ts = (T + S - 1) / S, ncol = (eg_ngram - 1) * eg_heads;
    const int Ha = std::max(Hd, Hs / S), Ho = S > 1 ? Hmax / S : Hmax;
    const int TA = std::min(T, attn_tile);
    // persistent
    auto persistent = [&](Bump& m) {
        b.dbounds = m.take<int>(4);   // {0, T, pos0, -}
        b.iota = m.take<int>((size_t) T * K + strata::prefill::mmq::IDS_PAD);
        b.toks = m.take<int>(T);
        b.xq = m.take<uint8_t>(q8a);
        b.xq2 = m.take<uint8_t>(q8a);
        b.X = m.take<float>((size_t) T * hc * E);
        b.R = b.X;   // (X = R + post * y and R = X: the same values everywhere in the batched path)
        b.x = m.take<float>((size_t) T * E);
        b.post = m.take<float>((size_t) T * 8);
        b.hcp = m.take<float>((size_t) T * 8 * HC_SPLIT);
        b.hcs = m.take<float>((size_t) T * HC_SPLIT);
        b.score = m.take<float>((size_t) score_tile * (C / kpool + 1));
        b.chunk = m.take<uint8_t>((size_t) T * KV_RB);
        b.ew = m.take<float>((size_t) T * K);
        b.eids = m.take<int>((size_t) T * K);
        b.mb = m.take<int>(nl_max + 1);
        b.mb2 = m.take<int>(nl_max + 1);
        b.mcnt = m.take<int>(nl_max + 1);
        b.rows_tok = m.take<int>((size_t) T * K + strata::prefill::mmq::IDS_PAD);
        b.row_of = m.take<int>((size_t) T * K);
        b.eg_ids = m.take<int64_t>((size_t) T * ncol);
        b.emask = m.take<int>(T);
        // exchanges: one buffer each (the alternation RS -> AG -> RS .. and the waits order every reuse)
        b.ar_buf[0] = b.ar_buf[1] = m.take<float>((size_t) S * ts * E / 2 + 64);
        b.ag_buf[0] = b.ag_buf[1] = m.take<float>((size_t) T * E / 2 + 64);
        b.cb_buf[0] = b.cb_buf[1] = m.take<float>(std::getenv("HY4F_CB32") ? (size_t) S * TA * (Hd / S) * CBW()
                                                                           : (size_t) S * TA * (Hd / S) * CB_SLOT / 4);
        b.yn = m.take<__half>((size_t) ts * E);
        b.ag_flag = m.take<int>(4);
        b.ar_flag = m.take<int>(4);
        b.cb_flag = m.take<int>(4);
        b.ctr = m.take<int>(16);
        b.cl_flag = m.take<int>(4);
        if (S > 1) {
            b.cellsx[0] = m.take<int>((size_t) TA * n_cap);
            b.cellsx[1] = m.take<int>((size_t) TA * n_cap);
        }
        b.wdq = m.take<float>(std::max((size_t) Hd, (size_t) Hmax / S) * kvl * nope + (size_t) Hmax / S * vh * kvl);
        b.mr = has_mtp ? m.take<float>((size_t) T * E) : nullptr;
        b.ones = m.take<float>((size_t) T * 8);
        b.hprev = m.take<float>(E);
        b.hold = m.take<float>(E);
    };
    auto attn = [&](Bump& m) {
        b.xp = m.take<float>((size_t) T * xp_ld);
        b.qr = m.take<float>((size_t) T * q_lora);
        b.qp = m.take<float>((size_t) T * qp_ld);
        b.qabs = m.take<float>((size_t) TA * Ha * kvl);
        b.o = m.take<float>((size_t) TA * Ho * kvl);
        b.part = S > 1 ? m.take<float>((size_t) TA * Hd * CBW()) : nullptr;
        b.a = m.take<float>((size_t) T * Hmax / S * vh);
        b.ao = m.take<float>((size_t) T * E);
        b.ikn = m.take<float>((size_t) T * IK);
        b.ivg = m.take<float>((size_t) T * 2 * IK);
        b.iw = m.take<float>((size_t) T * IH);
        b.cells = m.take<int>((size_t) TA * n_cap);
    };
    auto ffn = [&](Bump& m) {
        b.lg = m.take<float>((size_t) T * n_exp);
        b.xm = m.take<float>((size_t) T * xm_ld);
        b.gu = m.take<float>(std::max((size_t) b.cap * n_ffx, (size_t) T * 2 * E));   // (h rows: run_glu writes n_ff; the MTP's concat)
        b.dn = m.take<float>((size_t) b.cap * latent);
        b.moe = m.take<float>((size_t) T * latent);
        b.mu = m.take<float>((size_t) T * E);
        b.shh = m.take<float>((size_t) T * std::max(n_ffx / S, n_ff / S));
        b.shd = m.take<float>((size_t) T * std::max((size_t) E, (size_t) 2 * n_ff / S));
        b.ffo = m.take<float>((size_t) T * E);
    };
    auto engr = [&](Bump& m) {
        b.eg_r = m.take<float>((size_t) T * ncol * eg_D);
        b.eg_kv = m.take<float>((size_t) T * (hc + 1) * E);
    };
    Bump m0, ma, mf, me;
    persistent(m0);
    attn(ma);
    ffn(mf);
    engr(me);
    const size_t arena = std::max({ma.off, mf.off, me.off});
    uint8_t* base = nullptr;
    if (o.verbose)
        std::fprintf(stderr, "hy4f: batch buffers for %d tokens: persistent %.0f MiB, arena %.0f MiB (attention %.0f, ffn %.0f, engram %.0f)\n",
                     T, m0.off / 1048576.0, arena / 1048576.0, ma.off / 1048576.0, mf.off / 1048576.0, me.off / 1048576.0);
    HY_CK(cudaMalloc(&base, m0.off + arena));
    HY_CK(cudaMemset(base, 0, m0.off + arena));
    m0 = Bump{base, 0};
    persistent(m0);
    ma = Bump{base + m0.off, 0};
    mf = ma;
    me = ma;
    attn(ma);
    ffn(mf);
    engr(me);
    b.bytes = m0.off + arena;
    b.arena_base = base + m0.off;
    b.arena_bytes = arena;
    {
        std::vector<float> one((size_t) T * 8, 1.0f);
        HY_CK(cudaMemcpy(b.ones, one.data(), sizeof(float) * one.size(), cudaMemcpyHostToDevice));
    }
    b.pp = b.dbounds + 2;
    strata::prefill::mmq::iota(b.iota, (int64_t) T * K + strata::prefill::mmq::IDS_PAD, b.st);
}

void batch_setup(int Tmax) {
    bsh.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) batch_alloc(sh[i], bsh[i], Tmax);
    Peers ar{}, cb{}, ag{}, cl{};
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
        cl.buf[j][0] = reinterpret_cast<float*>(bsh[j].cellsx[0]);
        cl.buf[j][1] = reinterpret_cast<float*>(bsh[j].cellsx[1]);
        cl.flag[j] = bsh[j].cl_flag;
    }
    for (auto& b : bsh) b.ar = ar, b.cb = cb, b.ag = ag, b.cl = cl;
    if (o.verbose)
        std::fprintf(stderr, "hy4f: batch buffers for %d tokens: %.0f MiB per GPU\n", Tmax, bsh[0].bytes / 1048576.0);
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
    mq::quantize(x, nullptr, b.xq, w.type, cols, ldx, T, b.st);
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
    b.mmq->run(p, b.st);
}
void blin(Shard& s, BShard& b, const std::string& n, const float* x, int ldx, float* y, int ldy, int T) {
    blin(s, b, s.w.at(n), x, ldx, y, ldy, T);
}

// the payload to every shard, one peer at a time in ring order (this shard last)
void push(Shard& s, BShard& b, const float* src, size_t per, int mode, int Hl, int W, const Peers& P, int* ctr) {
    const int blocks = std::min(nb((int) (per / 4), 512), 160);
    for (int r = 1; r <= S; ++r) {
        const int j = (s.g + r) % S;
        kb_push<<<blocks, 512, 0, b.st>>>(src, per, mode, 0, Hl, W, P, s.g, S, b.step, ctr, j);
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
    float* X = b.mtp ? b.mr : b.X;
    const float* post = b.mtp ? b.ones : b.post;
    const int hcn = b.mtp ? 1 : hc;
    if (S == 1) {
        kb_ar_post<<<T, 1024, 0, b.st>>>(y, E, b.ar, s.g, S, b.step, f32(s, norm), eps, X, X, post, hcn, (size_t) T * E);
        return;
    }
    const int ts = (T + S - 1) / S, blocks = std::min(nb(ts * E / 8, 512), 160);
    ++b.step;
    kb_rs_push<<<blocks, 512, 0, b.st>>>(y, T, E, ts, b.ar, s.g, b.step, b.ctr, s.g);   // (this shard's own slice)
    for (int r = 1; r < S; ++r) {
        const int j = peer_of(s.g, r);
        kb_rs_push<<<blocks, 512, 0, b.st>>>(y, T, E, ts, b.ar, s.g, b.step, b.ctr, j);
        if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.ar, s.g, j, b.step);
    }
    kb_wait_k<<<1, 32, 0, b.st>>>(b.ar, s.g, S, b.step);
    kb_rs_post<<<ts, 1024, 0, b.st>>>(T, E, ts, b.ar, s.g, S, b.step, f32(s, norm), eps, b.yn);
    ++b.step;
    kb_ag_push<<<blocks, 512, 0, b.st>>>(b.yn, T, E, ts, b.ag, s.g, b.step, b.ctr + 8, s.g);
    for (int r = 1; r < S; ++r) {
        const int j = peer_of(s.g, r);
        kb_ag_push<<<blocks, 512, 0, b.st>>>(b.yn, T, E, ts, b.ag, s.g, b.step, b.ctr + 8, j);
        if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.ag, s.g, j, b.step);
    }
    kb_wait_k<<<1, 32, 0, b.st>>>(b.ag, s.g, S, b.step);
    kb_ag_post<<<T, 1024, 0, b.st>>>(E, b.ag, s.g, S, b.step, X, X, post, hcn);
}

void b_norm_in(Shard& s, BShard& b, const std::string& P, const std::string& k, const std::string& norm, int n_mix,
               float mag, int T, size_t t0 = 0) {
    const float* X = b.X + t0 * hc * E;
    kb_hc_mix<<<dim3(HC_SPLIT, T), 256, 0, b.st>>>(X, f32(s, P + k + "_fn.weight"), hc * E, n_mix, b.hcp, b.hcs);
    kb_hc_apply_norm<<<T, 1024, 0, b.st>>>(X, b.hcp, b.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                           f32(s, norm), E, hc, n_mix, eps, hc_eps, mag, b.x, b.post);
}

void b_engram(Shard& s, BShard& b, int e, int il, int pos0, int T) {
    const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
    const std::string P = "blk." + std::to_string(il) + ".";
    kb_engram_hash<<<T, 32, 0, b.st>>>(s.hist, b.dbounds + 2, s.eg_map, (int64_t) eg_map.size(), eg_pad, s.eg_mult + (size_t) e * ng,
                                       s.eg_vsz + (size_t) e * ncol, s.eg_off + (size_t) e * ncol, ng, nh, b.eg_ids);
    kb_engram_rows<<<dim3(ncol, T), 128, 0, b.st>>>(eg_table, b.eg_ids, f32(s, P + "engram_lookup_norm.weight"), b.eg_r,
                                                    eg_D, ncol, 1.1920929e-07f);
    blin(s, b, P + "engram_kv.weight", b.eg_r, ncol * eg_D, b.eg_kv, (hc + 1) * E, T);
    kb_engram_mix<<<T, 1024, 0, b.st>>>(b.X, b.R, b.eg_kv, f32(s, P + "engram_kv_b.weight"), f32(s, P + "engram_gate.weight"),
                                        f32(s, P + "engram_hc_scale.weight"), f32(s, P + "engram_pre_gamma.weight"), E, hc,
                                        1.1920929e-07f, b.emask);
}

void b_attention(Shard& s, BShard& b, int il, int pos0, int T) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const bool sw = swa[il] != 0;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
    if (b.mtp) kb_rms<<<T, 1024, 0, b.st>>>(b.mr, E, f32(s, P + "attn_norm.weight"), E, eps, b.x, E);
    else b_norm_in(s, b, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
    blin(s, b, P + "xproj", b.x, E, b.xp, (int) xp_ld, T);
    kb_rms<<<T, 1024, 0, b.st>>>(b.xp + oq, (int) xp_ld, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, b.qr, q_lora);
    kb_kv_prep<<<T, 576, 0, b.st>>>(b.xp + okv, (int) xp_ld, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th,
                                    sw ? b.chunk_cur + (size_t) b.chunk_off * KV_RB : s.kv[il], b.pp, sw ? 1 : 2, S, s.g, 0);
    blin(s, b, P + "qproj", b.qr, q_lora, b.qp, (int) qp_ld, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const int Ha = sw ? Hl : Hd;   // heads absorbed / attended on this shard
    const bool mqa = sw ? (Hl == 16 || Hl == 32) : (Hd == 32 && S > 1);
    const int TA = mqa ? std::min(T, attn_tile) : T;
    float* iq = b.qp + (size_t) Hd * QH;
    const int64_t ld_sc = o.max_context / kpool + 1;
    if (!sw) {   // the indexer: pooled keys of the groups completed in the chunk (selection per tile below)
        kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, b.st>>>(iq, (int) qp_ld, IH, IK, IK - rope, rope, b.pp, th);
        kb_rms<<<T, 128, 0, b.st>>>(b.xp + oik, (int) xp_ld, f32(s, P + "indexer.k_norm.weight"), IK, eps, b.ikn, IK);
        const DW& ivw = s.w.at(P + "ivg");
        kb_q8_gemv<<<dim3(nb(2 * IK * 32), T), 256, 0, b.st>>>((const uint8_t*) ivw.d, b.ikn, IK, b.ivg, 2 * IK, IK, 2 * IK);
        const int g_first = pos0 / kpool, n_groups = (pos0 + T) / kpool - g_first;
        if (n_groups > 0)
            kb_pool_groups<<<n_groups, 128, 0, b.st>>>(b.ivg, s.ring[il], b.pp, T,
                                                       f32(s, P + "indexer_compressor_ape.weight"),
                                                       f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK,
                                                       kpool, rope, eps, th);
        kb_ring_chunk<<<IRING, 256, 0, b.st>>>(b.ivg, s.ring[il], T, b.pp, IK, kpool);
        blin(s, b, P + "indexer.proj.weight", b.x, E, b.iw, IH, T);
        k_scale<<<nb(T * IH), 256, 0, b.st>>>(b.iw, T * IH, 1.0f / std::sqrt((float) (IK * IH)));
    }
    kb_deq<<<std::min(nb(Ha * kvl * nope), 1024), 256, 0, b.st>>>(kb.d, kb.type, (size_t) Ha * kvl * nope, b.wdq);
    float* wvb = b.wdq + (size_t) Ha * kvl * nope;   // (v_b after k_b in the scratch)
    kb_deq<<<std::min(nb(Hl * vh * kvl), 1024), 256, 0, b.st>>>(vb.d, vb.type, (size_t) Hl * vh * kvl, wvb);
    const float* sinks = f32(s, P + "attn_sinks.weight");
    // b.cells is otherwise unused during SWA. Stay inside its allocation even
    // when loading short contexts or using nonstandard tile/chunk sizes.
    const int first_row = std::max(0, pos0 - win + 1);
    const int expand_rows = pos0 + T - first_row;
    const __half* expanded = nullptr;
    if ((mqa_opt & 2) && sw && mqa && b.chunk_cur == b.chunk && b.pp_attn == b.pp
        && (size_t)expand_rows * KV_ROW * sizeof(__half) <= (size_t)std::min(b.Tmax, attn_tile) * n_cap * sizeof(int)) {
        auto* dst = reinterpret_cast<__half*>(b.cells);
        kb_swa_expand<<<expand_rows, 96, 0, b.st>>>(s.kv[il], b.chunk_cur, b.pp_attn, win, first_row, dst);
        expanded = dst;
    }
    for (int t0 = 0; t0 < T; t0 += TA) {
        const int tt = std::min(TA, T - t0);
        // q_abs[t][h] = k_b_h q_nope[t][h]: k_b_h [kvl][nope] row-major
        bgemm(b, kvl, tt, nope, b.wdq, nope, (long long) kvl * nope, b.qp + (size_t) t0 * qp_ld, (int) qp_ld, QH, b.qabs,
              Ha * kvl, kvl, Ha);
        const float* qt = b.qp + (size_t) t0 * qp_ld;
        if (sw) {
            if (Hl == 16)
                kb_mqa_attn<16><<<tt, 256, mqa_smem<16>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk_cur,
                                                                 nullptr, 0, b.pp_attn, b.toff + t0, win, 1, 0, sinks + s.g * Hl, scale,
                                                                 b.o, b.part, CBW(), expanded, first_row, (mqa_opt & 1) && Hd >= Hl);
            else if (Hl == 32)
                kb_mqa_attn<32><<<tt, 256, mqa_smem<32>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk_cur,
                                                                 nullptr, 0, b.pp_attn, b.toff + t0, win, 1, 0, sinks + s.g * Hl, scale,
                                                                 b.o, b.part, CBW(), expanded, first_row, (mqa_opt & 1) && Hd >= Hl);
            else
                kb_attn2<<<dim3(Hl, T), 256, 0, b.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk, nullptr, 0,
                                                        b.pp, win, 1, 0, sinks + s.g * Hl, scale, b.o, nullptr, Hl);
        } else {
            // the indexer split (deep prompts): shard g scores / selects rows [g*q, (g+1)*q) of the tile, the cells are
            // all-gathered (one peer at a time) - the score is O(depth) and was computed by all four GPUs
            const bool split = mqa && S > 1 && pos0 + t0 >= idx_split_from;
            int* cells = b.cells;
            int ra = 0, rz = tt;
            const int q = (tt + S - 1) / S;
            if (split) {
                ++b.step;
                cells = b.cellsx[b.step & 1];
                ra = std::min(tt, s.g * q);
                rz = std::min(tt, (s.g + 1) * q);
            }
            for (int s0 = t0 + ra; s0 < t0 + rz; s0 += score_tile) {
                const int st = std::min(score_tile, t0 + rz - s0);
                const int n_pool_max = (pos0 + s0 + st - 1) / kpool;
                if (pos0 + s0 + st > top_k && n_pool_max > 0) {
                    if (IH == 16 && IK == 128 && !std::getenv("HY4F_OLD_SCORE"))
                        kb_score_tc<<<dim3(std::min(32, (n_pool_max + 255) / 256), (st + 7) / 8), 256, 0, b.st>>>(
                            iq, (int) qp_ld, s.pooled[il], b.iw, IH, kpool, top_k, b.pp, s0, st, n_pool_max, b.score,
                            (int) ld_sc);
                    else
                        kb_score<<<dim3(nb(n_pool_max * 32), st), 256, 0, b.st>>>(iq, (int) qp_ld, s.pooled[il], b.iw, IH,
                                                                                  IK, IH, kpool, top_k, b.pp, s0, b.score,
                                                                                  (int) ld_sc);
                }
                kb_select_kpool<1024><<<st, 1024, 0, b.st>>>(b.score, (int) ld_sc, b.pp, s0, t0, kpool, top_k,
                                                             top_k / kpool - tail, n_cap, cells);
            }
            if (split) {
                const size_t per = (size_t) q * n_cap;
                const int blocks = std::min(nb((int) (per / 4), 512), 64);
                for (int r = 0; r < S; ++r) {   // (r = 0: this shard - a copy onto itself that sets its own flag)
                    const int j = r == 0 ? s.g : peer_of(s.g, r);
                    kb_push<<<blocks, 512, 0, b.st>>>(reinterpret_cast<const float*>(cells + (size_t) s.g * per), per, 0, 0, 0,
                                                      0, b.cl, s.g, S, b.step, b.ctr + 12, j);
                    if (paired && r > 0) kb_wait_one<<<1, 32, 0, b.st>>>(b.cl, s.g, j, b.step);
                }
                kb_wait_k<<<1, 32, 0, b.st>>>(b.cl, s.g, S, b.step);
            }
            if (mqa) {
                kb_mqa_attn<32><<<tt, 256, mqa_smem<32>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], nullptr,
                                                                 cells, n_cap, b.pp, t0, win, S, s.g, nullptr, scale,
                                                                 nullptr, b.part, CBW(), nullptr, 0, mqa_opt & 1);
                ++b.step;
                if (std::getenv("HY4F_CB32")) {
                    const size_t per = (size_t) tt * Hl * CBW();
                    push(s, b, b.part, per, 1, Hl, CBW(), b.cb, b.ctr + 4);
                    kb_combine<<<tt, 512, 0, b.st>>>(kvl, CBW(), s.g * Hl, Hl, sinks, b.cb, s.g, S, b.step, per, b.o);
                } else {   // fp16 slots: half the bytes (the local partial goes through the same path)
                    const int blocks = std::min(nb(tt * Hl * 256, 512), 160);
                    kb_push_cb16<<<blocks, 512, 0, b.st>>>(b.part, tt, Hd, CBW(), Hl, b.cb, s.g, b.step, b.ctr + 4, s.g);
                    for (int r = 1; r < S; ++r) {
                        const int j = peer_of(s.g, r);
                        kb_push_cb16<<<blocks, 512, 0, b.st>>>(b.part, tt, Hd, CBW(), Hl, b.cb, s.g, b.step, b.ctr + 4, j);
                        if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.cb, s.g, j, b.step);
                    }
                    kb_wait_k<<<1, 32, 0, b.st>>>(b.cb, s.g, S, b.step);
                    kb_combine16<<<tt, 256, 0, b.st>>>(kvl, s.g * Hl, Hl, sinks, b.cb, s.g, S, b.step, tt, b.o);
                }
            } else if (S > 1) {
                kb_attn2<<<dim3(Hd, T), 256, 0, b.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], nullptr, b.cells,
                                                        n_cap, b.pp, win, S, s.g, nullptr, scale, nullptr, b.part, Hd);
                ++b.step;
                const size_t per = (size_t) T * Hl * CBW();
                push(s, b, b.part, per, 1, Hl, CBW(), b.cb, b.ctr + 4);
                kb_combine<<<T, 512, 0, b.st>>>(kvl, CBW(), s.g * Hl, Hl, sinks, b.cb, s.g, S, b.step, per, b.o);
            } else {
                kb_attn2<<<dim3(Hd, T), 256, 0, b.st>>>(b.qabs, b.qp, (int) qp_ld, QH, nope, th, s.kv[il], nullptr, b.cells,
                                                        n_cap, b.pp, win, 1, 0, sinks, scale, b.o, nullptr, Hd);
            }
        }
        // a[t][h] = v_b_h o[t][h] (v_b_h [vh][kvl] row-major), then the sigmoid gate
        bgemm(b, vh, tt, kvl, wvb, kvl, (long long) vh * kvl, b.o, Hl * kvl, kvl, b.a + (size_t) t0 * Hl * vh, Hl * vh, vh, Hl);
        kb_gate_mul<<<dim3(nb(Hl * vh), tt), 256, 0, b.st>>>(b.a + (size_t) t0 * Hl * vh, Hl * vh,
                                                             b.xp + og + (size_t) t0 * xp_ld, (int) xp_ld, tt);
    }
    if (sw && b.ring_T > 0)
        kb_ring_write<<<std::min(b.ring_T, win), 152, 0, b.st>>>(b.chunk_cur, s.kv[il], b.ring_T, b.pp_attn, win);
    blin(s, b, P + "attn_output.weight", b.a, Hl * vh, b.ao, E, T);
    b_ar_post(s, b, b.ao, P + "post_attention_norm.weight", T);
}

void b_ffn(Shard& s, BShard& b, int il, int T) {
    namespace mq = strata::prefill::mmq;
    const std::string P = "blk." + std::to_string(il) + ".";
    if (b.mtp) kb_rms<<<T, 1024, 0, b.st>>>(b.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, b.x, E);
    else b_norm_in(s, b, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag, T);
    if (il < dense_lead) {
        const int fl = n_ff / S;
        blin(s, b, P + "dense_gu", b.x, E, b.shd, 2 * fl, T);
        kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, b.st>>>(b.shd, 2 * fl, fl, T, b.shh);
        blin(s, b, P + "ffn_down.weight", b.shh, fl, b.ffo, E, T);
        b_ar_post(s, b, b.ffo, P + "post_ffw_norm.weight", T);
        return;
    }
    const int K = n_used, e0 = ebound[s.g], nl = ebound[s.g + 1] - ebound[s.g], fl = n_ffx / S;
    blin(s, b, P + "ffn_gate_inp.weight", b.x, E, b.lg, n_exp, T);
    router(s, b.lg, f32(s, P + "exp_probs_b.bias"), T, b.eids, b.ew, b.st);
    kb_moe_count<<<nl, 256, 0, b.st>>>(b.eids, T * K, e0, b.mcnt);
    kb_moe_scan<<<1, std::max(nl, 32), 0, b.st>>>(b.mcnt, nl, b.mb);
    kb_moe_place<<<nl, 256, 0, b.st>>>(b.eids, T * K, K, e0, nl, b.mb, b.rows_tok, b.row_of);
    HY_CK(cudaMemcpyAsync(b.hmb, b.mb, sizeof(int) * (nl + 1), cudaMemcpyDeviceToHost, b.st));
    blin(s, b, P + "xmoe", b.x, E, b.xm, (int) xm_ld, T);
    HY_CK(cudaStreamSynchronize(b.st));   // (the xmoe product queued behind it keeps the GPU busy meanwhile)
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
            kb_shift_bounds<<<nb(eb - ea + 1), 256, 0, b.st>>>(b.mb + ea, b.mb2, eb - ea);
            bounds = b.mb2;
        }
        if (rows > 0) {
            mq::quantize(b.xm, b.rows_tok + ra, b.xq, wgu.type, latent, (int64_t) xm_ld, rows, b.st);
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
            p.ld_dst = n_ffx;   // (the fused product writes h = silu(g) * u, n_ff values per row)
            b.mmq->run_glu(p, clampv, b.st);
            mq::quantize(b.gu, nullptr, b.xq2, wd.type, n_ffx, n_ffx, rows, b.st);
            p.w = static_cast<const uint8_t*>(wd.d) + (size_t) ea * wd.rb * (size_t) wd.ne[1];
            p.type = wd.type;
            p.w_rows = latent;
            p.w_cols = n_ffx;
            p.expert_bytes = wd.rb * (size_t) wd.ne[1];
            p.xq = b.xq2;
            p.dst = b.dn;
            p.ld_dst = latent;
            b.mmq->run(p, b.st);
        }
        kb_moe_combine<<<dim3(nb(latent), T), 256, 0, b.st>>>(b.dn, latent, b.row_of, b.ew, K, b.moe, latent, ra, rb, ea > 0);
        ea = eb;
    }
    blin(s, b, P + "ffn_routed_up.weight", b.moe, latent, b.mu, E, T);
    kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, b.st>>>(b.xm + latent, (int) xm_ld, fl, T, b.shh);
    blin(s, b, P + "ffn_down_shexp.weight", b.shh, fl, b.shd, E, T);
    kb_add<<<nb(T * E), 256, 0, b.st>>>(b.mu, b.shd, b.ffo, (size_t) T * E);
    b_ar_post(s, b, b.ffo, P + "post_ffw_norm.weight", T);
}

// one chunk on one shard: tokens b.toks (already on the device), positions pos0 ..; logits of the last token
struct Span {   // embedding rows over the chunk's token rows [t0, t0 + n) (host [n][E])
    int t0, n;
    const float* rows;
};
void b_chunk(Shard& s, BShard& b, int pos0, int T, bool want_logits, const std::vector<Span>& spans = {}) {
    b.hb[0] = 0;
    b.hb[1] = T;
    b.hb[2] = pos0;
    b.hb[3] = pos0 > 0 ? pos0 - 1 : 0;   // the MTP rows' first position (it lags the trunk by one)
    HY_CK(cudaMemcpyAsync(b.dbounds, b.hb, sizeof(int) * 4, cudaMemcpyHostToDevice, b.st));
    b.mtp = false;
    b.pp = b.dbounds + 2;
    b.chunk_cur = b.chunk, b.chunk_off = 0, b.pp_attn = b.pp, b.toff = 0, b.ring_T = T;
    const DW& emb = s.w.at("token_embd.weight");
    kb_embed<<<dim3(nb(E), T), 256, 0, b.st>>>((const uint8_t*) emb.d, emb.rb, b.toks, b.dbounds + 2, b.X, b.R, E, hc, s.hist);
    HY_CK(cudaMemsetAsync(b.emask, 0, sizeof(int) * T, b.st));
    for (const Span& sp : spans) {   // (staged through the FFN arena: free at the head of the chunk)
        HY_CK(cudaMemcpyAsync(b.gu, sp.rows, sizeof(float) * (size_t) sp.n * E, cudaMemcpyHostToDevice, b.st));
        kb_embed_rows<<<dim3(nb(E), sp.n), 256, 0, b.st>>>(b.gu, sp.t0, b.dbounds + 2, b.X, E, hc, s.hist, b.emask);
    }
    for (int il = 0; il < L; ++il) {
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) b_engram(s, b, (int) e, il, pos0, T);
        b_attention(s, b, il, pos0, T);
        b_ffn(s, b, il, T);
    }
    // the final hidden states (hc head + output norm): every row when the MTP layer follows, else the last one
    const int th = has_mtp ? T : 1;
    b_norm_in(s, b, "", "output_hc", "output_norm.weight", hc, 0.0f, th, (size_t) (T - th));
    const float* hlast = b.x + (size_t) (th - 1) * E;
    if (want_logits) {
        {
            const DW& ow = s.w.at("output.weight");
            if (ow.type == T_BF16)
                kn::bf16_gemv_fp32_mmvf(hlast, reinterpret_cast<const uint16_t*>(ow.d), s.logits, E, (int) ow.ne[1], b.st);
            else
                throw std::runtime_error("hy4f: output.weight is not BF16");
        }
        HY_CK(cudaMemcpyAsync(hlogits + (size_t) s.g * (V / S), s.logits, sizeof(float) * (V / S), cudaMemcpyDeviceToHost,
                              b.st));
    }
    if (!has_mtp) return;
    // the MTP layer over (h[p], token[p+1]) for p = pos0-1 .. pos0+T-2 (the chunk's last row waits for the next token)
    const int Tm = pos0 > 0 ? T : T - 1;
    HY_CK(cudaMemcpyAsync(b.hold, b.hprev, sizeof(float) * E, cudaMemcpyDeviceToDevice, b.st));   // (the old one, kept)
    HY_CK(cudaMemcpyAsync(b.hprev, hlast, sizeof(float) * E, cudaMemcpyDeviceToDevice, b.st));
    if (!dsh.empty()) HY_CK(cudaMemcpyAsync(dsh[s.g].h, hlast, sizeof(float) * E, cudaMemcpyDeviceToDevice, b.st));
    if (Tm <= 0) return;
    if (pos0 > 0) b_mtp(s, b, b.hold, T - 1, b.toks, Tm, pos0 - 1);
    else b_mtp(s, b, nullptr, T - 1, b.toks + 1, Tm, 0);
}

// the MTP layer over Tm rows at positions *dbounds[3] ..: h = [hrow0 (if given); this pass's x rows 0 .. nx-1], the next
// tokens toks[0 .. Tm)
void b_mtp(Shard& s, BShard& b, const float* hrow0, int nx, const int* toks, int Tm, int mpos0) {
    float* hin = b.ao;   // (attention arena: free until the MTP attention)
    const int r0 = hrow0 ? 1 : 0;
    if (hrow0) HY_CK(cudaMemcpyAsync(hin, hrow0, sizeof(float) * E, cudaMemcpyDeviceToDevice, b.st));
    if (nx > 0) HY_CK(cudaMemcpyAsync(hin + (size_t) r0 * E, b.x, sizeof(float) * (size_t) nx * E, cudaMemcpyDeviceToDevice, b.st));
    const std::string P = "blk." + std::to_string(L) + ".";
    const DW& emb = s.w.at("token_embd.weight");
    float* me = b.mu;    // (FFN arena)
    float* cat = b.gu;
    kd_embed_rows<<<dim3(nb(E), Tm), 256, 0, b.st>>>((const uint8_t*) emb.d, emb.rb, toks, E, me);
    kb_rms<<<Tm, 1024, 0, b.st>>>(me, E, f32(s, P + "nextn.enorm.weight"), E, eps, cat, 2 * E);
    kb_rms<<<Tm, 1024, 0, b.st>>>(hin, E, f32(s, P + "nextn.hnorm.weight"), E, eps, cat + E, 2 * E);
    blin(s, b, P + "nextn.eh_proj.weight", cat, 2 * E, b.mr, E, Tm);
    b.mtp = true;
    b.pp = b.dbounds + 3;
    b_attention(s, b, L, mpos0, Tm);
    b_ffn(s, b, L, Tm);
    b.mtp = false;
    b.pp = b.dbounds + 2;
}

// ---- EXPERIMENTAL (off; HY4F_MB=2): measured slower than one batch (1466 vs 1838 tok/s at 6K) and the threaded form
// breaks parity (the MMQ scratch pool is per device and not thread safe).  Kept for the DMA-exchange follow-up.
// two micro-batches: the chunk's halves A (rows 0 .. TA) and B on two streams, half a layer apart, so one's
// exchanges run while the other computes.  B's attention waits for A's (its KV, pooled keys and indexer ring rows);
// A's attention at layer l waits for B's at l-2 (the SWA chunk buffers alternate by layer parity).  The waits on peers
// are single-block kernels (a waiting stream holds one SM, never the other stream's).
struct MbState {
    BShard B;
    uint8_t* chunk[2] = {nullptr, nullptr};   // the SWA chunk buffers [Tmax rows], by layer parity
    std::vector<cudaEvent_t> akv, battn;      // per layer: A's attention done, B's attention done
    cudaEvent_t aemb = nullptr, ah = nullptr, bh = nullptr, done = nullptr;
};
std::vector<MbState> mbs;

void b_chunk2(Shard& s, BShard& A, MbState& M, int pos0, int T, bool want_logits, const std::vector<Span>& spA,
              const std::vector<Span>& spB) {
    BShard& B = M.B;
    const int TA = T / 2, TB = T - TA, pB = pos0 + TA;
    // the halves run on two host threads (the MoE reads its expert bounds back to the host: one thread would wait for
    // one half while the other half's work is not queued); an event is waited on only after the other thread recorded
    // it this chunk (host counters)
    std::atomic<int> a_rec{-1}, b_rec{-1}, a_stage{0};
    std::mutex mu;
    std::condition_variable cv;
    auto mark = [&](std::atomic<int>& c, int v) {
        {
            std::lock_guard<std::mutex> lk(mu);
            c.store(v);
        }
        cv.notify_all();
    };
    auto await = [&](std::atomic<int>& c, int v) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return c.load() >= v; });
    };
    const DW& emb = s.w.at("token_embd.weight");
    auto embed = [&](BShard& b, int Tn, const std::vector<Span>& sp) {
        kb_embed<<<dim3(nb(E), Tn), 256, 0, b.st>>>((const uint8_t*) emb.d, emb.rb, b.toks, b.dbounds + 2, b.X, b.R, E, hc,
                                                    s.hist);
        HY_CK(cudaMemsetAsync(b.emask, 0, sizeof(int) * Tn, b.st));
        for (const Span& x : sp) {
            HY_CK(cudaMemcpyAsync(b.gu, x.rows, sizeof(float) * (size_t) x.n * E, cudaMemcpyHostToDevice, b.st));
            kb_embed_rows<<<dim3(nb(E), x.n), 256, 0, b.st>>>(b.gu, x.t0, b.dbounds + 2, b.X, E, hc, s.hist, b.emask);
        }
    };
    A.hb[0] = 0, A.hb[1] = TA, A.hb[2] = pos0, A.hb[3] = pos0 > 0 ? pos0 - 1 : 0;
    B.hb[0] = 0, B.hb[1] = TB, B.hb[2] = pB, B.hb[3] = pB - 1;
    A.mtp = B.mtp = false;
    A.pp = A.dbounds + 2;
    B.pp = B.dbounds + 2;
    std::string err_b;
    std::thread tb([&] {
        try {
            HY_CK(cudaSetDevice(s.dev));
            HY_CK(cudaMemcpyAsync(B.dbounds, B.hb, sizeof(int) * 4, cudaMemcpyHostToDevice, B.st));
            embed(B, TB, spB);
            await(a_stage, 1);
            HY_CK(cudaStreamWaitEvent(B.st, M.aemb, 0));   // (B's Engram reads A's history; B reads A's dbounds)
            for (int il = 0; il < L; ++il) {
                B.chunk_cur = M.chunk[il & 1], B.chunk_off = TA, B.pp_attn = A.dbounds + 2, B.toff = TA, B.ring_T = T;
                for (size_t e = 0; e < eg_layers.size(); ++e)
                    if (eg_layers[e] == il) b_engram(s, B, (int) e, il, pB, TB);
                await(a_rec, il);
                HY_CK(cudaStreamWaitEvent(B.st, M.akv[il], 0));
                b_attention(s, B, il, pB, TB);
                HY_CK(cudaEventRecord(M.battn[il], B.st));
                mark(b_rec, il);
                b_ffn(s, B, il, TB);
            }
            const int thB = has_mtp ? TB : 1;
            b_norm_in(s, B, "", "output_hc", "output_norm.weight", hc, 0.0f, thB, (size_t) (TB - thB));
            const float* hlastB = B.x + (size_t) (thB - 1) * E;
            if (want_logits) {
                const DW& ow = s.w.at("output.weight");
                if (ow.type != T_BF16) throw std::runtime_error("hy4f: output.weight is not BF16");
                kn::bf16_gemv_fp32_mmvf(hlastB, reinterpret_cast<const uint16_t*>(ow.d), s.logits, E, (int) ow.ne[1], B.st);
                HY_CK(cudaMemcpyAsync(hlogits + (size_t) s.g * (V / S), s.logits, sizeof(float) * (V / S),
                                      cudaMemcpyDeviceToHost, B.st));
            }
            if (has_mtp) {
                await(a_stage, 2);
                HY_CK(cudaStreamWaitEvent(B.st, M.ah, 0));   // (A kept its old last row and handed over its new one)
                HY_CK(cudaMemcpyAsync(A.hprev, hlastB, sizeof(float) * E, cudaMemcpyDeviceToDevice, B.st));
                if (!dsh.empty()) HY_CK(cudaMemcpyAsync(dsh[s.g].h, hlastB, sizeof(float) * E, cudaMemcpyDeviceToDevice, B.st));
                await(a_stage, 3);
                HY_CK(cudaStreamWaitEvent(B.st, M.bh, 0));   // (B's MTP attention reads A's MTP KV / pooled / ring)
                b_mtp(s, B, B.hprev, TB - 1, B.toks, TB, pB - 1);
            }
            HY_CK(cudaEventRecord(M.done, B.st));
        } catch (const std::exception& e) {
            err_b = e.what();
        }
        mark(b_rec, 1 << 20);
    });
    try {
        HY_CK(cudaMemcpyAsync(A.dbounds, A.hb, sizeof(int) * 4, cudaMemcpyHostToDevice, A.st));
        embed(A, TA, spA);
        HY_CK(cudaEventRecord(M.aemb, A.st));
        mark(a_stage, 1);
        for (int il = 0; il < L; ++il) {
            A.chunk_cur = M.chunk[il & 1], A.chunk_off = 0, A.pp_attn = A.pp, A.toff = 0, A.ring_T = 0;
            if (il >= 2) {
                await(b_rec, il - 2);
                HY_CK(cudaStreamWaitEvent(A.st, M.battn[il - 2], 0));
            }
            for (size_t e = 0; e < eg_layers.size(); ++e)
                if (eg_layers[e] == il) b_engram(s, A, (int) e, il, pos0, TA);
            b_attention(s, A, il, pos0, TA);
            HY_CK(cudaEventRecord(M.akv[il], A.st));
            mark(a_rec, il);
            b_ffn(s, A, il, TA);
        }
        if (has_mtp) {
            b_norm_in(s, A, "", "output_hc", "output_norm.weight", hc, 0.0f, TA, 0);
            HY_CK(cudaMemcpyAsync(A.hold, A.hprev, sizeof(float) * E, cudaMemcpyDeviceToDevice, A.st));   // (the old last row)
            HY_CK(cudaMemcpyAsync(B.hprev, A.x + (size_t) (TA - 1) * E, sizeof(float) * E, cudaMemcpyDeviceToDevice, A.st));
            HY_CK(cudaEventRecord(M.ah, A.st));
            mark(a_stage, 2);
            if (pos0 > 0) b_mtp(s, A, A.hold, TA - 1, A.toks, TA, pos0 - 1);
            else if (TA > 1) b_mtp(s, A, nullptr, TA - 1, A.toks + 1, TA - 1, 0);
            HY_CK(cudaEventRecord(M.bh, A.st));
            mark(a_stage, 3);
        } else {
            mark(a_stage, 3);
        }
    } catch (...) {
        mark(a_rec, 1 << 20);
        mark(a_stage, 1 << 20);
        tb.join();
        throw;
    }
    tb.join();
    if (!err_b.empty()) throw std::runtime_error(err_b);
    HY_CK(cudaStreamWaitEvent(A.st, M.done, 0));   // (everything after this chunk orders after both halves)
}

void mb_setup(int Tmax) {
    mbs.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) {
        Shard& s = sh[i];
        MbState& M = mbs[i];
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaStreamCreateWithFlags(&M.B.st, cudaStreamNonBlocking));
        batch_alloc(s, M.B, Tmax);
        for (int p = 0; p < 2; ++p) HY_CK(cudaMalloc(&M.chunk[p], (size_t) 2 * Tmax * KV_RB));
        const int Lm = L + 1;
        M.akv.resize(Lm);
        M.battn.resize(Lm);
        for (int il = 0; il < Lm; ++il) {
            HY_CK(cudaEventCreateWithFlags(&M.akv[il], cudaEventDisableTiming));
            HY_CK(cudaEventCreateWithFlags(&M.battn[il], cudaEventDisableTiming));
        }
        for (cudaEvent_t* e : {&M.aemb, &M.ah, &M.bh, &M.done}) HY_CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    }
    Peers ar{}, cb{}, ag{}, cl{};
    for (int j = 0; j < S; ++j) {
        BShard& b = mbs[j].B;
        ar.buf[j][0] = b.ar_buf[0], ar.buf[j][1] = b.ar_buf[1], ar.flag[j] = b.ar_flag;
        ag.buf[j][0] = b.ag_buf[0], ag.buf[j][1] = b.ag_buf[1], ag.flag[j] = b.ag_flag;
        cb.buf[j][0] = b.cb_buf[0], cb.buf[j][1] = b.cb_buf[1], cb.flag[j] = b.cb_flag;
        cl.buf[j][0] = reinterpret_cast<float*>(b.cellsx[0]), cl.buf[j][1] = reinterpret_cast<float*>(b.cellsx[1]);
        cl.flag[j] = b.cl_flag;
    }
    for (auto& M : mbs) M.B.ar = ar, M.B.cb = cb, M.B.ag = ag, M.B.cl = cl;
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaDeviceSynchronize());
    }
}
