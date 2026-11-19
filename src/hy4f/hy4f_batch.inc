// src/hy4f/hy4f_batch.inc - maya-hy4f: the batched forward (a chunk of T tokens), included inside Model::Impl.
// The prompt path (and later the speculative verify): projections and experts through llama.cpp's MMQ (int8 tensor
// cores, prefill::mmq), F32 matrices through cuBLAS, attention by the batched kernels of hy4f_bkern.cuh, the
// exchanges as eager push / wait kernels with a host step counter.  Launched eagerly from the per-shard threads.

struct BShard {
    float* unpacked_router = nullptr;
    int Tmax = 0;
    int layer_tile = 0;
    int* tile_bounds = nullptr;
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
    int *cells = nullptr, *select_fallback = nullptr;
    unsigned *pivots = nullptr, *counts = nullptr;
    uint2* candidates = nullptr;
    float *lg = nullptr, *ew = nullptr, *xm = nullptr, *gu = nullptr, *dn = nullptr, *moe = nullptr, *mu = nullptr,
          *shh = nullptr, *shd = nullptr, *ffo = nullptr;
    int *eids = nullptr, *mb = nullptr, *mb2 = nullptr, *mcnt = nullptr, *rows_tok = nullptr, *row_of = nullptr;
    cudaEvent_t routing_ready = nullptr;
    int* hmb = nullptr;   // pinned copy of the bounds (the MMQ grid and the rows are host values)
    int64_t cap = 0;      // rows of the expert buffers
    // expert offload: the host experts of a launch.  Two buffers and a copy stream: the next launch's experts arrive
    // while the current one computes (HY4F_OFFLOAD_STAGE_OVERLAP=0: the copies run on the compute stream).
    uint8_t* stage[2] = {nullptr, nullptr};
    cudaStream_t copy_st = nullptr;
    cudaEvent_t copy_done[2] = {nullptr, nullptr}, use_done[2] = {nullptr, nullptr};
    int stage_turn = 0;
    float *eg_r = nullptr, *eg_kv = nullptr;
    int64_t* eg_ids = nullptr;
    float* ar_buf[2] = {nullptr, nullptr};   // reduce-scatter receive [src][ts][E] bf16
    float* ag_buf[2] = {nullptr, nullptr};   // all-gather receive [T][E] fp16
    __half* dsa_half = nullptr;
    __half* yn = nullptr;                    // this shard's normalized slice [ts][E] fp16
    const __half* pending_ag = nullptr;
    cudaStream_t ag_compute = nullptr;
    cudaEvent_t ag_slice[4]{}, ag_complete = nullptr;
    int overlap_rows = 0;
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
const bool fuse_hc = std::getenv("HY4F_PF_HC_FUSE") != nullptr;
bool mtp_kv_only = std::getenv("HY4F_PF_MTP_KV_ONLY") != nullptr;
const bool fuse_pack_add = std::getenv("HY4F_PF_PACK_ADD") != nullptr;
int dma_mask = std::getenv("HY4F_PF_DMA") ? std::atoi(std::getenv("HY4F_PF_DMA")) : 0;   // RS=1, AG=2, DSA=4, residual=8
bool paired = !std::getenv("HY4F_RING_PUSH");   // exchanges in XOR-paired phases (S a power of two)
int peer_of(int g, int r) const { return paired && (S & (S - 1)) == 0 ? (g ^ r) : (g + r) % S; }
int cap_num = 3, cap_den = 2;   // expert rows buffered per shard: cap_num/cap_den x the even share (ranges past it)
bool compact_arena = std::getenv("HY4F_PF_COMPACT") != nullptr;
int attn_tile = 512;
// Split the head groups and token scratch independently of the expert prefill chunk.
int single_prefill_mode() const {
    const char* setting=std::getenv("HY4F_SINGLE_PF");
    if (S!=1 || !setting) return 0;
    return !std::strcmp(setting,"mqa") ? 2 : !std::strcmp(setting,"exact") ? 1 : 0;
}
bool direct_dsa = std::getenv("HY4F_DIRECT_DSA") != nullptr;
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
    if (mbs.empty() && std::getenv("HY4F_PF_ROUTE_EVENT"))
        HY_CK(cudaEventCreateWithFlags(&b.routing_ready, cudaEventDisableTiming));
    if (std::any_of(s.w.begin(), s.w.end(), [](const auto& e) { return e.second.type == T_ROUTER_PACKED; }))
        b.unpacked_router = dalloc(size_t(n_exp) * E);
    if (!b.st) b.st = s.st;
    if (S == 4 && paired && std::getenv("HY4F_PF_AG_OVERLAP") && (dma_mask & 2)) {
        HY_CK(cudaStreamCreateWithFlags(&b.ag_compute, cudaStreamNonBlocking));
        for (auto& e : b.ag_slice) HY_CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&b.ag_complete, cudaEventDisableTiming));
    }
    if (S == 1 && !single_prefill_mode()) attn_tile = Tmax;
    const int T = Tmax, K = n_used, QH = nope + rope, Hmax = std::max(Hd, Hs);
    const char* tile_setting = std::getenv("HY4F_SINGLE_LAYER_TILE");
    b.layer_tile = offload && compact_arena && single_prefill_mode() && tile_setting
        ? std::clamp(std::atoi(tile_setting), 1, T) : T;
    const int F = b.layer_tile;
    const bool sliced = F < T;
    const int64_t C = o.max_context;
    b.mmq = new strata::prefill::mmq::Context();
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<16>()));
    HY_CK(cudaFuncSetAttribute(kb_mqa_attn<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<32>()));
    if (single_prefill_mode()==2)
        HY_CK(cudaFuncSetAttribute((kb_mqa_attn<32, 2>), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqa_smem<32>()));
    HY_BL(cublasCreate(&b.blas));
    HY_BL(cublasSetStream(b.blas, b.st));
    HY_BL(cublasSetMathMode(b.blas, CUBLAS_DEFAULT_MATH));
    HY_CK(cudaHostAlloc((void**) &b.hb, sizeof(int) * 4, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &b.hmb, sizeof(int) * (nl_max + 1), cudaHostAllocPortable));
    b.cap = std::min((int64_t) T * K, std::max<int64_t>((int64_t) cap_num * T * K / (cap_den * S), 2 * T));
    if (offload && single_prefill_mode()) {
        // Each expert occurs at most once per token. b_ffn already splits a range
        // by its actual routed row count before launching either of its matrices.
        b.cap=T;
    }
    if (offload) {
        for (int i = 0; i < 2; ++i) {
            HY_CK(cudaMalloc(&b.stage[i], off_stage / 2 + 4096));
            HY_CK(cudaMemset(b.stage[i], 0, off_stage / 2 + 4096));
            HY_CK(cudaEventCreateWithFlags(&b.copy_done[i], cudaEventDisableTiming));
            HY_CK(cudaEventCreateWithFlags(&b.use_done[i], cudaEventDisableTiming));
        }
        const char* ov = std::getenv("HY4F_OFFLOAD_STAGE_OVERLAP");
        if (!ov || std::atoi(ov)) HY_CK(cudaStreamCreateWithFlags(&b.copy_st, cudaStreamNonBlocking));
    }
    xp_ld = ((size_t) q_lora + KV_ROW + (size_t) Hmax / S * vh + IK + 3) / 4 * 4;
    qp_ld = std::max((size_t) Hd * QH + (size_t) IH * IK, (size_t) Hs / S * QH);
    xm_ld = (size_t) latent + 2 * n_ffx / S;
    const size_t q8a = std::max({strata::prefill::mmq::q8_bytes(b.cap, std::max(latent, n_ffx)),
                                 strata::prefill::mmq::q8_bytes(F, std::max({E, Hmax / S * vh, n_ff / S, 4096})),
                                 strata::prefill::mmq::q8_bytes(T, std::max(2*E, n_ffx / S)),
                                 strata::prefill::mmq::q8_bytes(F, eg_D * (eg_ngram - 1) * eg_heads)});
    const int ts = (T + S - 1) / S, ncol = (eg_ngram - 1) * eg_heads;
    const int Ha = std::max(Hd, Hs / S), Ho = S > 1 ? Hmax / S : Hmax;
    const int TA = std::min(T, attn_tile);
    // persistent
    auto persistent = [&](Bump& m) {
        b.dbounds = m.take<int>(4);   // {0, T, pos0, -}
        if (sliced) b.tile_bounds = m.take<int>(4);
        b.iota = m.take<int>((size_t) T * K + strata::prefill::mmq::IDS_PAD);
        b.toks = m.take<int>(T);
        const size_t pack_rs = (dma_mask & 1) ? (size_t)T * E * sizeof(__nv_bfloat16) : 0;
        const size_t pack_cb = (dma_mask & 4) ? (size_t)TA * Hd * CB_SLOT : 0;
        b.xq = m.take<uint8_t>(std::max({q8a, pack_rs, pack_cb}));
        b.xq2 = m.take<uint8_t>(sliced ? strata::prefill::mmq::q8_bytes(b.cap, n_ffx) : q8a);
        b.X = m.take<float>((size_t) T * hc * E);
        b.R = b.X;   // (X = R + post * y and R = X: the same values everywhere in the batched path)
        b.x = m.take<float>((size_t) T * E);
        b.post = m.take<float>((size_t) T * 8);
        b.hcp = m.take<float>((size_t) T * 8 * HC_SPLIT);
        b.hcs = m.take<float>((size_t) T * HC_SPLIT);
        b.score = m.take<float>((size_t) score_tile * (C / kpool + 1));
        b.chunk = m.take<uint8_t>((size_t) F * KV_RB);
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
        if (!sliced) {
            b.ar_buf[0] = b.ar_buf[1] = m.take<float>((size_t) S * ts * E / 2 + 64);
            b.ag_buf[0] = b.ag_buf[1] = m.take<float>((size_t) T * E / 2 + 64);
            b.cb_buf[0] = b.cb_buf[1] = m.take<float>(std::getenv("HY4F_CB32") ? (size_t) S * TA * (Hd / S) * CBW()
                                                                               : (size_t) S * TA * (Hd / S) * CB_SLOT / 4);
            b.yn = m.take<__half>((size_t) ts * E);
        }
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
        b.mr = has_mtp ? (compact_arena ? b.X : m.take<float>((size_t) T * E)) : nullptr;
        b.ones = m.take<float>((size_t) T * 8);
        b.hprev = m.take<float>(E);
        b.hold = m.take<float>(E);
    };
    auto attn = [&](Bump& m) {
        b.xp = m.take<float>((size_t) F * xp_ld);
        b.qr = m.take<float>((size_t)F * (compact_arena ? std::max(q_lora, Hmax / S * vh) : q_lora));
        b.qp = m.take<float>((size_t) F * qp_ld);
        b.qabs = m.take<float>((size_t) TA * Ha * kvl);
        b.o = m.take<float>((size_t) TA * Ho * kvl);
        if (std::getenv("HY4F_PF_DSA_HALF") && S==4 && kpool==4 && Hd==32)
            b.dsa_half=m.take<__half>(((C+S-1)/S)*KV_ROW);
        b.part = S > 1 ? m.take<float>((size_t) TA * Hd * CBW()) : nullptr;
        b.a = compact_arena ? b.qr : m.take<float>((size_t) F * Hmax / S * vh);
        b.ao = compact_arena ? b.x : m.take<float>((size_t) F * E);
        b.ikn = m.take<float>((size_t) F * IK);
        b.ivg = m.take<float>((size_t) F * 2 * IK);
        b.iw = m.take<float>((size_t) F * IH);
        b.cells = m.take<int>((size_t) TA * n_cap);
        const char* select = std::getenv("HY4F_PF_SELECT");
        if (select && std::atoi(select) && C >= 98304) {
            b.select_fallback = m.take<int>(score_tile);
            b.pivots = m.take<unsigned>(score_tile);
            b.counts = m.take<unsigned>(score_tile);
            b.candidates = m.take<uint2>((size_t)score_tile * SELECT_CAND_LIMIT);
        }
    };
    auto ffn = [&](Bump& m) {
        if (compact_arena) {
            // xm and the routed sum must survive every expert range. The other
            // FFN buffers are private either to the experts or to the final projections.
            b.xm = m.take<float>((size_t)T * xm_ld);
            b.moe = m.take<float>((size_t)T * latent);
            Bump experts = m, final = m;
            const size_t work = std::max({(size_t)b.cap * n_ffx, (size_t)b.cap * latent,
                                          (size_t)T * 2 * E, (size_t)T * n_exp});
            b.lg = b.gu = b.dn = experts.take<float>(work);
            b.mu = final.take<float>((size_t)T * E);
            b.shh = final.take<float>(std::max((size_t)T * (n_ffx / S), (size_t)F * (n_ff / S)));
            b.shd = b.ffo = final.take<float>(std::max((size_t)T * E, (size_t)F * (2 * n_ff / S)));
            m.off = std::max(experts.off, final.off);
            return;
        }
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
        b.eg_r = m.take<float>((size_t) F * ncol * eg_D);
        b.eg_kv = m.take<float>((size_t) F * (hc + 1) * E);
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
    if (w.type == T_F32 || w.type == T_ROUTER_PACKED) {
        const float* matrix = (const float*)w.d;
        if (w.type == T_ROUTER_PACKED) {
            kb_router_unpack<<<nb(rows * cols), 256, 0, b.st>>>((const uint16_t*)w.d, b.unpacked_router, size_t(rows) * cols);
            matrix = b.unpacked_router;
        }
        const float one = 1.0f, zero = 0.0f;
        HY_BL(cublasSgemm(b.blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &one, matrix, cols, x, ldx, &zero, y,
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

void b_copy(Shard& s, BShard& b, Peers& P, int j, const void* src, size_t offset, size_t len) {
    auto dst = reinterpret_cast<uint8_t*>(P.buf[j][b.step & 1]) + offset;
    if (len && j == s.g)
        HY_CK(cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToDevice, b.st));
    else if (len)
        HY_CK(cudaMemcpyPeerAsync(dst, sh[j].dev, src, s.dev, len, b.st));
    kb_copy_done<<<1, 32, 0, b.st>>>(P.flag[j], s.g, b.step);
    if (paired && j != s.g) kb_wait_one<<<1, 32, 0, b.st>>>(P, s.g, j, b.step);
}

// all-reduce of y [T][E] then RMS * w, X = R + post * y: reduce-scatter (bf16) by token slices, all-gather (fp16)
void b_ar_post(Shard& s, BShard& b, const float* y, const std::string& norm, int T, const float* addend = nullptr) {
    float* X = b.mtp ? b.mr : b.X;
    const float* post = b.mtp ? b.ones : b.post;
    const int hcn = b.mtp ? 1 : hc;
    if (S == 1) {
        kb_ar_post<<<T, 1024, 0, b.st>>>(y, E, b.ar, s.g, S, b.step, f32(s, norm), eps, X, X, post, hcn, (size_t) T * E);
        return;
    }
    const int ts = (T + S - 1) / S, blocks = std::min(nb(ts * E / 8, 512), 160);
    ++b.step;
    if (dma_mask & 1) {
        auto pack = reinterpret_cast<__nv_bfloat162*>(b.xq);
        if (addend)
            kb_rs_pack_add<<<nb(T * E / 4), 256, 0, b.st>>>(reinterpret_cast<const float4*>(y),
                    reinterpret_cast<const float4*>(addend), pack, T * E / 4);
        else kb_rs_pack<<<nb(T * E / 2), 256, 0, b.st>>>(y, pack, T * E / 2);
        for (int r = 0; r < S; ++r) {
            const int j = peer_of(s.g, r), row = std::min(T, j * ts);
            const int nr = std::min(ts, T - row);
            b_copy(s, b, b.ar, j, reinterpret_cast<uint8_t*>(pack) + (size_t)row * E * 2,
                   (size_t)s.g * ts * E * 2, (size_t)nr * E * 2);
        }
    } else {
        kb_rs_push<<<blocks, 512, 0, b.st>>>(y, T, E, ts, b.ar, s.g, b.step, b.ctr, s.g);   // (this shard's own slice)
        for (int r = 1; r < S; ++r) {
            const int j = peer_of(s.g, r);
            kb_rs_push<<<blocks, 512, 0, b.st>>>(y, T, E, ts, b.ar, s.g, b.step, b.ctr, j);
            if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.ar, s.g, j, b.step);
        }
    }
    kb_wait_k<<<1, 32, 0, b.st>>>(b.ar, s.g, S, b.step);
    kb_rs_post<<<ts, 1024, 0, b.st>>>(T, E, ts, b.ar, s.g, S, b.step, f32(s, norm), eps, b.yn);
    ++b.step;
    const bool overlap = b.ag_compute && mbs.empty() && !b.mtp && E == 3072 && hc == 4 && T >= 128;
    b.overlap_rows = overlap ? T : 0;
    if (dma_mask & 2) {
        const int row = std::min(T, s.g * ts), nr = std::min(ts, T - row);
        for (int r = 0; r < S; ++r) {
            const int j = peer_of(s.g, r);
            b_copy(s, b, b.ag, j, b.yn, (size_t)row * E * 2, (size_t)nr * E * 2);
            if (!overlap) continue;
            // In an XOR phase the peer's ACK orders its slice on this device.
            HY_CK(cudaEventRecord(b.ag_slice[j], b.st));
            if (!fuse_hc) {
                const int start = std::min(T, j * ts), len = std::min(ts, T - start);
                const __half* yn = reinterpret_cast<const __half*>(b.ag.buf[s.g][b.step & 1]);
                HY_CK(cudaStreamWaitEvent(b.ag_compute, b.ag_slice[j]));
                if (len) kb_ag_residual<<<dim3(nb(E / 4), len, hc), 256, 0, b.ag_compute>>>(
                    yn + (size_t)start * E, X + (size_t)start * hc * E, post + (size_t)start * 8, E, hcn);
            }
        }
    } else {
        kb_ag_push<<<blocks, 512, 0, b.st>>>(b.yn, T, E, ts, b.ag, s.g, b.step, b.ctr + 8, s.g);
        for (int r = 1; r < S; ++r) {
            const int j = peer_of(s.g, r);
            kb_ag_push<<<blocks, 512, 0, b.st>>>(b.yn, T, E, ts, b.ag, s.g, b.step, b.ctr + 8, j);
            if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.ag, s.g, j, b.step);
        }
    }
    kb_wait_k<<<1, 32, 0, b.st>>>(b.ag, s.g, S, b.step);
    if (overlap && !fuse_hc) {
        HY_CK(cudaEventRecord(b.ag_complete, b.ag_compute));
        HY_CK(cudaStreamWaitEvent(b.st, b.ag_complete));
        return;
    }
    if (fuse_hc && !b.mtp && hc == 4 && E == 3072 && HC_SPLIT == 16 && mbs.empty()) {
        b.pending_ag = reinterpret_cast<const __half*>(b.ag.buf[s.g][b.step & 1]);
        return;
    }
    if ((dma_mask & 8) && E % 4 == 0)
        kb_ag_residual<<<dim3(nb(E / 4), T, hcn), 256, 0, b.st>>>(
            reinterpret_cast<const __half*>(b.ag.buf[s.g][b.step & 1]), X, post, E, hcn);
    else
        kb_ag_post<<<T, 1024, 0, b.st>>>(E, b.ag, s.g, S, b.step, X, X, post, hcn);
}

void b_norm_in(Shard& s, BShard& b, const std::string& P, const std::string& k, const std::string& norm, int n_mix,
               float mag, int T, size_t t0 = 0) {
    if (b.pending_ag) {
        float* X = b.X + t0 * hc * E;
        const float* fn = f32(s, P + k + "_fn.weight");
        const float* scale = f32(s, P + k + "_scale.weight");
        const float* base = f32(s, P + k + "_base.weight");
        const float* nw = f32(s, norm);
        if (T == b.overlap_rows && t0 == 0 && T > 0) {
            const int ts = (T + S - 1) / S;
            for (int r = 0; r < S; ++r) {
                const int j = peer_of(s.g, r), row = std::min(T, j * ts), len = std::min(ts, T - row);
                HY_CK(cudaStreamWaitEvent(b.ag_compute, b.ag_slice[j]));
                if (len) kb_hc_norm_fused<true><<<len, 1024, 0, b.ag_compute>>>(
                    X + (size_t)row * hc * E, b.pending_ag + (size_t)row * E, b.post + (size_t)row * 8,
                    fn, scale, base, nw, n_mix, eps, hc_eps, mag, b.x + (size_t)row * E, b.post + (size_t)row * 8);
            }
            HY_CK(cudaEventRecord(b.ag_complete, b.ag_compute));
            HY_CK(cudaStreamWaitEvent(b.st, b.ag_complete));
        } else {
            kb_hc_norm_fused<true><<<T, 1024, 0, b.st>>>(X, b.pending_ag + t0 * E, b.post + t0 * 8,
                fn, scale, base, nw, n_mix, eps, hc_eps, mag, b.x, b.post);
        }
        b.pending_ag = nullptr;
        return;
    }
    const float* X = b.X + t0 * hc * E;
    kb_hc_mix<<<dim3(HC_SPLIT, T), 256, 0, b.st>>>(X, f32(s, P + k + "_fn.weight"), hc * E, n_mix, b.hcp, b.hcs);
    kb_hc_apply_norm<<<T, 1024, 0, b.st>>>(X, b.hcp, b.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                           f32(s, norm), E, hc, n_mix, eps, hc_eps, mag, b.x, b.post);
}

void b_engram(Shard& s, BShard& b, int e, int il, int pos0, int T) {
    if (b.pending_ag) {
        kb_ag_residual<<<dim3(nb(E / 4), T, hc), 256, 0, b.st>>>(b.pending_ag, b.X, b.post, E, hc);
        b.pending_ag = nullptr;
    }
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
    const bool cache_only = b.mtp && mtp_kv_only;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
    if (b.mtp) kb_rms<<<T, 1024, 0, b.st>>>(b.mr, E, f32(s, P + "attn_norm.weight"), E, eps, b.x, E);
    else b_norm_in(s, b, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
    blin(s, b, P + "xproj", b.x, E, b.xp, (int) xp_ld, T);
    if (!cache_only) kb_rms<<<T, 1024, 0, b.st>>>(b.xp + oq, (int) xp_ld, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, b.qr, q_lora);
    uint8_t* dedicated = full_batch_writes && !sw ? full_layer_mirror(s,il) : nullptr;
    const bool borrowed = !sw && sparse_limit && mbs.empty() && pos0+T <= sparse_limit;
    if (dedicated && borrowed)
        kb_kv_prep<true,true,true><<<T,576,0,b.st>>>(b.xp+okv,(int)xp_ld,f32(s,P+"attn_kv_a_norm.weight"),
            kvl,rope,eps,th,s.kv[il],b.pp,2,S,s.g,0,s.prefix_kv.at(il),top_k,sparse_base,sparse_pitch,sparse_skip,nullptr,dedicated,full_stride());
    else if (dedicated)
        kb_kv_prep<true,true><<<T,576,0,b.st>>>(b.xp+okv,(int)xp_ld,f32(s,P+"attn_kv_a_norm.weight"),
            kvl,rope,eps,th,s.kv[il],b.pp,2,S,s.g,0,s.prefix_kv.at(il),top_k,0,full_stride(),0,dedicated);
    else if (borrowed)
        kb_kv_prep<true,true><<<T,576,0,b.st>>>(b.xp + okv,(int)xp_ld,f32(s,P+"attn_kv_a_norm.weight"),
            kvl,rope,eps,th,s.kv[il],b.pp,2,S,s.g,0,s.prefix_kv.at(il),top_k,sparse_base,sparse_pitch,sparse_skip);
    else if (!sw && pos0 < top_k && !s.prefix_kv.empty())
        kb_kv_prep<true><<<T,576,0,b.st>>>(b.xp + okv, (int)xp_ld, f32(s,P+"attn_kv_a_norm.weight"),
            kvl,rope,eps,th,s.kv[il],b.pp,2,S,s.g,0,s.prefix_kv.at(il),top_k);
    else
        kb_kv_prep<<<T, 576, 0, b.st>>>(b.xp + okv, (int) xp_ld, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th,
                                    sw ? b.chunk_cur + (size_t) b.chunk_off * KV_RB : s.kv[il], b.pp, sw ? 1 : 2, S, s.g, 0);
    if (cache_only && sw) {
        if (b.ring_T > 0)
            kb_ring_write<<<std::min(b.ring_T, win), 152, 0, b.st>>>(b.chunk_cur, s.kv[il], b.ring_T, b.pp_attn, win);
        return;
    }
    if (!cache_only) blin(s, b, P + "qproj", b.qr, q_lora, b.qp, (int) qp_ld, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const int Ha = sw ? Hl : Hd;   // heads absorbed / attended on this shard
    const int single=single_prefill_mode();
    const bool mqa = sw ? (Hl == 16 || Hl == 32 || (single==2 && Hl==64)) : (Hd == 32 && (S>1 || single==2));
    const int TA = mqa || single ? std::min(T, attn_tile) : T;
    float* iq = b.qp + (size_t) Hd * QH;
    const int64_t ld_sc = o.max_context / kpool + 1;
    if (!sw) {   // the indexer: pooled keys of the groups completed in the chunk (selection per tile below)
        if (!cache_only) kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, b.st>>>(iq, (int) qp_ld, IH, IK, IK - rope, rope, b.pp, th);
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
        if (cache_only) return;
        blin(s, b, P + "indexer.proj.weight", b.x, E, b.iw, IH, T);
        k_scale<<<nb(T * IH), 256, 0, b.st>>>(b.iw, T * IH, 1.0f / std::sqrt((float) (IK * IH)));
    }
    if (kb.packed_projection)
        kb_deq_project<192><<<nb(Ha * kvl * 2 * 32), 256, 0, b.st>>>((const uint8_t*)kb.d, b.wdq, Ha * kvl);
    else kb_deq<<<std::min(nb(Ha * kvl * nope), 1024), 256, 0, b.st>>>(kb.d, kb.type, (size_t) Ha * kvl * nope, b.wdq);
    float* wvb = b.wdq + (size_t) Ha * kvl * nope;   // (v_b after k_b in the scratch)
    if (vb.packed_projection)
        kb_deq_project<512><<<nb(Hl * vh * 4 * 32), 256, 0, b.st>>>((const uint8_t*)vb.d, wvb, Hl * vh);
    else kb_deq<<<std::min(nb(Hl * vh * kvl), 1024), 256, 0, b.st>>>(vb.d, vb.type, (size_t) Hl * vh * kvl, wvb);
    const float* sinks = f32(s, P + "attn_sinks.weight");
    // b.cells is otherwise unused during SWA. Stay inside its allocation even
    // when loading short contexts or using nonstandard tile/chunk sizes.
    const int first_row = std::max(0, pos0 - win + 1);
    const int expand_rows = pos0 + T - first_row;
    const __half* expanded = nullptr;
    if ((mqa_opt & 6) && sw && mqa && b.chunk_cur == b.chunk && b.pp_attn == b.pp
        && (size_t)expand_rows * KV_ROW * sizeof(__half) <= (size_t)std::min(b.Tmax, attn_tile) * n_cap * sizeof(int)) {
        auto* dst = reinterpret_cast<__half*>(b.cells);
        kb_swa_expand<<<expand_rows, 96, 0, b.st>>>(s.kv[il], b.chunk_cur, b.pp_attn, win, first_row, dst);
        expanded = dst;
    }
    const __half* dsa_half = nullptr;
    if (!sw && b.dsa_half && mbs.empty() && T >= 512) {
        const int nr=(pos0+T+S-1-s.g)/S;
        if (nr) kb_dsa_expand<<<nr,96,0,b.st>>>(s.kv[il],b.dsa_half);
        dsa_half=b.dsa_half;
    }
    for (int t0 = 0; t0 < T; t0 += TA) {
        const int tt = std::min(TA, T - t0);
        // q_abs[t][h] = k_b_h q_nope[t][h]: k_b_h [kvl][nope] row-major
        bgemm(b, kvl, tt, nope, b.wdq, nope, (long long) kvl * nope, b.qp + (size_t) t0 * qp_ld, (int) qp_ld, QH, b.qabs,
              Ha * kvl, kvl, Ha);
        const float* qt = b.qp + (size_t) t0 * qp_ld;
        if (sw) {
            if ((mqa_opt & 4) && expanded && b.part && Hd >= 16 && Hl == 16 && win == 512 && pos0+t0 >= win-1) {
                auto* qh = reinterpret_cast<__half*>(b.o);
                auto* ph = reinterpret_cast<__half*>(b.qabs);
                const int base = pos0 + t0 - win + 1 - first_row;
                kb_mqa_queries<16><<<dim3(nb(16 * 576), tt),256,0,b.st>>>(b.qabs, qt, (int)qp_ld, QH, nope, th, pos0+t0, scale, qh);
                kb_swa_scores<<<dim3(16, (tt+3)/4),256,0,b.st>>>(qh, expanded, base, tt, b.part);
                kb_swa_probs<<<tt,512,0,b.st>>>(b.part,sinks+s.g*Hl,ph);
                kb_swa_values<<<dim3(4,tt),256,0,b.st>>>(ph,expanded,base,b.o);
            } else if (Hl == 16)
                kb_mqa_attn<16><<<tt, 256, mqa_smem<16>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk_cur,
                                                                 nullptr, 0, b.pp_attn, b.toff + t0, win, 1, 0, sinks + s.g * Hl, scale,
                                                                 b.o, b.part, CBW(), expanded, first_row, (mqa_opt & 1) && Hd >= Hl);
            else if (Hl == 32)
                kb_mqa_attn<32><<<tt, 256, mqa_smem<32>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], b.chunk_cur,
                                                                 nullptr, 0, b.pp_attn, b.toff + t0, win, 1, 0, sinks + s.g * Hl, scale,
                                                                 b.o, b.part, CBW(), expanded, first_row, (mqa_opt & 1) && Hd >= Hl);
            else if (single==2 && Hl==64)
                kb_mqa_attn<32,2><<<dim3(tt,2),256,mqa_smem<32>(),b.st>>>(b.qabs,qt,(int)qp_ld,QH,nope,th,s.kv[il],
                    b.chunk_cur,nullptr,0,b.pp_attn,b.toff+t0,win,1,0,sinks,scale,b.o,nullptr,CBW(),expanded,first_row);
            else
                kb_attn2<<<dim3(Hl,tt),256,0,b.st>>>(b.qabs,qt,(int)qp_ld,QH,nope,th,s.kv[il],b.chunk_cur,nullptr,0,
                    b.pp_attn,win,1,0,sinks+s.g*Hl,scale,b.o,nullptr,Hl,b.toff+t0);
        } else {
            // the indexer split (deep prompts): shard g scores / selects rows [g*q, (g+1)*q) of the tile, the cells are
            // all-gathered (one peer at a time) - the score is O(depth) and was computed by all four GPUs
            const bool split = mqa && S > 1 && pos0 + t0 >= idx_split_from;
            const int groups = top_k / kpool - tail;
            const bool compact = split && std::getenv("HY4F_PF_GROUPS") && mbs.empty()
                && (int64_t)pos0 + t0 + tt <= (int64_t)65536 * kpool && (pos0 + t0) / kpool >= groups
                && pos0 + t0 + 1 > top_k;
            const bool direct = direct_dsa && S == 4 && kpool == 4;
            const uint16_t* selected = nullptr;
            int* cells = b.cells;
            int ra = 0, rz = tt;
            const int q = (tt + S - 1) / S;
            if (split) {
                ++b.step;
                cells = compact ? b.cells : b.cellsx[b.step & 1];
                ra = std::min(tt, s.g * q);
                rz = std::min(tt, (s.g + 1) * q);
            }
            for (int s0 = t0 + ra; s0 < t0 + rz; s0 += score_tile) {
                const int st = std::min(score_tile, t0 + rz - s0);
                const int n_pool_max = (pos0 + s0 + st - 1) / kpool;
                if (pos0 + s0 + st > top_k && n_pool_max > 0) {
                    if (IH == 16 && IK == 128 && n_pool_max >= 2048 && st >= 4 && std::getenv("HY4F_SCORE_TILE")) {
                        kb_score_prefill<<<dim3(std::min(64, (n_pool_max + 63) / 64), (st + 3) / 4), 256, 0, b.st>>>(
                            iq, (int)qp_ld, s.pooled[il], b.iw, IH, kpool, top_k, b.pp, s0, st, n_pool_max, b.score, (int)ld_sc);
                    } else if (IH == 16 && IK == 128 && !std::getenv("HY4F_OLD_SCORE"))
                        kb_score_tc<<<dim3(std::min(32, (n_pool_max + 255) / 256), (st + 7) / 8), 256, 0, b.st>>>(
                            iq, (int) qp_ld, s.pooled[il], b.iw, IH, kpool, top_k, b.pp, s0, st, n_pool_max, b.score,
                            (int) ld_sc);
                    else
                        kb_score<<<dim3(nb(n_pool_max * 32), st), 256, 0, b.st>>>(iq, (int) qp_ld, s.pooled[il], b.iw, IH,
                                                                                  IK, IH, kpool, top_k, b.pp, s0, b.score,
                                                                                  (int) ld_sc);
                }
                int* fallback = nullptr;
                if (b.candidates && pos0 + s0 >= 98304) {
                    fallback = b.select_fallback;
                    const int parts = std::max(4, std::min(48, (pos0 + s0 + st) / kpool / 2048));
                    kb_select_pivot<<<st, 256, 0, b.st>>>(b.score, (int)ld_sc, b.pp, s0, kpool, top_k / kpool - tail,
                                                         b.pivots, b.counts, fallback);
                    kb_select_filter<<<dim3(parts, st), 256, 0, b.st>>>(b.score, (int)ld_sc, b.pp, s0, kpool,
                                                                       b.pivots, b.counts, fallback, b.candidates);
                    kb_select_finish<<<st, 256, 0, b.st>>>(b.candidates, b.counts, fallback, b.pp, s0, t0, kpool,
                                                          top_k / kpool - tail, n_cap, cells);
                }
                kb_select_kpool<1024><<<st, 1024, 0, b.st>>>(b.score, (int) ld_sc, b.pp, s0, t0, kpool, top_k,
                                                             top_k / kpool - tail, n_cap, cells, fallback);
            }
            if (compact) {
                const int stride = (groups + 7) / 8 * 8, rows = rz - ra;
                auto pack = reinterpret_cast<uint16_t*>(b.xq2);
                if (rows)
                    kb_pack_groups<<<rows, 256, 0, b.st>>>(cells + (size_t)ra * n_cap, pack,
                                                          rows, n_cap, kpool, groups, stride);
                for (int r = 0; r < S; ++r)
                    b_copy(s, b, b.cl, peer_of(s.g, r), pack, (size_t)ra * stride * 2, (size_t)rows * stride * 2);
                kb_wait_k<<<1, 32, 0, b.st>>>(b.cl, s.g, S, b.step);
                if (direct) selected = reinterpret_cast<const uint16_t*>(b.cellsx[b.step & 1]);
                else kb_unpack_groups<<<tt, 256, 0, b.st>>>(reinterpret_cast<const uint16_t*>(b.cellsx[b.step & 1]),
                    cells, n_cap, kpool, groups, stride, b.pp, t0);
            } else if (split) {
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
            if (mqa && S==1) {
                kb_mqa_attn<32><<<tt,256,mqa_smem<32>(),b.st>>>(b.qabs,qt,(int)qp_ld,QH,nope,th,s.kv[il],
                        nullptr,cells,n_cap,b.pp,t0,win,1,0,sinks,scale,b.o,nullptr,CBW());
            } else if (mqa) {
                if ((mqa_opt & 8) && direct && n_cap <= 2048 && std::max(Hd,Hs)/S >= 16 && mbs.empty()) {
                    auto* qh = reinterpret_cast<__half*>(b.part);
                    auto* ph = reinterpret_cast<__half*>(b.o);
                    const int start = pos0 + t0;
                    kb_mqa_queries<32><<<dim3(nb(32*576),tt),256,0,b.st>>>(b.qabs,qt,(int)qp_ld,QH,nope,th,start,scale,qh);
                    kb_dsa_scores<<<dim3(16,tt),256,0,b.st>>>(qh,s.kv[il],cells,selected,groups,top_k,n_cap,start,s.g,b.qabs,dsa_half);
                    kb_dsa_probs<<<dim3(tt,4),256,0,b.st>>>(b.qabs,start,top_k,groups,n_cap,s.g,ph,b.part,CBW());
                    kb_dsa_values<<<dim3(8,tt),256,0,b.st>>>(ph,s.kv[il],cells,selected,groups,top_k,n_cap,start,s.g,b.part,CBW(),dsa_half);
                } else kb_mqa_attn<32><<<tt, 256, mqa_smem<32>(), b.st>>>(b.qabs, qt, (int) qp_ld, QH, nope, th, s.kv[il], nullptr,
                                                                 cells, n_cap, b.pp, t0, win, S, s.g, nullptr, scale,
                                                                 nullptr, b.part, CBW(), dsa_half, 0, mqa_opt & 1, direct ? top_k : 0, groups, selected);
                ++b.step;
                if (std::getenv("HY4F_CB32")) {
                    const size_t per = (size_t) tt * Hl * CBW();
                    push(s, b, b.part, per, 1, Hl, CBW(), b.cb, b.ctr + 4);
                    kb_combine<<<tt, 512, 0, b.st>>>(kvl, CBW(), s.g * Hl, Hl, sinks, b.cb, s.g, S, b.step, per, b.o);
                } else {   // fp16 slots: half the bytes (the local partial goes through the same path)
                    const int blocks = std::min(nb(tt * Hl * 256, 512), 160);
                    if (dma_mask & 4) {
                        auto pack = reinterpret_cast<uint8_t*>(b.xq);
                        kb_pack_cb16<<<dim3(tt,Hd), 256, 0, b.st>>>(b.part, tt, Hd, CBW(), Hl, pack);
                        const size_t len = (size_t)tt * Hl * CB_SLOT;
                        for (int r = 0; r < S; ++r) {
                            const int j = peer_of(s.g, r);
                            b_copy(s, b, b.cb, j, pack + (size_t)j * len, (size_t)s.g * len, len);
                        }
                    } else {
                        kb_push_cb16<<<blocks, 512, 0, b.st>>>(b.part, tt, Hd, CBW(), Hl, b.cb, s.g, b.step, b.ctr + 4, s.g);
                        for (int r = 1; r < S; ++r) {
                            const int j = peer_of(s.g, r);
                            kb_push_cb16<<<blocks, 512, 0, b.st>>>(b.part, tt, Hd, CBW(), Hl, b.cb, s.g, b.step, b.ctr + 4, j);
                            if (paired) kb_wait_one<<<1, 32, 0, b.st>>>(b.cb, s.g, j, b.step);
                        }
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
                kb_attn2<<<dim3(Hd,tt),256,0,b.st>>>(b.qabs,qt,(int)qp_ld,QH,nope,th,s.kv[il],nullptr,b.cells,
                    n_cap,b.pp,win,1,0,sinks,scale,b.o,nullptr,Hd,t0);
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
    const auto& ownership = expert_bounds(il);
    const int K = n_used, e0 = ownership[s.g], nl = ownership[s.g + 1] - ownership[s.g], fl = n_ffx / S;
    blin(s, b, P + "ffn_gate_inp.weight", b.x, E, b.lg, n_exp, T);
    router(s, b.lg, f32(s, P + "exp_probs_b.bias"), T, b.eids, b.ew, b.st);
    kb_moe_count<<<nl, 256, 0, b.st>>>(b.eids, T * K, e0, b.mcnt);
    kb_moe_scan<<<1, std::max(nl, 32), 0, b.st>>>(b.mcnt, nl, b.mb);
    // A short chunk fits all possible slots in the existing prefill arena.
    // The MMQ kernels still take each expert's exact boundaries from the GPU.
    const bool pad = std::getenv("HY4F_PF_SHORT_ROUTE") && T <= 64 && int64_t(T) * K <= b.cap && !offload;
    if (pad) HY_CK(cudaMemsetAsync(b.rows_tok, 0, sizeof(int) * T * K, b.st));
    kb_moe_place<<<nl, 256, 0, b.st>>>(b.eids, T * K, K, e0, nl, b.mb, b.rows_tok, b.row_of);
    if (!pad) {
        HY_CK(cudaMemcpyAsync(b.hmb, b.mb, sizeof(int) * (nl + 1), cudaMemcpyDeviceToHost, b.st));
        if (b.routing_ready) HY_CK(cudaEventRecord(b.routing_ready, b.st));
    }
    blin(s, b, P + "xmoe", b.x, E, b.xm, (int) xm_ld, T);
    if (!pad) {
        if (b.routing_ready) HY_CK(cudaEventSynchronize(b.routing_ready));
        else HY_CK(cudaStreamSynchronize(b.st));
    }
    const DW& wgu = s.w.at(P + "ffn_gu_exps");
    const DW& wd = s.w.at(P + "ffn_down_exps.weight");
    if (!s.pf_rows.empty() && !pad)
        for (int e = 0; e < nl; ++e) {
            const int c = b.hmb[e + 1] - b.hmb[e];
            s.pf_rows[(size_t) il * nl_max + e] += c;
            s.pf_touch[(size_t) il * nl_max + e] += c > 0;
        }
    // expert offload: a range lies wholly in VRAM or wholly in host memory, and a host range fits the staging buffer
    const OffLayer* ol = offload ? &s.off.at(il) : nullptr;
    const bool hyb = ol && ol->hybrid;   // tiers: [0, n_hot) base in VRAM, [n_hot, n_res) cold in VRAM, the rest cold on the host
    const int stage_n = ol ? (int) std::max<size_t>(1, off_stage / 2 / (hyb ? ol->cstride : ol->stride)) : 0;
    bool first = true;
    // the experts in ranges of at most cap rows (one range unless the routing piles onto this shard)
    for (int ea = 0; ea < nl;) {
        if (ol && ea >= ol->n_res && !first && b.hmb[ea + 1] == b.hmb[ea]) {   // an idle host expert is not fetched
            ++ea;
            continue;
        }
        const int elim = !ol ? nl : hyb && ea < ol->n_hot ? ol->n_hot : ea < ol->n_res ? ol->n_res : std::min(nl, ea + stage_n);
        int eb = pad ? nl : ea;
        int64_t mx = pad ? T : 0;
        while (eb < elim && (eb == ea || b.hmb[eb + 1] - b.hmb[ea] <= b.cap)) {
            mx = std::max<int64_t>(mx, b.hmb[eb + 1] - b.hmb[eb]);
            ++eb;
        }
        if (ol && ea >= ol->n_res)
            while (eb - 1 > ea && b.hmb[eb] == b.hmb[eb - 1]) --eb;
        const int ra = pad ? 0 : b.hmb[ea], rb = pad ? T * K : b.hmb[eb];
        const int64_t rows = rb - ra;
        if (rows > b.cap) throw std::runtime_error("hy4f: one expert has more rows than the buffers");
        const int* bounds = b.mb + ea;
        if (ra != 0) {
            kb_shift_bounds<<<nb(eb - ea + 1), 256, 0, b.st>>>(b.mb + ea, b.mb2, eb - ea);
            bounds = b.mb2;
        }
        if (rows > 0) {
            const bool cold_tier = hyb && ea >= ol->n_hot;   // (the range lies in one tier)
            mq::quantize(b.xm, b.rows_tok + ra, b.xq, cold_tier ? ol->cold_gu : wgu.type, latent, (int64_t) xm_ld, rows, b.st);
            mq::Product p{};
            const uint8_t* wg = static_cast<const uint8_t*>(wgu.d) + (size_t) ea * wgu.rb * (size_t) wgu.ne[1];
            const uint8_t* wdn = static_cast<const uint8_t*>(wd.d) + (size_t) ea * wd.rb * (size_t) wd.ne[1];
            int type_gu = wgu.type, type_dn = wd.type, staged = -1;
            size_t step_gu = expert_stride(wgu), step_dn = expert_stride(wd);
            if (ol) {
                const bool cold = cold_tier;
                const size_t st = cold ? ol->cstride : ol->stride;
                if (cold) type_gu = ol->cold_gu, type_dn = ol->cold_dn, step_gu = step_dn = st;
                if (ea < (hyb ? ol->n_hot : ol->n_res)) wg = ol->vram + (size_t) ea * st;
                else if (ea < ol->n_res) wg = ol->cvram + (size_t) (ea - ol->n_hot) * st;
                else {
                    const uint8_t* host = hyb ? ol->chost : ol->host;
                    const std::vector<int>& home = hyb ? ol->chome : ol->home;
                    staged = b.stage_turn++ & 1;
                    const cudaStream_t cs = b.copy_st ? b.copy_st : b.st;
                    uint8_t* stage = b.stage[staged];
                    if (b.copy_st) HY_CK(cudaStreamWaitEvent(cs, b.use_done[staged]));   // (the buffer's last launch is over)
                    for (int l = ea; l < eb;) {
                        if (b.hmb[l + 1] == b.hmb[l]) {   // (an idle expert's weights are never read)
                            ++l;
                            continue;
                        }
                        int m = l + 1;
                        while (m < eb && b.hmb[m + 1] != b.hmb[m] && home[m] == home[m - 1] + 1) ++m;
                        HY_CK(cudaMemcpyAsync(stage + (size_t) (l - ea) * st, host + (size_t) home[l] * st,
                                              (size_t) (m - l) * st, cudaMemcpyHostToDevice, cs));
                        l = m;
                    }
                    if (b.copy_st) {
                        HY_CK(cudaEventRecord(b.copy_done[staged], cs));
                        HY_CK(cudaStreamWaitEvent(b.st, b.copy_done[staged]));
                    }
                    wg = stage;
                }
                wdn = wg + (cold ? ol->cgu_bytes : ol->gu_bytes);
            }
            p.w = wg;
            p.type = type_gu;
            p.w_rows = 2 * n_ffx;
            p.w_cols = latent;
            p.expert_bytes = step_gu;
            p.n = eb - ea;
            p.xq = b.xq;
            p.bounds = bounds;
            p.ids = b.iota;
            p.total_rows = rows;
            p.max_rows = mx;
            p.dst = b.gu;
            p.ld_dst = n_ffx;   // (the fused product writes h = silu(g) * u, n_ff values per row)
            b.mmq->run_glu(p, clampv, b.st);
            if (pad) kb_zero_padded_rows<<<(unsigned)rows, 256, 0, b.st>>>(b.gu, n_ffx, b.mb + nl);
            mq::quantize(b.gu, nullptr, b.xq2, type_dn, n_ffx, n_ffx, rows, b.st);
            p.w = wdn;
            p.type = type_dn;
            p.w_rows = latent;
            p.w_cols = n_ffx;
            p.expert_bytes = step_dn;
            p.xq = b.xq2;
            p.dst = b.dn;
            p.ld_dst = latent;
            b.mmq->run(p, b.st);
            if (staged >= 0 && b.copy_st) HY_CK(cudaEventRecord(b.use_done[staged], b.st));
        }
        kb_moe_combine<<<dim3(nb(latent), T), 256, 0, b.st>>>(b.dn, latent, b.row_of, b.ew, K, b.moe, latent, ra, rb, !first);
        first = false;
        ea = eb;
    }
    blin(s, b, P + "ffn_routed_up.weight", b.moe, latent, b.mu, E, T);
    kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, b.st>>>(b.xm + latent, (int) xm_ld, fl, T, b.shh);
    blin(s, b, P + "ffn_down_shexp.weight", b.shh, fl, b.shd, E, T);
    if (fuse_pack_add && S > 1 && (dma_mask & 1) && E % 4 == 0) {
        b_ar_post(s, b, b.mu, P + "post_ffw_norm.weight", T, b.shd);
        return;
    }
    kb_add<<<nb(T * E), 256, 0, b.st>>>(b.mu, b.shd, b.ffo, (size_t) T * E);
    b_ar_post(s, b, b.ffo, P + "post_ffw_norm.weight", T);
}

// Attention and the dense lead consume only one token tile at a time.
// Routed experts still see all tokens, amortizing one host transfer over the full chunk.
void b_layer(Shard& s, BShard& b, int il, int start, int T) {
    const bool sliced = T > b.layer_tile;
    for (int t = 0; t < T; t += b.layer_tile) {
        const int n = std::min(T - t, b.layer_tile);
        BShard tile;
        BShard* active = &b;
        if (sliced) {
            tile = b;
            tile.dbounds = b.tile_bounds;
            kb_tile_bounds<<<1, 1, 0, b.st>>>(tile.dbounds, start + t, n);
            tile.X = tile.R = b.X + (size_t)t * hc * E;
            if (b.mtp) tile.mr = b.mr + (size_t)t * E;
            tile.emask = b.emask + t;
            tile.pp = tile.pp_attn = tile.dbounds + 2;
            tile.chunk_cur = b.chunk;
            tile.toff = tile.chunk_off = 0;
            tile.ring_T = n;
            active = &tile;
        }
        if (!b.mtp) {
            for (size_t e = 0; e < eg_layers.size(); ++e)
                if (eg_layers[e] == il) b_engram(s, *active, (int)e, il, start + t, n);
        }
        b_attention(s, *active, il, start + t, n);
        if (il < dense_lead) b_ffn(s, *active, il, n);
    }
    if (il >= dense_lead && !(b.mtp && mtp_kv_only)) b_ffn(s, b, il, T);
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
    for (int il = 0; il < L; ++il) b_layer(s, b, il, pos0, T);
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
    const int r0 = hrow0 ? 1 : 0;
    const std::string P = "blk." + std::to_string(L) + ".";
    const DW& emb = s.w.at("token_embd.weight");
    float* me = b.mr;    // consumed by enorm before eh_proj writes its output
    float* cat = b.gu;
    kd_embed_rows<<<dim3(nb(E), Tm), 256, 0, b.st>>>((const uint8_t*) emb.d, emb.rb, toks, E, me);
    kb_rms<<<Tm, 1024, 0, b.st>>>(me, E, f32(s, P + "nextn.enorm.weight"), E, eps, cat, 2 * E);
    const float* hnorm = f32(s, P + "nextn.hnorm.weight");
    if (hrow0) kb_rms<<<1, 1024, 0, b.st>>>(hrow0, E, hnorm, E, eps, cat + E, 2 * E);
    if (nx) kb_rms<<<nx, 1024, 0, b.st>>>(b.x, E, hnorm, E, eps, cat + (size_t)r0 * 2 * E + E, 2 * E);
    blin(s, b, P + "nextn.eh_proj.weight", cat, 2 * E, b.mr, E, Tm);
    b.mtp = true;
    b.pp = b.dbounds + 3;
    b_layer(s, b, L, mpos0, Tm);
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
