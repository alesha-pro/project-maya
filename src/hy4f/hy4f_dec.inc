// src/hy4f/hy4f_dec.inc - maya-hy4f: the multi-token decode step (T <= DT tokens: the speculative verify) and the MTP
// draft layer (blk.L, "nextn"), both captured as one CUDA graph per shard per T.  Included inside Model::Impl.
//
// Verify(T): the trunk over tokens toks[0..T) at positions pos .. pos+T-1 (writes their KV; a rejected tail is simply
// overwritten later: every cache is position-indexed, the SWA ring has RING_PAD spare rows), logits of every row and
// the rows' final hidden states h (hc head + output norm) kept on the device for the MTP.
// MTP(T): the draft layer over (h[t], next token[t]) at positions mpos .. mpos+T-1 (its own DSA KV), the draft token =
// argmax of the head over the last row.  llama.cpp hy-v4-flash.cpp graph_mtp: concat(RMS(emb) enorm, RMS(h) hnorm) ->
// eh_proj -> an ordinary sparse layer with a plain residual -> shared_head_norm -> the trunk's output head.

#ifdef HY4F_DT6
static constexpr int DT = 6;   // (the DFly build: the block = bonus + 5 drafts; -DHY4F_DT6)
#else
static constexpr int DT = 4;   // (6 costs ~40 MB per GPU and ~3% of prefill / T<=4 decode: the default build stays at 4)
#endif
static constexpr int NSP = 16;
bool short_graphs = std::getenv("HY4F_SHORT_GRAPH") != nullptr;
int moe_blocks = std::getenv("HY4F_MOE_BLOCKS") ? std::atoi(std::getenv("HY4F_MOE_BLOCKS")) : 328;
int dec_score_blocks = std::getenv("HY4F_DEC_SCORE_BLOCKS") ? std::atoi(std::getenv("HY4F_DEC_SCORE_BLOCKS")) : 256;   // split-K blocks per token of the decode attention
bool push_all = std::getenv("HY4F_PUSH_ALL") != nullptr;
bool push_seq = std::getenv("HY4F_PUSH_PER_PEER") == nullptr;
bool push_one = std::getenv("HY4F_PUSH_ONEFENCE") != nullptr;
bool ll_overlap = std::getenv("HY4F_LL_FUSED") != nullptr;
bool tail_old = std::getenv("HY4F_TAIL_OLD") != nullptr;
int ar_push_blocks = std::getenv("HY4F_PUSH_BLOCKS") ? std::atoi(std::getenv("HY4F_PUSH_BLOCKS")) : 1;
int ar_push_threads = std::getenv("HY4F_PUSH_THREADS") ? std::atoi(std::getenv("HY4F_PUSH_THREADS")) : 256;

struct IoT {   // a multi-token step's inputs (pinned host -> device at the head of each graph)
    int pos;
    int n;
    int toks[8];
    int last_pos;  // final row of a teacher-forced MTP update
};

// Hold both captured topologies on the host. Only the active one needs executable
// graph memory on the GPU. A switch can happen when decoding crosses top_k or
// when a later request reuses a shorter prefix; ordinary steps reuse the same exec.
struct DecodeGraph {
    cudaGraph_t shape[2] = {};
    cudaGraphExec_t exec = nullptr;
    bool short_active = false;

    void capture(cudaGraph_t graph, bool dense_dsa, bool both) {
        if (both) {
            shape[int(dense_dsa)] = graph;
            if (!dense_dsa) return;
        }
        HY_CK(cudaGraphInstantiate(&exec, graph, 0));
        short_active = dense_dsa;
        if (!both) HY_CK(cudaGraphDestroy(graph));
    }
    cudaGraphExec_t select(bool dense_dsa) {
        if (dense_dsa != short_active) {
            HY_CK(cudaGraphExecDestroy(exec));
            exec = nullptr;
            HY_CK(cudaGraphInstantiate(&exec, shape[int(dense_dsa)], 0));
            short_active = dense_dsa;
        }
        return exec;
    }
    void clear() {
        if (exec) cudaGraphExecDestroy(exec);
        for (auto graph : shape) if (graph) cudaGraphDestroy(graph);
    }
};

struct DShard {
    cudaStream_t attention_side = nullptr;
    cudaEvent_t attention_fork = nullptr, attention_join = nullptr;
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr, projected = nullptr, shared_done = nullptr;
    Q81* shq = nullptr;
    IoT *dio_v = nullptr, *dio_m = nullptr;
    float *X = nullptr, *x = nullptr, *h = nullptr, *post = nullptr, *hcp = nullptr, *hcs = nullptr, *ones = nullptr;
    float *xp = nullptr, *qr = nullptr, *qp = nullptr, *qabs = nullptr, *o = nullptr, *part = nullptr, *a = nullptr,
          *ao = nullptr, *ikn = nullptr, *ivg = nullptr, *iw = nullptr, *score = nullptr;
    int *cells = nullptr, *eids = nullptr, *eloc = nullptr;
    unsigned *sel_st = nullptr, *sel_hist = nullptr;
    int *mg_n = nullptr, *mg_exp = nullptr, *mg_off = nullptr, *mg_slot = nullptr;   // the step's expert groups
    int* sel_bc = nullptr;
    float *lg = nullptr, *ew = nullptr, *xm = nullptr, *ga16 = nullptr, *dn16 = nullptr, *moe = nullptr, *mu = nullptr,
          *shh = nullptr, *sh = nullptr, *ga = nullptr, *ffo = nullptr;
    int64_t* eg_ids = nullptr;
    float *eg_r = nullptr, *eg_kv = nullptr, *logits = nullptr, *argv = nullptr;
    int* argi = nullptr;
    float *me = nullptr, *cat = nullptr, *mr = nullptr, *mh = nullptr;
    float *spm = nullptr, *spl = nullptr, *spacc = nullptr;   // the split-K attention partials [DT][NSP][Hmax][..]
    Q81 *xq = nullptr, *xlq = nullptr, *q16 = nullptr;
    float *ysc = nullptr, *tpart = nullptr, *tppost = nullptr;   // the two-kernel all-reduce tail's scratch
    float* ar_buf[2] = {nullptr, nullptr};
    float* cb_buf[2] = {nullptr, nullptr};
    int *ar_flag = nullptr, *cb_flag = nullptr, *ar_ctr = nullptr, *cb_ctr = nullptr, *arrive = nullptr;
    PeersD ar{}, cb{};
    DecodeGraph gv[DT + 1], gm[DT + 1], gc;  // verify(T), mtp(T), chained MTP
};
std::vector<DShard> dsh;
struct DFShard {   // the DFly drafter's per-shard state (hy4f_dfly.inc)
    IoT *dio = nullptr, *cio = nullptr;
    int *snap = nullptr, *dc = nullptr, *argi = nullptr;   // snap: [draft layer][DT][n_cap] the verify rows' top-k cells
    float *cat = nullptr, *fz = nullptr, *z = nullptr, *ga = nullptr, *sh = nullptr, *pe = nullptr, *cc = nullptr,
          *cgu = nullptr, *ch = nullptr, *cd = nullptr, *corr = nullptr, *argv = nullptr;
    cudaGraphExec_t gb = nullptr, gk[8] = {};
};
std::vector<DFShard> dfs;
std::vector<int> df_slot;   // per target layer: the draft layer reading it, -1
bool has_mtp = false;
IoT *hio_v = nullptr, *hio_m = nullptr;
float* hlogT = nullptr;    // [DT][V] pinned
float* hargv = nullptr;    // [S][DT] pinned (per shard max)
int* hargi = nullptr;
float* hmargv = nullptr;   // [S] (MTP draft)
float* hmlog = nullptr;    // [V] pinned: the MTP head's logits row (for drafts sampled from it)
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
    d.sel_st = (unsigned*) ia((size_t) DT * 4);
    d.sel_hist = (unsigned*) ia((size_t) DT * 2048);
    d.sel_bc = ia((size_t) DT * SEL_NB * 2);
    d.mg_n = ia(4);
    d.mg_exp = ia(128);
    d.mg_off = ia(129);
    d.mg_slot = ia(128);
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
    d.spm = dalloc((size_t) DT * NSP * std::max(Hd, Hs / S));
    d.spl = dalloc((size_t) DT * NSP * std::max(Hd, Hs / S));
    d.spacc = dalloc((size_t) DT * NSP * std::max(Hd, Hs / S) * kvl);
    d.xq = qa((size_t) DT * std::max({2 * E, Hmax / S * vh, n_ff / S, ncol * eg_D, latent}));
    d.xlq = qa((size_t) DT * K * latent);
    d.q16 = qa((size_t) DT * K * n_ffx);
    if (std::getenv("HY4F_ATTN_OVERLAP")) {
        HY_CK(cudaStreamCreateWithFlags(&d.attention_side, cudaStreamNonBlocking));
        HY_CK(cudaEventCreateWithFlags(&d.attention_fork, cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&d.attention_join, cudaEventDisableTiming));
    }
    if (std::getenv("HY4F_FFN_OVERLAP")) {
        d.shq = qa((size_t)DT * (n_ffx / S));
        HY_CK(cudaStreamCreateWithFlags(&d.side, cudaStreamNonBlocking));
        HY_CK(cudaEventCreateWithFlags(&d.fork, cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&d.projected, cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&d.shared_done, cudaEventDisableTiming));
    }
    for (int p = 0; p < 2; ++p) {
        d.ar_buf[p] = dalloc((size_t) S * DT * E);
        d.cb_buf[p] = dalloc((size_t) S * DT * (Hd / S) * CBW());
    }
    d.ar_flag = ia(4);
    d.ysc = dalloc((size_t) DT * E);
    d.tpart = dalloc((size_t) DT * 16 * 32);
    d.tppost = dalloc((size_t) DT * 4);
    d.cb_flag = ia(4);
    d.ar_ctr = ia(1);
    d.cb_ctr = ia(1);
    d.arrive = ia(16);   // [0..4] all-reduce pushes / tail, [8..12] combine
}

void dec_mmvq(Shard& s, const std::string& n, const Q81* xq, float* y, int T, cudaStream_t stream = nullptr) {
    const DW& w = s.w.at(n);
    if (w.type != T_Q8_0) throw std::runtime_error("hy4f: " + n + " is not Q8_0");
    kn::native_q8_0_mmvq(w.d, xq, y, (int) w.ne[0], (int) (w.ne[1] * w.ne[2]), T, stream ? stream : s.st);
}
int rows_of(Shard& s, const std::string& n) {
    const DW& w = s.w.at(n);
    return (int) (w.ne[1] * w.ne[2]);
}

HcNext hc_next(Shard& s, const std::string& P, const std::string& k, const std::string& norm, int n_mix, float mag) {
    HcNext n;
    n.fn = f32(s, P + k + "_fn.weight");
    n.scale = f32(s, P + k + "_scale.weight");
    n.base = f32(s, P + k + "_base.weight");
    n.nw = f32(s, norm);
    n.n_mix = n_mix;
    n.mag = mag;
    return n;
}
bool fuse_ok() const { return E == 3072 && hc <= 4 && !std::getenv("HY4F_NO_FUSE"); }
void d_ar(Shard& s, DShard& d, const float* x1, const float* x2, const std::string& norm, int T, bool mtp,
          const HcNext* nx = nullptr) {
    float* X = mtp ? d.mr : d.X;
    const float* post = mtp ? d.ones : d.post;
    const int hcn = mtp ? 1 : hc;
    if (std::getenv("HY4F_AR1")) {
        kd_ar_post<<<1, 1024, sizeof(float) * T * E, s.st>>>(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, f32(s, norm), eps, X, post,
                                                              hcn);
        return;
    }
    const int tn = T * E;
    const bool overlap = ll_overlap && d.ar.bf == 2 && nx && !mtp && E == 3072 && T <= 4 && !tail_old;
    if (overlap) {
        dim3 grid(E / 256 + 1, T);
        int* arrive = d.arrive + 4;
        const float* w = f32(s, norm);
        auto args = std::make_tuple(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, arrive, w, d.X, d.post, hc,
                                    *nx, d.ysc, d.tpart, d.tppost);
        std::apply([&](auto&... val) {
            void* params[] = { (void*)&val... };
            HY_CK(cudaLaunchCooperativeKernel((const void*)kd_art_a<true>, grid, dim3(256), params, 0, s.st));
        }, args);
    } else if (push_seq)
        (push_one ? kd_push_seq1 : kd_push_seq)<<<ar_push_blocks, ar_push_threads, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, d.ar, s.g, S, d.ar_ctr,
                                                                          d.arrive);
    else if (push_all)
        kd_push_all<<<ar_push_blocks * (S - 1), 256, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, d.ar, s.g, S, ar_push_blocks,
                                                                d.ar_ctr, d.arrive);
    else
        for (int r = 1; r < S; ++r)
            kd_push1<<<ar_push_blocks, 256, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, d.ar, s.g, (s.g + r) % S, d.ar_ctr, d.arrive);
    if (nx && !mtp && E % 256 == 0 && E / 256 <= 16 && !tail_old) {   // (two wide kernels)
        if (!overlap) kd_art_a<false><<<dim3(E / 256, T), 256, 0, s.st>>>(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, d.arrive + 4, f32(s, norm), d.X, d.post,
                                                     hc, *nx, d.ysc, d.tpart, d.tppost);
        kd_art_b<<<dim3(E / 256, T), 256, 0, s.st>>>(E, f32(s, norm), eps, d.X, d.post, hc, *nx, hc_eps, d.ysc, d.tpart, d.tppost,
                                                     d.x, d.xq, overlap ? d.ar_ctr : nullptr);
    } else if (nx && !mtp)
        kd_ar_tail_hc<<<T, 1024, 0, s.st>>>(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, d.arrive + 4, f32(s, norm), eps, d.X, d.post,
                                            hc, *nx, hc_eps, d.x, d.xq);
    else
        kd_ar_tail<<<T, 1024, 0, s.st>>>(x1, x2, T, E, d.ar, s.g, S, d.ar_ctr, d.arrive + 4, norm.empty() ? nullptr : f32(s, norm),
                                          eps, X, post, hcn);
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
                                        1.1920929e-07f, nullptr);
}

void d_indexer_cache(Shard& s, DShard& d, const std::string& P, int il, const int* pp, int T,
                     int xpc, int oik, float th, cudaStream_t stream) {
    kb_rms<<<T, 128, 0, stream>>>(d.xp + oik, xpc, f32(s, P + "indexer.k_norm.weight"), IK, eps, d.ikn, IK);
    const DW& ivw = s.w.at(P + "ivg");
    kb_q8_gemv<<<dim3(nb(2 * IK * 32), T), 256, 0, stream>>>((const uint8_t*) ivw.d, d.ikn, IK, d.ivg, 2 * IK, IK, 2 * IK);
    kb_pool_groups<<<T / kpool + 1, 128, 0, stream>>>(d.ivg, s.ring[il], pp, T, f32(s, P + "indexer_compressor_ape.weight"),
                                                    f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK,
                                                    kpool, rope, eps, th);
    kb_ring_chunk<<<IRING, 256, 0, stream>>>(d.ivg, s.ring[il], T, pp, IK, kpool);
}

void d_attention(Shard& s, DShard& d, int il, const int* pp, int T, bool mtp, bool ready = false, bool dense_dsa = false, const int* last_pos = nullptr) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const bool sw = swa[il] != 0;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const int xpc = rows_of(s, P + "xproj"), qpc = !sw && dense_dsa ? Hd * QH : rows_of(s, P + "qproj");
    const int oq = 0, okv = q_lora, og = q_lora + KV_ROW, oik = og + Hl * vh;
    if (!ready) {
        if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "attn_norm.weight"), E, eps, d.x, E);
        else d_norm_in(s, d, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
        k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    }
    dec_mmvq(s, P + "xproj", d.xq, d.xp, T);
    // KV and the pooled index keys depend only on xproj. Compute them on
    // another graph branch during the query projection and absorption.
    const auto side = d.attention_side;
    const auto keys = side ? side : s.st;
    if (side) {
        HY_CK(cudaEventRecord(d.attention_fork, s.st));
        HY_CK(cudaStreamWaitEvent(side, d.attention_fork));
    }
    kb_kv_prep<<<T, 576, 0, keys>>>(d.xp + okv, xpc, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th, s.kv[il], pp,
                                    sw ? 3 : 2, S, s.g, win + RING_PAD);
    if (!sw) d_indexer_cache(s, d, P, il, pp, T, xpc, oik, th, keys);
    if (side) HY_CK(cudaEventRecord(d.attention_join, side));
    if (last_pos) {
        // All accepted rows contribute keys and indexer state. Only the final
        // row contributes a query, residual, FFN output and draft logits.
        d.mr += (size_t) (T - 1) * E;
        d.x += (size_t) (T - 1) * E;
        d.xp += (size_t) (T - 1) * xpc;
        if (dense_dsa) d.cells += (size_t) (T - 1) * n_cap;
        pp = last_pos;
        T = 1;
    }
    kb_rms<<<T, 1024, 0, s.st>>>(d.xp + oq, xpc, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, d.qr, q_lora);
    k_q8<<<nb(T * q_lora), 256, 0, s.st>>>(d.qr, T * q_lora, d.xq);
    if (!sw && dense_dsa) {
        const DW& q = s.w.at(P + "qproj");
        kn::native_q8_0_mmvq(q.d, d.xq, d.qp, (int) q.ne[0], qpc, T, s.st);
    } else dec_mmvq(s, P + "qproj", d.xq, d.qp, T);
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const int Ha = sw ? Hl : Hd;
    if (kb.type == T_Q8_0) {
        const uint8_t* kw = (const uint8_t*) kb.d;
        const int g = nb(Ha * kvl * 32);
        switch (T) {
        case 1: kd_absorb<1><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        case 2: kd_absorb<2><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        case 3: kd_absorb<3><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        case 4: kd_absorb<4><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        case 5: kd_absorb<5><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        default: kd_absorb<6><<<g, 256, 0, s.st>>>(kw, d.qp, qpc, QH, d.qabs, nope, kvl, Ha); break;
        }
    }
    else
        kb_absorb_bf16<<<dim3(nb(Ha * kvl), T), 256, 0, s.st>>>((const uint16_t*) kb.d, d.qp, qpc, QH, d.qabs, nope, kvl, Ha);
    if (side) HY_CK(cudaStreamWaitEvent(s.st, d.attention_join));
    const bool split = !std::getenv("HY4F_DEC_ATTN2");
    if (sw) {
        if (split && Hl == 16) {
            kd_mqa_dec<16><<<dim3(NSP, T), 256, mqd_smem<16>(), s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, 0, pp,
                                                                       win, 1, 0, scale, d.spm, d.spl, d.spacc);
            kd_mqa_merge<<<dim3(Hl, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hl,
                                                        f32(s, P + "attn_sinks.weight") + s.g * Hl, d.o, nullptr, 0);
        } else {
            kb_attn2<<<dim3(Hl, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, nullptr, 0, pp, win, 1,
                                                    0, f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, d.o, nullptr, Hl);
        }
    } else {
        float* iq = d.qp + (size_t) Hd * QH;
        if (!dense_dsa) kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, s.st>>>(iq, qpc, IH, IK, IK - rope, rope, pp, th);
        if (!dense_dsa) {
            kb_f32_gemv<<<dim3(nb(IH * 32), T), 256, 0, s.st>>>(f32(s, P + "indexer.proj.weight"), d.x, E, d.iw, IH, E, IH,
                                                                1.0f / std::sqrt((float) (IK * IH)));
            const int ld_sc = (int) (o.max_context / kpool + 1);
            if (IH == 16 && IK == 128 && !std::getenv("HY4F_OLD_SCORE"))   // (grid for the whole context: the graph is fixed)
                (T <= 4 ? kb_score_tc : kd_score_tc8)<<<dim3(dec_score_blocks, 1), 256, 0, s.st>>>(iq, qpc, s.pooled[il], d.iw, IH, kpool,
                                                                                             top_k, pp, 0, T, ld_sc - 1, d.score, ld_sc);
            else
                kb_score<<<dim3(nb(ld_sc * 32), T), 256, 0, s.st>>>(iq, qpc, s.pooled[il], d.iw, IH, IK, IH, kpool, top_k, pp, 0,
                                                                    d.score, ld_sc);
            if (std::getenv("HY4F_OLD_SELECT")) {
                kb_select_kpool<1024><<<T, 1024, 0, s.st>>>(d.score, ld_sc, pp, 0, 0, kpool, top_k, top_k / kpool - tail, n_cap,
                                                           d.cells);
            } else {   // (multi-block: 32 blocks per token)
                const int tp = top_k / kpool - tail;
                kd_sel_init<<<T, 256, 0, s.st>>>(pp, kpool, tp, d.sel_st, d.sel_hist);
                const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
                for (int p2 = 0; p2 < 3; ++p2) {
                    kd_sel_hist<<<dim3(SEL_NB, T), 256, 0, s.st>>>(d.score, ld_sc, pp, kpool, top_k, d.sel_st, d.sel_hist,
                                                                   shifts[p2], widths[p2]);
                    kd_sel_resolve<<<T, 1024, 0, s.st>>>(pp, top_k, d.sel_st, d.sel_hist, shifts[p2], widths[p2]);
                }
                kd_sel_count<<<dim3(SEL_NB, T), 256, 0, s.st>>>(d.score, ld_sc, pp, kpool, top_k, d.sel_st, d.sel_bc);
                kd_sel_write<<<dim3(SEL_NB, T), 256, 0, s.st>>>(d.score, ld_sc, pp, kpool, top_k, tp, n_cap, d.sel_st, d.sel_bc,
                                                                d.cells);
            }
        }
        if (!mtp && !dfs.empty() && df_slot[il] >= 0)   // (the DFly drafter reads this layer's top-k of every row)
            HY_CK(cudaMemcpyAsync(dfs[s.g].snap + (size_t) df_slot[il] * DT * n_cap, d.cells, sizeof(int) * T * n_cap,
                                  cudaMemcpyDeviceToDevice, s.st));
        if (S > 1) {
            if (split && Hd == 32) {
                (dense_dsa ? kd_mqa_dec<32, true> : kd_mqa_dec<32>)<<<dim3(NSP, T), 256, mqd_smem<32>(), s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], d.cells,
                                                                           n_cap, pp, win, S, s.g, scale, d.spm, d.spl,
                                                                           d.spacc);
                kd_mqa_merge<<<dim3(Hd, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hd, nullptr, nullptr, d.part, CBW());
            } else {
                kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp,
                                                        win, S, s.g, nullptr, scale, nullptr, d.part, Hd);
            }
            if (push_seq)
                (push_one ? kd_push_seq1 : kd_push_seq)<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd,
                                                                                    CBW(), Hl, d.cb, s.g, S, d.cb_ctr, d.arrive + 8);
            else if (push_all)
                kd_push_all<<<ar_push_blocks * (S - 1), 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl,
                                                                        d.cb, s.g, S, ar_push_blocks, d.cb_ctr, d.arrive + 8);
            else
                for (int r = 1; r < S; ++r)
                    kd_push1<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl, d.cb, s.g,
                                                               (s.g + r) % S, d.cb_ctr, d.arrive + 8);
            kd_cb_tail<<<T * Hl, 256, 0, s.st>>>(d.part, T, Hd, kvl, CBW(), s.g * Hl, Hl, f32(s, P + "attn_sinks.weight"),
                                                 d.cb, s.g, S, d.cb_ctr, d.arrive + 12, d.o);
        } else {
            kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp, win,
                                                    1, 0, f32(s, P + "attn_sinks.weight"), scale, d.o, nullptr, Hd);
        }
    }
    if (vb.type == T_Q8_0) {
        const uint8_t* vw = (const uint8_t*) vb.d;
        const int g = nb(Hl * vh * 32);
        switch (T) {
        case 1: kd_vb_gate<1><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        case 2: kd_vb_gate<2><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        case 3: kd_vb_gate<3><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        case 4: kd_vb_gate<4><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        case 5: kd_vb_gate<5><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        default: kd_vb_gate<6><<<g, 256, 0, s.st>>>(vw, d.o, d.xp + og, xpc, d.a, kvl, vh, Hl); break;
        }
    }
    else
        kb_vb_gate_bf16<<<dim3(nb(Hl * vh * 32), T), 256, 0, s.st>>>((const uint16_t*) vb.d, d.o, d.xp + og, xpc, d.a, kvl, vh,
                                                                     Hl);
    k_q8<<<nb(T * Hl * vh), 256, 0, s.st>>>(d.a, T * Hl * vh, d.xq);
    dec_mmvq(s, P + "attn_output.weight", d.xq, d.ao, T);
    if (!mtp && fuse_ok()) {   // (the FFN's input comes out of the all-reduce tail)
        const HcNext nx = hc_next(s, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag);
        d_ar(s, d, d.ao, nullptr, P + "post_attention_norm.weight", T, mtp, &nx);
    } else {
        d_ar(s, d, d.ao, nullptr, P + "post_attention_norm.weight", T, mtp);
    }
}

void d_ffn(Shard& s, DShard& d, int il, int T, bool mtp, bool ready = false, const HcNext* nx_after = nullptr) {
    const std::string P = "blk." + std::to_string(il) + ".";
    if (!ready) {
        if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, d.x, E);
        else d_norm_in(s, d, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag, T);
        k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
    }
    if (il < dense_lead) {
        const int fl = n_ff / S;
        dec_mmvq(s, P + "dense_gu", d.xq, d.ga, T);
        kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(d.ga, 2 * fl, fl, T, d.shh);
        k_q8<<<nb(T * fl), 256, 0, s.st>>>(d.shh, T * fl, d.xq);
        dec_mmvq(s, P + "ffn_down.weight", d.xq, d.ffo, T);
        d_ar(s, d, d.ffo, nullptr, P + "post_ffw_norm.weight", T, mtp, nx_after);
        return;
    }
    const int K = n_used, e0 = ebound[s.g], e1 = ebound[s.g + 1], fl = n_ffx / S;
    const int xmc = rows_of(s, P + "xmoe");
    const auto st = d.side ? d.side : s.st;
    if (d.side) {
        HY_CK(cudaEventRecord(d.fork, s.st));
        HY_CK(cudaStreamWaitEvent(d.side, d.fork));
        dec_mmvq(s, P + "xmoe", d.xq, d.xm, T, st);
        kd_q8_slots<<<dim3(nb(latent), T * K), 256, 0, st>>>(d.xm, xmc, latent, K, d.xlq);
        HY_CK(cudaEventRecord(d.projected, st));
        // Shared projection only needs xmoe. Continue on the side stream
        // while routed slots run on the capture's original stream.
        kd_ffn_q8<<<dim3(nb(fl), T), 256, 0, st>>>(nullptr, nullptr, nullptr, K, 0, d.xm + latent, xmc, fl, nullptr, d.shq);
        dec_mmvq(s, P + "ffn_down_shexp.weight", d.shq, d.sh, T, st);
        HY_CK(cudaEventRecord(d.shared_done, st));
    }
    kb_f32_gemv<<<dim3(nb(n_exp * 32), T), 256, 0, s.st>>>(f32(s, P + "ffn_gate_inp.weight"), d.x, E, d.lg, n_exp, E, n_exp,
                                                           1.0f);
    if (n_exp == 512 && K == 16 && !std::getenv("HY4F_OLD_ROUTER"))
        kd_router16<<<T, 512, 0, s.st>>>(d.lg, f32(s, P + "exp_probs_b.bias"), w_scale, e0, e1, d.eloc, d.ew);
    else {
        router(s, d.lg, f32(s, P + "exp_probs_b.bias"), T, d.eids, d.ew);
        kd_local_ids<<<1, 128, 0, s.st>>>(d.eids, d.eloc, T * K, e0, e1);
    }
    if (!d.side) {
        dec_mmvq(s, P + "xmoe", d.xq, d.xm, T);
        kd_q8_slots<<<dim3(nb(latent), T * K), 256, 0, s.st>>>(d.xm, xmc, latent, K, d.xlq);
    }
    const bool grouped = !std::getenv("HY4F_IDS_LOOP");
    const bool mp = T > 4;   // (more than 4 slots per expert possible: the multi-pass copy of the grouped kernel)
    if (grouped) kd_moe_groups<<<1, T * K <= 64 ? 64 : 128, 0, s.st>>>(d.eloc, T * K, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot);
    auto ids_mmvq = [&](const DW& w, const void* xq, float* y, int n_in, int n_out) {
        const size_t eb = w.rb * (size_t) w.ne[1];
        if (grouped) {
            if (w.type == T_Q3_K)
                (mp ? kn::native_q3_k_mmvq_grouped_mp : kn::native_q3_k_mmvq_grouped)(w.d, eb, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot,
                                                                                    xq, y, n_in, n_out, moe_blocks, s.st);
            else if (w.type == T_Q4_K)
                (mp ? kn::native_q4_k_mmvq_grouped_mp : kn::native_q4_k_mmvq_grouped)(w.d, eb, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot,
                                                                                    xq, y, n_in, n_out, moe_blocks, s.st);
            else throw std::runtime_error("hy4f: expert type " + std::to_string(w.type));
            return;
        }
        if (w.type == T_Q3_K) kn::native_q3_k_mmvq_ids_loop(w.d, eb, d.eloc, T * K, xq, false, y, n_in, n_out, s.st);
        else if (w.type == T_Q4_K) kn::native_q4_k_mmvq_ids_loop(w.d, eb, d.eloc, T * K, xq, false, y, n_in, n_out, s.st);
        else throw std::runtime_error("hy4f: expert type " + std::to_string(w.type));
    };
    if (d.side) HY_CK(cudaStreamWaitEvent(s.st, d.projected));
    ids_mmvq(s.w.at(P + "ffn_gu_exps"), d.xlq, d.ga16, latent, 2 * n_ffx);
    k_swiglu_gu_q8<<<nb(T * K * n_ffx), 256, 0, s.st>>>(d.ga16, n_ffx, T * K * n_ffx, clampv, d.q16);
    ids_mmvq(s.w.at(P + "ffn_down_exps.weight"), d.q16, d.dn16, n_ffx, latent);
    // d.q16 is dead after the expert down projection. Reuse it for the
    // shorter shared-expert activation so both inputs survive concurrently.
    const int fused_fl = d.side ? 0 : fl;
    kd_ffn_q8<<<dim3(nb(latent + fused_fl), T), 256, 0, s.st>>>(d.dn16, d.ew, d.eloc, K, latent,
                                                           d.xm + latent, xmc, fused_fl, d.xq, d.q16);
    dec_mmvq(s, P + "ffn_routed_up.weight", d.xq, d.mu, T);
    if (d.side) HY_CK(cudaStreamWaitEvent(s.st, d.shared_done));
    else dec_mmvq(s, P + "ffn_down_shexp.weight", d.q16, d.sh, T);
    d_ar(s, d, d.mu, d.sh, P + "post_ffw_norm.weight", T, mtp, nx_after);
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
    case 4: kd_bf16_gemv<4><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    case 5: kd_bf16_gemv<5><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    default: kd_bf16_gemv<6><<<nb(rows * 32), 256, 0, s.st>>>(W, x, E, rows, d.logits); break;
    }
    kd_argmax<<<T, 1024, 0, s.st>>>(d.logits, rows, s.g * rows, argv, argi);
}

void verify_ops(Shard& s, DShard& d, int T, bool dense_dsa = false) {
    HY_CK(cudaMemcpyAsync(d.dio_v, hio_v, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &d.dio_v->pos;
    if (dense_dsa) kd_dense_cells<<<dim3(nb(n_cap), T), 256, 0, s.st>>>(pp, n_cap, d.cells);
    const DW& emb = s.w.at("token_embd.weight");
    kb_embed<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, d.dio_v->toks, pp, d.X, d.X, E, hc, s.hist);
    auto has_engram = [&](int il) {
        for (int e : eg_layers)
            if (e == il) return true;
        return false;
    };
    const bool fuse = fuse_ok();
    bool ready = false;   // the next sublayer's x / xq / post already made by the previous all-reduce tail
    for (int il = 0; il < L; ++il) {
        for (size_t e = 0; e < eg_layers.size(); ++e)
            if (eg_layers[e] == il) d_engram(s, d, (int) e, il, pp, T);
        d_attention(s, d, il, pp, T, false, ready, dense_dsa);
        HcNext nx;
        const HcNext* nxp = nullptr;
        if (fuse && il + 1 < L && !has_engram(il + 1)) {
            const std::string Pn = "blk." + std::to_string(il + 1) + ".";
            nx = hc_next(s, Pn, "hc_attn", Pn + "attn_norm.weight", 2 * hc, hc_mag);
            nxp = &nx;
        } else if (fuse && il + 1 == L) {
            nx = hc_next(s, "", "output_hc", "output_norm.weight", hc, 0.0f);
            nxp = &nx;
        }
        d_ffn(s, d, il, T, false, fuse, nxp);
        ready = nxp != nullptr;
    }
    if (!ready) d_norm_in(s, d, "", "output_hc", "output_norm.weight", hc, 0.0f, T);
    HY_CK(cudaMemcpyAsync(d.h, d.x, sizeof(float) * T * E, cudaMemcpyDeviceToDevice, s.st));
    head_rows(s, d, d.x, T, d.argv, d.argi);
    const int rows = V / S;
    HY_CK(cudaMemcpy2DAsync(hlogT + (size_t) s.g * rows, sizeof(float) * V, d.logits, sizeof(float) * rows,
                            sizeof(float) * rows, T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargv + (size_t) s.g * DT, d.argv, sizeof(float) * T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargi + (size_t) s.g * DT, d.argi, sizeof(int) * T, cudaMemcpyDeviceToHost, s.st));
}

void mtp_ops(Shard& s, DShard& d, int T, bool chain = false, bool dense_dsa = false) {
    HY_CK(cudaMemcpyAsync(d.dio_m, hio_m, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &d.dio_m->pos;
    if (dense_dsa) kd_dense_cells<<<dim3(nb(n_cap), T), 256, 0, s.st>>>(pp, n_cap, d.cells);
    const std::string P = "blk." + std::to_string(L) + ".";
    const DW& emb = s.w.at("token_embd.weight");
    kd_embed_rows<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, d.dio_m->toks, E, d.me);
    kb_rms<<<T, 1024, 0, s.st>>>(d.me, E, f32(s, P + "nextn.enorm.weight"), E, eps, d.cat, 2 * E);
    // h: the trunk rows of the last verify, or (a chained draft) the MTP's own last state (llama.cpp t_h_nextn)
    kb_rms<<<T, 1024, 0, s.st>>>(chain ? d.mh : d.h, E, f32(s, P + "nextn.hnorm.weight"), E, eps, d.cat + E, 2 * E);
    k_q8<<<nb(T * 2 * E), 256, 0, s.st>>>(d.cat, T * 2 * E, d.xq);
    dec_mmvq(s, P + "nextn.eh_proj.weight", d.xq, d.mr, T);
    if (T > 1 && std::getenv("HY4F_MTP_LAST")) {
        DShard view = d;  // the final row will rebase the residual and projection pointers in this capture
        d_attention(s, view, L, pp, T, true, false, dense_dsa, &d.dio_m->last_pos);
        d_ffn(s, view, L, 1, true);
    } else {
        d_attention(s, d, L, pp, T, true, false, dense_dsa);
        d_ffn(s, d, L, T, true);
    }
    kb_rms<<<1, 1024, 0, s.st>>>(d.mr + (size_t) (T - 1) * E, E, f32(s, P + "nextn.shared_head_norm.weight"), E, eps, d.mh, E);
    head_rows(s, d, d.mh, 1, d.argv + DT - 1, d.argi + DT - 1);
    HY_CK(cudaMemcpyAsync(hmargv + s.g, d.argv + DT - 1, sizeof(float), cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hmargi + s.g, d.argi + DT - 1, sizeof(int), cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hmlog + (size_t) s.g * (V / S), d.logits, sizeof(float) * (V / S), cudaMemcpyDeviceToHost, s.st));
}

void dec_setup() {
    HY_CK(cudaHostAlloc((void**) &hio_v, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hio_m, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hlogT, sizeof(float) * DT * V, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargv, sizeof(float) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargi, sizeof(int) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargv, sizeof(float) * S, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargi, sizeof(int) * S, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmlog, sizeof(float) * V, cudaHostAllocPortable));
    std::memset(hio_v, 0, sizeof(IoT));
    std::memset(hio_m, 0, sizeof(IoT));
    dsh.resize(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) dec_alloc(sh[i], dsh[i]);
    PeersD ar{}, cb{};
    for (int j = 0; j < S; ++j) {
        ar.buf[j][0] = dsh[j].ar_buf[0];
        ar.buf[j][1] = dsh[j].ar_buf[1];
        ar.flag[j] = dsh[j].ar_flag;
        cb.buf[j][0] = dsh[j].cb_buf[0];
        cb.buf[j][1] = dsh[j].cb_buf[1];
        cb.flag[j] = dsh[j].cb_flag;
    }
    // (decode all-reduce partials in BF16 like the prefill's reduce-scatter: half the PCIe bytes; HY4F_AR_F32 = F32)
    ar.bf = push_seq && !push_all && !push_one && !std::getenv("HY4F_AR1") && !std::getenv("HY4F_AR_F32") ? 1 : 0;
    if (ar.bf && !std::getenv("HY4F_AR_FLAGS")) ar.bf = 2;   // (LL: 2 BF16 + the step per 8-byte store, no fence/flag)
    for (auto& d : dsh) d.ar = ar, d.cb = cb;
    if (df_nl) df_setup();
    kn::native_mmvq_set_multi_exact(false);   // (one weight read for the T columns; set before the captures)
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaFuncSetAttribute(kd_ar_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) (sizeof(float) * DT * E)));
        HY_CK(cudaFuncSetAttribute(kd_mqa_dec<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqd_smem<16>()));
        HY_CK(cudaFuncSetAttribute(kd_mqa_dec<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqd_smem<32>()));
        if (short_graphs)
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mqd_smem<32>()));
        if (std::getenv("HY4F_DEC_EAGER")) {   // (debug: one eager verify per T with a check after it)
            for (int T = 1; T <= DT; ++T) {
                hio_v->pos = 0;
                hio_v->n = T;
                launch_parallel([&](Shard& q) { verify_ops(q, dsh[q.g], T); HY_CK(cudaStreamSynchronize(q.st)); });
                HY_CK(cudaGetLastError());
            }
        }
        DShard& d = dsh[s.g];
        for (int mode = 0; mode <= int(short_graphs); ++mode) {
            const bool dense_dsa = mode != 0;
            for (int T = 1; T <= DT; ++T) {
                cudaGraph_t gr = nullptr;
                // (MMVQ with 5..8 columns: the upstream layout's 2-warp reduction is not bitwise equal to one column; the
                // verify rows must equal plain steps (spec == greedy), so T >= 5 takes the exact layout)
                kn::native_mmvq_set_multi_exact(T >= 5 && !std::getenv("HY4F_FAST_T56"));
                HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                verify_ops(s, d, T, dense_dsa);
                HY_CK(cudaStreamEndCapture(s.st, &gr));
                kn::native_mmvq_set_multi_exact(false);
                d.gv[T].capture(gr, dense_dsa, short_graphs);
                if (has_mtp) {
                    HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                    mtp_ops(s, d, T, false, dense_dsa);
                    HY_CK(cudaStreamEndCapture(s.st, &gr));
                    d.gm[T].capture(gr, dense_dsa, short_graphs);
                    if (T == 1) {
                        HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                        mtp_ops(s, d, 1, true, dense_dsa);
                        HY_CK(cudaStreamEndCapture(s.st, &gr));
                        d.gc.capture(gr, dense_dsa, short_graphs);
                    }
                }
            }
        }
        if (df_nl) df_capture(s);
        HY_CK(cudaDeviceSynchronize());
    }
}

void dec_launch(bool mtp, int T, bool chain = false) {
    const bool dense_dsa = short_graphs && (mtp ? hio_m : hio_v)->pos + T <= top_k;
    launch_parallel([&](Shard& s) {
        DShard& d = dsh[s.g];
        DecodeGraph& graph = chain ? d.gc : mtp ? d.gm[T] : d.gv[T];
        HY_CK(cudaGraphLaunch(graph.select(dense_dsa), s.st));
    });
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaStreamSynchronize(s.st));
    }
    HY_CK(cudaGetLastError());
}

#ifdef HY4F_VISION
bool encode_img(const vision::Image& img, std::vector<float>& rows, int& n_tokens, std::string& err) {
    if (!venc) throw std::runtime_error("hy4f: no vision encoder loaded");
    if (bsh.empty()) batch_setup(2048);
    // the shard whose batch arena is biggest relative to the need; the arena's contents are scratch between calls
    Shard& s = sh[0];
    BShard& b = bsh[0];
    const size_t need = venc->arena_bytes(img.w, img.h);
    if (need > b.arena_bytes)
        throw std::runtime_error("hy4f: image needs a " + std::to_string(need >> 20) + " MiB arena, the batch arena has " +
                                 std::to_string(b.arena_bytes >> 20) + " MiB (a larger prefill chunk or a smaller image)");
    HY_CK(cudaSetDevice(s.dev));
    vision::Arena ar;
    ar.dev_ptr = b.arena_base;
    ar.bytes = b.arena_bytes;
    ar.gpu = s.dev;
    ar.stream = (CUstream_st*) s.st;
    ar.cublas = b.blas;
    vision::Output out;
    if (!venc->encode(img, out, err, ar, nullptr)) return false;
    HY_BL(cublasSetStream(b.blas, s.st));
    rows.swap(out.embd);
    n_tokens = out.n_tokens;
    return true;
}
#endif
