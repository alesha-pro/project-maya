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
bool fuse_rms_q81 = std::getenv("HY4F_RMS_Q81") && std::atoi(std::getenv("HY4F_RMS_Q81")) != 0;
bool hc_fixed_post = std::getenv("HY4F_HC_FIXED_POST") && std::atoi(std::getenv("HY4F_HC_FIXED_POST")) == 1;
bool short_graphs = std::getenv("HY4F_SHORT_GRAPH") != nullptr;
bool q3_pair_layout = DT == 4 && std::getenv("HY4F_PACK_Q3") && !std::getenv("HY4F_IDS_LOOP")
                      && !std::getenv("HY4F_DEC_EAGER");
bool q3_packed = false, q3_layout_failed = false;
bool q3_eligible(const DW& w) const {
    return w.type == T_Q3_K && w.ne[0] == 1536 && (w.ne[1] == 1536 || w.ne[1] == 3072);
}
void change_q3_layout(bool packed) {
    if (q3_layout_failed) throw std::runtime_error("hy4f: expert layout conversion failed; reload the model");
    if (!q3_pair_layout || packed == q3_packed) return;
    q3_layout_failed = true;
    launch_parallel([&](Shard& s) {
        for (int il = dense_lead; il < L + int(has_mtp); ++il) {
            const auto p = "blk." + std::to_string(il) + ".";
            for (const char* suffix : {"ffn_gu_exps", "ffn_down_exps.weight"}) {
                const auto& w = s.w.at(p + suffix);
                if (!q3_eligible(w)) continue;
                const int pairs = int(w.rb * size_t(w.ne[1]) * size_t(w.ne[2]) / sizeof(packed_q3::Pair));
                const int grid = std::min(4096, (pairs + 7) / 8);
                if (packed) packed_q3::convert_pairs<true><<<grid, 256, 0, s.st>>>((unsigned*)w.d, pairs);
                else packed_q3::convert_pairs<false><<<grid, 256, 0, s.st>>>((unsigned*)w.d, pairs);
                HY_CK(cudaGetLastError());
            }
        }
        HY_CK(cudaStreamSynchronize(s.st));
    });
    q3_packed = packed;
    q3_layout_failed = false;
}
int moe_blocks = std::getenv("HY4F_MOE_BLOCKS") ? std::atoi(std::getenv("HY4F_MOE_BLOCKS")) : 328;
int dec_score_blocks = std::getenv("HY4F_DEC_SCORE_BLOCKS") ? std::atoi(std::getenv("HY4F_DEC_SCORE_BLOCKS")) : 256;   // split-K blocks per token of the decode attention
bool push_all = std::getenv("HY4F_PUSH_ALL") != nullptr;
bool push_seq = std::getenv("HY4F_PUSH_PER_PEER") == nullptr;
bool push_one = std::getenv("HY4F_PUSH_ONEFENCE") != nullptr;
bool ll_overlap = std::getenv("HY4F_LL_FUSED") != nullptr;
bool tail_old = std::getenv("HY4F_TAIL_OLD") != nullptr;
bool ar_ll4 = std::getenv("HY4F_AR_LL4") && std::atoi(std::getenv("HY4F_AR_LL4")) != 0;
bool ar_ll3 = std::getenv("HY4F_AR_LL3") && std::atoi(std::getenv("HY4F_AR_LL3")) != 0;
int ar_push_blocks = std::getenv("HY4F_PUSH_BLOCKS") ? std::atoi(std::getenv("HY4F_PUSH_BLOCKS")) : 1;
int ar_push_threads = std::getenv("HY4F_PUSH_THREADS") ? std::atoi(std::getenv("HY4F_PUSH_THREADS")) : 256;

struct IoT {   // a multi-token step's inputs (pinned host -> device at the head of each graph)
    int pos;
    int n;
    int toks[8];
    int last_pos;  // final row of a teacher-forced MTP update
};

// Hold the captured topologies on the host. Only the active one needs executable
// graph memory on the GPU. A switch can happen when decoding crosses top_k or
// when a later request reuses a shorter prefix; ordinary steps reuse the same exec.
struct DecodeGraph {
    cudaGraph_t shape[3] = {};
    cudaGraphExec_t exec = nullptr;
    int active = 0;

    void capture(cudaGraph_t graph, int mode, bool multiple) {
        if (multiple) {
            shape[mode] = graph;
            if (mode != 1) return;
        }
        HY_CK(cudaGraphInstantiate(&exec, graph, 0));
        active = mode;
        if (!multiple) HY_CK(cudaGraphDestroy(graph));
    }
    cudaGraphExec_t select(int mode) {
        if (mode != active) {
            HY_CK(cudaGraphExecDestroy(exec));
            exec = nullptr;
            HY_CK(cudaGraphInstantiate(&exec, shape[mode], 0));
            active = mode;
        }
        return exec;
    }
    void clear() {
        if (exec) cudaGraphExecDestroy(exec);
        for (auto graph : shape) if (graph) cudaGraphDestroy(graph);
    }
};

enum HeadMode { FullHead, GreedyHead, RankedHead, HeadModes };

struct DShard {
    cudaStream_t engram_side = nullptr;
    cudaEvent_t engram_fork = nullptr, engram_ready = nullptr;
    Q81* engram_q = nullptr;
    cudaStream_t hc_side = nullptr;
    cudaEvent_t hc_fork = nullptr, hc_join = nullptr;
    cudaStream_t attention_side = nullptr;
    cudaEvent_t attention_fork = nullptr, attention_join = nullptr;
    cudaStream_t side = nullptr;
    cudaEvent_t fork = nullptr, projected = nullptr, shared_done = nullptr;
    Q81* shq = nullptr;
    __half* packed_query = nullptr;
    packed_index::Cache* index_cache = nullptr;
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
    HeadValue *rank_tiles=nullptr, *rank_final=nullptr;
    float *me = nullptr, *cat = nullptr, *mr = nullptr, *mh = nullptr;
    float *spm = nullptr, *spl = nullptr, *spacc = nullptr;   // the split-K attention partials [DT][NSP][Hmax][..]
    Q81 *xq = nullptr, *xlq = nullptr, *q16 = nullptr;
    float *ysc = nullptr, *tpart = nullptr, *tppost = nullptr;   // the two-kernel all-reduce tail's scratch
    unsigned char* ar_generation = nullptr;
    float* ar_buf[2] = {nullptr, nullptr};
    float* cb_buf[2] = {nullptr, nullptr};
    int *ar_flag = nullptr, *cb_flag = nullptr, *ar_ctr = nullptr, *cb_ctr = nullptr, *arrive = nullptr;
    PeersD ar{}, cb{}, head_cb{};
    HeadCertificate cert;
    DecodeGraph verify_heads[DT+1][HeadModes];
    DecodeGraph mtp_heads[HeadModes];
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
int draft_head_type = 0;
bool cert_head = false, cert_target = false, cert_mtp = false, cert_ranked = false;
bool certified_mtp_active = false;
bool greedy_draft_head = false;
IoT *hio_v = nullptr, *hio_m = nullptr;
float* hlogT = nullptr;    // [DT][V] pinned
float* hargv = nullptr;    // [S][DT] pinned (per shard max)
int* hargi = nullptr;
float* hmargv = nullptr;   // [S] (MTP draft)
const bool compact_io = std::getenv("HY4F_COMPACT_LOGITS") != nullptr;
bool mtp_full_row = false;
HeadValue* rank_host=nullptr;  // [row][shard][head_candidates]
bool rank_active=false;
bool rank_from_mtp=false;
unsigned rank_pruned_rows=0;  // rows requiring BF16 replay for a full-logit fallback
bool rank_geometry() const {
    const int n=V/S;
    return n>=head_candidates && ((n+1023)/1024)*head_candidates<=2048;
}
bool separate_target_head() const { return cert_target || cert_ranked; }
bool separate_mtp_head() const { return draft_head_type || cert_mtp || cert_ranked; }

float* hmlog = nullptr;    // [V] pinned: the MTP head's logits row (for drafts sampled from it)
int* hmargi = nullptr;
int64_t m_next = 0;        // next MTP position

bool packed_index_dispatch = std::getenv("HY4F_DEC_PACKED_INDEX") != nullptr;
int packed_index_groups = -1, packed_index_next = 0;

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
    if (packed_index_dispatch && IH==16 && IK==128 && top_k==2048 && kpool==4) {
        HY_CK(cudaMalloc(&d.index_cache, L*sizeof(packed_index::Cache)));
        HY_CK(cudaMemset(d.index_cache,0,L*sizeof(packed_index::Cache)));
    }
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
    if (std::getenv("HY4F_PACK_QUERY"))
        HY_CK(cudaMalloc(&d.packed_query, sizeof(__half) * size_t(DT) * Ha * 576));
    d.dn16 = dalloc((size_t) DT * K * latent);
    d.moe = dalloc((size_t) DT * latent);
    d.mu = dalloc((size_t) DT * E);
    d.shh = dalloc((size_t) DT * std::max(n_ffx / S, n_ff / S));
    d.sh = dalloc((size_t) DT * E);
    d.ga = dalloc((size_t) DT * 2 * n_ff / S);
    d.ffo = dalloc((size_t) DT * E);
    HY_CK(cudaMalloc(&d.eg_ids, sizeof(int64_t) * DT * ncol));
    d.eg_r = dalloc((size_t) DT * ncol * eg_D);
    if (std::getenv("HY4F_ENGRAM_AHEAD") && !eg_layers.empty()) {
        HY_CK(cudaMalloc(&d.engram_q, sizeof(Q81) * DT * ((ncol * eg_D + 31) / 32)));
        HY_CK(cudaStreamCreateWithFlags(&d.engram_side, cudaStreamNonBlocking));
        HY_CK(cudaEventCreateWithFlags(&d.engram_fork, cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&d.engram_ready, cudaEventDisableTiming));
    }
    d.eg_kv = dalloc((size_t) DT * (hc + 1) * E);
    d.logits = dalloc((size_t) DT * (V / S));
    if (cert_head) {
        const int rows=V/S;
        d.cert.weight=static_cast<float2*>(s.w.at("cert.output.norms").d);
        HY_CK(cudaMalloc(&d.cert.input,DT*sizeof(float2)));
        HY_CK(cudaMalloc(&d.cert.lower,(size_t)DT*nb(rows,cert_tile)*sizeof(float2)));
        HY_CK(cudaMalloc(&d.cert.upper,(size_t)DT*rows*sizeof(float)));
        HY_CK(cudaMalloc(&d.cert.count,sizeof(int)));
        HY_CK(cudaMalloc(&d.cert.ids,(size_t)rows*sizeof(int)));
        if (cert_ranked) {
            HY_CK(cudaMalloc(&d.cert.rank_hist,DT*cert_rank_bins*sizeof(unsigned)));
            HY_CK(cudaMalloc(&d.cert.rank_cut,DT*sizeof(float)));
        }
    }
    if (rank_host) {
        const int blocks=(V/S+1023)/1024;
        HY_CK(cudaMalloc(&d.rank_tiles,sizeof(HeadValue)*DT*blocks*head_candidates));
        HY_CK(cudaMalloc(&d.rank_final,sizeof(HeadValue)*DT*head_candidates));
    }
    d.argv = dalloc(DT + int(compact_io));
    d.argi = ia(DT);
    d.me = dalloc((size_t) DT * E);
    d.cat = dalloc((size_t) DT * 2 * E);
    d.mr = dalloc((size_t) DT * E);
    d.mh = dalloc((size_t) DT * E);
    const int split_heads = replicate_prefix() ? 64 : std::max(Hd, Hs/S);
    d.spm = dalloc((size_t)DT * NSP * split_heads);
    d.spl = dalloc((size_t)DT * NSP * split_heads);
    d.spacc = dalloc((size_t)DT * NSP * split_heads * kvl);
    d.xq = qa((size_t) DT * std::max({2 * E, Hmax / S * vh, n_ff / S, ncol * eg_D, latent}));
    d.xlq = qa((size_t) DT * K * latent);
    d.q16 = qa((size_t) DT * K * n_ffx);
    if (std::getenv("HY4F_HC_OVERLAP") && !std::getenv("HY4F_AR1") && E == 3072 && !ll_overlap && !tail_old) {
        HY_CK(cudaStreamCreateWithFlags(&d.hc_side,cudaStreamNonBlocking));
        HY_CK(cudaEventCreateWithFlags(&d.hc_fork,cudaEventDisableTiming));
        HY_CK(cudaEventCreateWithFlags(&d.hc_join,cudaEventDisableTiming));
    }
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
        // Keep separate address ranges for T=3 and T=4. Their compact epochs
        // can wrap even when only one width has been running for a long time.
        const size_t extra = ar_ll4 && E % 4 == 0 ? (size_t)S * E * (3+4)
                           : ar_ll3 && E % 3 == 0 ? (size_t)S * (E/3) * (3+4) * 2 : 0;
        d.ar_buf[p] = dalloc((size_t) S * DT * E + extra);
        d.cb_buf[p] = dalloc((size_t) S * DT * (Hd / S) * CBW());
    }
    d.ar_flag = ia(4);
    d.ysc = dalloc((size_t) DT * E);
    d.tpart = dalloc((size_t) DT * 16 * 32);
    d.tppost = dalloc((size_t) DT * 4);
    d.cb_flag = ia(std::getenv("HY4F_HEAD_SEND") ? 4+DT*Hd : 4);
    d.ar_ctr = ia(ar_ll3 || ar_ll4 ? 3 : 1);
    if (ar_ll4 && E % 4 == 0) {
        const size_t bytes = (size_t)(E/4) * (3+4) * 2;
        d.ar_generation = reinterpret_cast<unsigned char*>(dalloc((bytes+sizeof(float)-1)/sizeof(float)));
    }
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
void d_hc_prepare(Shard& s, DShard& d, const HcNext& nx, int T) {
    HY_CK(cudaEventRecord(d.hc_fork,s.st));
    HY_CK(cudaStreamWaitEvent(d.hc_side,d.hc_fork));
    kd_art_prepare<<<dim3(E/256,T),256,0,d.hc_side>>>(d.X,E,hc,nx,d.tpart);
    HY_CK(cudaEventRecord(d.hc_join,d.hc_side));
}
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
    PeersD ar = d.ar;
    int* counter = d.ar_ctr;
    unsigned char* generations = d.ar_generation;
    const bool four = ar_ll4 && E % 4 == 0;
    const bool three = ar_ll3 && E % 3 == 0;
    if ((three || four) && ar.bf == 2 && (T == 3 || T == 4)) {
        ar.bf = four ? 4 : 3;
        counter += T - 2;
        const size_t words_per_token = four ? E : (E/3)*2;
        const size_t offset = (size_t)S * DT * E + (T == 4 ? (size_t)S * words_per_token * 3 : 0);
        if (four) {
            ar.exception_stride = S * T * (E/4);
            if (T == 4) generations += 2 * 3 * (E/4);
        }
        for (int j = 0; j < S; ++j)
            for (int parity = 0; parity < 2; ++parity)
                ar.buf[j][parity] += offset;
    }
    const bool scatter = std::getenv("HY4F_HC_SCATTER") != nullptr;
    const bool overlap = ll_overlap && ar.bf == 2 && nx && !mtp && E == 3072 && T <= 4 && !tail_old;
    if (overlap) {
        dim3 grid(E / 256 + 1, T);
        int* arrive = d.arrive + 4;
        const float* w = f32(s, norm);
        auto args = std::make_tuple(x1, x2, T, E, ar, s.g, S, counter, arrive, w, d.X, d.post, hc,
                                    *nx, d.ysc, d.tpart, d.tppost);
        std::apply([&](auto&... val) {
            void* params[] = { (void*)&val... };
            HY_CK(cudaLaunchCooperativeKernel((const void*)(scatter ? kd_art_a<true,false,true> : kd_art_a<true>), grid, dim3(256), params, 0, s.st));
        }, args);
    } else if (ar.bf == 4)
        kd_push_ll4<<<1, 256, 0, s.st>>>(x1, x2, tn, ar, s.g, S, counter, generations);
    else if (ar.bf == 3)
        kd_push_ll3<<<1, 256, 0, s.st>>>(x1, x2, tn, ar, s.g, S, counter);
    else if (push_seq)
        (push_one ? kd_push_seq1 : kd_push_seq)<<<ar_push_blocks, ar_push_threads, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, ar, s.g, S, counter,
                                                                          d.arrive);
    else if (push_all)
        kd_push_all<<<ar_push_blocks * (S - 1), 256, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, ar, s.g, S, ar_push_blocks,
                                                                counter, d.arrive);
    else
        for (int r = 1; r < S; ++r)
            kd_push1<<<ar_push_blocks, 256, 0, s.st>>>(x1, x2, tn, 0, T, 0, 0, 0, ar, s.g, (s.g + r) % S, counter, d.arrive);
    if (nx && !mtp && E % 256 == 0 && E / 256 <= 16 && !tail_old) {   // (two wide kernels)
        const bool columns = !overlap && !d.hc_side && ar.bf >= 2 && std::getenv("HY4F_HC_COLUMNS");
        const bool tested_packet_shape = (ar.bf == 2 && T == 2) || (ar.bf == 4 && (T == 3 || T == 4));
        if (hc_fixed_post && columns && tested_packet_shape && S == 4 && E == 3072 && hc == 4 &&
            (nx->n_mix == 4 || nx->n_mix == 8)) {
            const bool pre_only = nx->n_mix == 4;
            auto partial = ar.bf == 2 ? (pre_only ? kd_art_columns_fixed<2,4> : kd_art_columns_fixed<2,8>)
                                       : (pre_only ? kd_art_columns_fixed<4,4> : kd_art_columns_fixed<4,8>);
            auto finish = pre_only ? kd_art_post_fixed<4> : kd_art_post_fixed<8>;
            partial<<<dim3(E / 256, T, 4), 256, 0, s.st>>>(x1, x2, T, E, ar.buf[s.g][0], ar.buf[s.g][1],
                    ar.exception_stride, s.g, counter, d.arrive + 4, f32(s, norm), d.X, d.post, *nx,
                    d.ysc, d.tpart, d.tppost);
            finish<<<dim3(E / 256, T), 256, 0, s.st>>>(E, f32(s, norm), eps, d.X, d.post, *nx, hc_eps,
                    d.ysc, d.tpart, d.tppost, d.x, d.xq);
            return;
        }
        if (columns)
            kd_art_columns<<<dim3(E/256,T,4),256,0,s.st>>>(x1,x2,T,E,ar,s.g,S,counter,d.arrive+4,
                f32(s,norm),d.X,d.post,hc,*nx,d.ysc,d.tpart,d.tppost);
        else if (!overlap) (d.hc_side ? kd_art_a<false,true> : scatter ? kd_art_a<false,false,true> : kd_art_a<false,false>)<<<dim3(E / 256, T), 256, 0, s.st>>>(x1, x2, T, E, ar, s.g, S, counter, d.arrive + 4, f32(s, norm), d.X, d.post,
                                                     hc, *nx, d.ysc, d.tpart, d.tppost);
        if (d.hc_side) HY_CK(cudaStreamWaitEvent(s.st,d.hc_join));
        kd_art_b<<<dim3(E / 256, T), 256, 0, s.st>>>(E, f32(s, norm), eps, d.X, d.post, hc, *nx, hc_eps, d.ysc, d.tpart, d.tppost,
                                                     d.x, d.xq, overlap ? counter : nullptr);
    } else if (nx && !mtp)
        kd_ar_tail_hc<<<T, 1024, 0, s.st>>>(x1, x2, T, E, ar, s.g, S, counter, d.arrive + 4, f32(s, norm), eps, d.X, d.post,
                                            hc, *nx, hc_eps, d.x, d.xq);
    else
        kd_ar_tail<<<T, 1024, 0, s.st>>>(x1, x2, T, E, ar, s.g, S, counter, d.arrive + 4, norm.empty() ? nullptr : f32(s, norm),
                                          eps, X, post, hcn);
}
void d_norm_in(Shard& s, DShard& d, const std::string& P, const std::string& k, const std::string& norm, int n_mix,
               float mag, int T) {
    kb_hc_mix<<<dim3(HC_SPLIT, T), 256, 0, s.st>>>(d.X, f32(s, P + k + "_fn.weight"), hc * E, n_mix, d.hcp, d.hcs);
    kb_hc_apply_norm<<<T, 1024, 0, s.st>>>(d.X, d.hcp, d.hcs, f32(s, P + k + "_scale.weight"), f32(s, P + k + "_base.weight"),
                                           f32(s, norm), E, hc, n_mix, eps, hc_eps, mag, d.x, d.post);
}

void d_engram_prepare(Shard& s, DShard& d, int e, int il, const int* pp, int T, cudaStream_t stream, Q81* q) {
    const int ng = eg_ngram, nh = eg_heads, ncol = (ng - 1) * nh;
    const std::string P = "blk." + std::to_string(il) + ".";
    kb_engram_hash<<<T, 32, 0, stream>>>(s.hist, pp, s.eg_map, (int64_t) eg_map.size(), eg_pad, s.eg_mult + (size_t) e * ng,
                                       s.eg_vsz + (size_t) e * ncol, s.eg_off + (size_t) e * ncol, ng, nh, d.eg_ids);
    kb_engram_rows<<<dim3(ncol, T), 128, 0, stream>>>(eg_table, d.eg_ids, f32(s, P + "engram_lookup_norm.weight"), d.eg_r,
                                                    eg_D, ncol, 1.1920929e-07f);
    k_q8<<<nb(T * ncol * eg_D), 256, 0, stream>>>(d.eg_r, T * ncol * eg_D, q);
    dec_mmvq(s, P + "engram_kv.weight", q, d.eg_kv, T, stream);
}

// Engram hashes and projections depend only on the speculative token history.
// Wait for the preceding Engram mix before reusing its temporary storage.
void d_engram_fork(Shard& s, DShard& d, int e, int il, const int* pp, int T) {
    HY_CK(cudaEventRecord(d.engram_fork, s.st));
    HY_CK(cudaStreamWaitEvent(d.engram_side, d.engram_fork));
    d_engram_prepare(s, d, e, il, pp, T, d.engram_side, d.engram_q);
    HY_CK(cudaEventRecord(d.engram_ready, d.engram_side));
}

void d_engram_mix(Shard& s, DShard& d, int il, int T, bool split_hc) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const dim3 grid(T, split_hc ? hc : 1);
    (split_hc ? kb_engram_mix<true> : kb_engram_mix<false>)<<<grid, 1024, 0, s.st>>>(d.X, d.X, d.eg_kv, f32(s, P + "engram_kv_b.weight"), f32(s, P + "engram_gate.weight"),
                                        f32(s, P + "engram_hc_scale.weight"), f32(s, P + "engram_pre_gamma.weight"), E, hc,
                                        1.1920929e-07f, nullptr);
}

void d_indexer_cache(Shard& s, DShard& d, const std::string& P, int il, const int* pp, int T,
                     const float* input, int stride, float th, cudaStream_t stream) {
    kb_rms<<<T, 128, 0, stream>>>(input, stride, f32(s, P + "indexer.k_norm.weight"), IK, eps, d.ikn, IK);
    const DW& ivw = s.w.at(P + "ivg");
    kb_q8_gemv<<<dim3(nb(2 * IK * 32), T), 256, 0, stream>>>((const uint8_t*) ivw.d, d.ikn, IK, d.ivg, 2 * IK, IK, 2 * IK);
    kb_pool_groups<<<T / kpool + 1, 128, 0, stream>>>(d.ivg, s.ring[il], pp, T, f32(s, P + "indexer_compressor_ape.weight"),
                                                    f32(s, P + "indexer_compressor_norm.weight"), s.pooled[il], IK,
                                                    kpool, rope, eps, th);
    kb_ring_chunk<<<IRING, 256, 0, stream>>>(d.ivg, s.ring[il], T, pp, IK, kpool);
}

void d_indexer_weights(Shard& s, DShard& d, const std::string& P, int T, cudaStream_t st) {
    kb_f32_gemv<<<dim3(nb(IH * 32), T), 256, 0, st>>>(f32(s, P + "indexer.proj.weight"), d.x, E, d.iw, IH, E, IH,
                                                        1.0f / std::sqrt((float) (IK * IH)));
}

void d_indexer_rope(float* iq, int qpc, int T, const int* pp, float th, cudaStream_t st) {
    kb_rope<<<dim3(nb(IH * rope / 2), T), 256, 0, st>>>(iq, qpc, IH, IK, IK - rope, rope, pp, th);
}

void d_indexer_select(Shard& s, DShard& d, const std::string& P, int il, const int* pp,
                      int T, int qpc, int QH, float th, cudaStream_t st, float* local_iq = nullptr, bool query_ready = false) {
    float* iq = local_iq ? local_iq : d.qp + (size_t) Hd * QH;
    if (!query_ready) {
        d_indexer_rope(iq,qpc,T,pp,th,st);
        d_indexer_weights(s,d,P,T,st);
    }
    const int ld_sc = (int) (o.max_context / kpool + 1);
    if (d.index_cache && packed_index_dispatch && il<L && T>=2 && T<=4) {
        const auto kernel = T==2 ? packed_index::score<2,2> :
                            T==3 ? packed_index::score<3,4> : packed_index::score<4,4>;
        kernel<<<T==2?256:512,128,0,st>>>(iq,qpc,s.pooled[il],d.index_cache+il,d.iw,pp,ld_sc-1,d.score,ld_sc);
    } else if (IH == 16 && IK == 128 && T <= 4 && std::getenv("HY4F_DEC_SCORE_TILE")) {
        const auto kernel = T == 1 ? kd_score_tile<1> : T == 2 ? kd_score_tile<2> : kd_score_tile<4>;
        kernel<<<dec_score_blocks,256,0,st>>>(iq,qpc,s.pooled[il],d.iw,IH,kpool,top_k,pp,0,T,ld_sc-1,d.score,ld_sc);
    } else if (IH == 16 && IK == 128 && !std::getenv("HY4F_OLD_SCORE"))   // (grid for the whole context: the graph is fixed)
        (T <= 4 ? kb_score_tc : kd_score_tc8)<<<dim3(dec_score_blocks, 1), 256, 0, st>>>(iq, qpc, s.pooled[il], d.iw, IH, kpool,
                                                                                     top_k, pp, 0, T, ld_sc - 1, d.score, ld_sc);
    else
        kb_score<<<dim3(nb(ld_sc * 32), T), 256, 0, st>>>(iq, qpc, s.pooled[il], d.iw, IH, IK, IH, kpool, top_k, pp, 0,
                                                            d.score, ld_sc);
    if (std::getenv("HY4F_OLD_SELECT")) {
        kb_select_kpool<1024><<<T, 1024, 0, st>>>(d.score, ld_sc, pp, 0, 0, kpool, top_k, top_k / kpool - tail, n_cap,
                                                   d.cells);
    } else {   // (multi-block: 32 blocks per token)
        const int tp = top_k / kpool - tail;
        kd_sel_init<<<T, 256, 0, st>>>(pp, kpool, tp, d.sel_st, d.sel_hist);
        const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
        for (int p2 = 0; p2 < 3; ++p2) {
            kd_sel_hist<<<dim3(SEL_NB, T), 256, 0, st>>>(d.score, ld_sc, pp, kpool, top_k, d.sel_st, d.sel_hist,
                                                           shifts[p2], widths[p2]);
            kd_sel_resolve<<<T, 1024, 0, st>>>(pp, top_k, d.sel_st, d.sel_hist, shifts[p2], widths[p2]);
        }
        kd_sel_count<<<dim3(SEL_NB, T), 256, 0, st>>>(d.score, ld_sc, pp, kpool, top_k, d.sel_st, d.sel_bc);
        kd_sel_write<<<dim3(SEL_NB, T), 256, 0, st>>>(d.score, ld_sc, pp, kpool, top_k, tp, n_cap, d.sel_st, d.sel_bc,
                                                        d.cells);
    }
}

// Full context-parallel graph. Read each x projection row once on one of three
// branches. FFN slot storage is free from the preceding residual through the
// attention join; keep the normalized query there while all branches read xq.
void d_early_index_inputs(Shard& s, DShard& d, const std::string& P, int il, const int* pp,
                          int T, int xpc, int qpc, float th) {
    const DW& xw=s.w.at(P+"xproj"), &qw=s.w.at(P+"qproj");
    const auto* xrows=static_cast<const uint8_t*>(xw.d);
    const auto* qrows=static_cast<const uint8_t*>(qw.d);
    const auto queries=d.side, keys=d.attention_side;
    float* iq=d.qp+(size_t)T*qpc;
    const int niq=IH*IK;

    HY_CK(cudaEventRecord(d.attention_fork,s.st));
    HY_CK(cudaStreamWaitEvent(queries,d.attention_fork));
    HY_CK(cudaStreamWaitEvent(keys,d.attention_fork));

    kn::native_q8_0_mmvq(xrows,d.xq,d.qr,E,q_lora,T,queries);
    kd_rms_q81<1><<<T,1024,0,queries>>>(d.qr,q_lora,f32(s,P+"attn_q_a_norm.weight"),
                                          q_lora,eps,nullptr,d.xlq);
    HY_CK(cudaEventRecord(d.projected,queries));
    kn::native_q8_0_mmvq(qrows+(size_t)qpc*qw.rb,d.xlq,iq,q_lora,niq,T,queries);
    d_indexer_rope(iq,niq,T,pp,th,queries);
    HY_CK(cudaEventRecord(d.shared_done,queries));

    // ivg holds the compact raw index input until k_norm consumes it. The
    // following projection then replaces it with the pool values and gates.
    kn::native_q8_0_mmvq(xrows+(size_t)(q_lora+xpc)*xw.rb,d.xq,d.ivg,E,IK,T,keys);
    d_indexer_cache(s,d,P,il,pp,T,d.ivg,IK,th,keys);
    d_indexer_weights(s,d,P,T,keys);
    HY_CK(cudaStreamWaitEvent(keys,d.shared_done));
    d_indexer_select(s,d,P,il,pp,T,niq,0,th,keys,iq,true);
    HY_CK(cudaEventRecord(d.attention_join,keys));

    // The contiguous KV and gate rows keep their relative layout in xp.
    kn::native_q8_0_mmvq(xrows+(size_t)q_lora*xw.rb,d.xq,d.xp,E,xpc,T,s.st);
}

void d_attention(Shard& s, DShard& d, int il, const int* pp, int T, bool mtp, bool ready = false, bool dense_dsa = false, const int* last_pos = nullptr, bool sparse_dsa = false, bool early_inputs = false) {
    const std::string P = "blk." + std::to_string(il) + ".";
    const bool sw = swa[il] != 0;
    const int Hl = sw ? Hs / S : Hd / S, QH = nope + rope;
    const float th = sw ? theta_swa : theta;
    const DW& kb = s.w.at(P + "attn_k_b.weight");
    const bool local_queries = !sw && (dense_dsa || sparse_dsa) && !s.prefix_kv.empty() && kb.type == T_Q8_0 &&
                               std::getenv("HY4F_LOCAL_QUERY");
    const int query_first = local_queries ? s.g * Hl : 0;
    const int Ha = (sw || local_queries) ? Hl : Hd;
    const int xrows=rows_of(s,P+"xproj"), qrows=rows_of(s,P+"qproj");
    const bool early=early_inputs && !sw && !mtp && !dense_dsa && !sparse_dsa && !last_pos &&
        (T==2 || T==3) && d.side && d.attention_side && fuse_rms_q81 && q_lora==1024 &&
        S==4 && E==3072 && Hd==32 && IH==16 && IK==128 && n_used*latent>=q_lora &&
        s.w.at(P+"xproj").type==T_Q8_0 && s.w.at(P+"qproj").type==T_Q8_0 &&
        xrows==q_lora+KV_ROW+Hl*vh+IK && qrows==Hd*QH+IH*IK;
    const int xpc=early ? xrows-q_lora-IK : xrows;
    const int qpc=early || (!sw && (dense_dsa || (sparse_dsa && local_queries))) ? Ha*QH : qrows;
    const int oq=0, okv=early ? 0 : q_lora, og=okv+KV_ROW, oik=og+Hl*vh;
    if (!ready) {
        if (mtp && fuse_rms_q81 && E == 3072) {
            kd_rms_q81<3><<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "attn_norm.weight"), E, eps, d.x, d.xq);
        } else {
            if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "attn_norm.weight"), E, eps, d.x, E);
            else d_norm_in(s, d, P, "hc_attn", P + "attn_norm.weight", 2 * hc, hc_mag, T);
            k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
        }
    }
    if (d.hc_side && !mtp && fuse_ok())
        d_hc_prepare(s,d,hc_next(s,P,"hc_ffn",P+"ffn_norm.weight",2*hc,hc_mag),T);
    if (early) d_early_index_inputs(s,d,P,il,pp,T,xpc,qpc,th);
    else dec_mmvq(s, P + "xproj", d.xq, d.xp, T);
    // KV and the pooled index keys depend only on xproj. Compute them on
    // another graph branch during the query projection and absorption.
    const auto side = early ? nullptr : d.attention_side;
    const auto keys = side ? side : s.st;
    if (side) {
        HY_CK(cudaEventRecord(d.attention_fork, s.st));
        HY_CK(cudaStreamWaitEvent(side, d.attention_fork));
    }
    if (!sw && sparse_limit && (dense_dsa || sparse_dsa))
        kb_kv_prep<true,true><<<T,576,0,keys>>>(d.xp+okv,xpc,f32(s,P+"attn_kv_a_norm.weight"),kvl,
            rope,eps,th,s.kv[il],pp,2,S,s.g,0,s.prefix_kv.at(il),top_k,sparse_base,sparse_pitch,sparse_skip);
    else if (!sw && !s.prefix_kv.empty())
        kb_kv_prep<true><<<T,576,0,keys>>>(d.xp+okv,xpc,f32(s,P+"attn_kv_a_norm.weight"),kvl,
            rope,eps,th,s.kv[il],pp,2,S,s.g,0,s.prefix_kv.at(il),top_k);
    else
        kb_kv_prep<<<T, 576, 0, keys>>>(d.xp + okv, xpc, f32(s, P + "attn_kv_a_norm.weight"), kvl, rope, eps, th, s.kv[il], pp,
                                    sw ? 3 : 2, S, s.g, win + RING_PAD);
    if (!sw && !early) d_indexer_cache(s,d,P,il,pp,T,d.xp+oik,xpc,th,keys);
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
    if (early) {
        HY_CK(cudaStreamWaitEvent(s.st,d.projected));
    } else if (fuse_rms_q81 && q_lora == 1024) {
        kd_rms_q81<1><<<T, 1024, 0, s.st>>>(d.xp + oq, xpc, f32(s, P + "attn_q_a_norm.weight"),
                                            q_lora, eps, nullptr, d.xq);
    } else {
        kb_rms<<<T, 1024, 0, s.st>>>(d.xp + oq, xpc, f32(s, P + "attn_q_a_norm.weight"), q_lora, eps, d.qr, q_lora);
        k_q8<<<nb(T * q_lora), 256, 0, s.st>>>(d.qr, T * q_lora, d.xq);
    }
    float* local_iq = nullptr;
    int index_stride = qpc;
    if (early) {
        const DW& q=s.w.at(P+"qproj");
        kn::native_q8_0_mmvq(q.d,d.xlq,d.qp,q_lora,qpc,T,s.st);
    } else if (!sw && (dense_dsa || (sparse_dsa && local_queries))) {
        const DW& q = s.w.at(P + "qproj");
        const auto* weights=static_cast<const uint8_t*>(q.d)+(size_t)query_first*QH*q.rb;
        kn::native_q8_0_mmvq(weights, d.xq, d.qp, (int) q.ne[0], qpc, T, s.st);
        if (sparse_dsa) {
            index_stride = IH * IK;
            local_iq = d.qp + (size_t)T * qpc;
            const auto* index_weights = static_cast<const uint8_t*>(q.d) + (size_t)Hd * QH * q.rb;
            kn::native_q8_0_mmvq(index_weights,d.xq,local_iq,(int)q.ne[0],index_stride,T,s.st);
        }
    } else dec_mmvq(s, P + "qproj", d.xq, d.qp, T);
    const bool async_select = !sw && !dense_dsa && side && std::getenv("HY4F_INDEX_OVERLAP");
    if (async_select) {
        HY_CK(cudaEventRecord(d.attention_fork,s.st));
        HY_CK(cudaStreamWaitEvent(side,d.attention_fork));
        d_indexer_select(s,d,P,il,pp,T,index_stride,QH,th,side,local_iq);
        HY_CK(cudaEventRecord(d.attention_join,side));
    }
    const DW& vb = s.w.at(P + "attn_v_b.weight");
    const float scale = 1.0f / std::sqrt((float) QH);
    const bool split = !std::getenv("HY4F_DEC_ATTN2");
    const bool pack_query = std::getenv("HY4F_PACK_QUERY") && split && kb.type==T_Q8_0 && kvl==512 && rope==64
        && ((sw && Hl==16) || (!sw && S>1 && Hd==32)) && (size_t)Ha*576<=size_t(n_used)*2*n_ffx*2;
    __half* qh = pack_query ? d.packed_query : nullptr;
    if (kb.packed_projection) {
        ProjectionArgs args{(const uint8_t*)kb.d, d.qp, Ha, qpc, QH, d.qabs, nullptr, pp, th, scale};
        if (local_queries) {
            const size_t first_row=(size_t)query_first*kvl;
            args.w+=first_row*nope;
            args.scale_base=reinterpret_cast<const __half*>(static_cast<const uint8_t*>(kb.d)+(size_t)Hd*kvl*nope)
                            + first_row*(nope/32);
        }
        launch_projection<192>(T, args, s.st);
    } else if (kb.type == T_Q8_0) {
        const uint8_t* kw = (const uint8_t*)kb.d + (size_t)query_first*kvl*kb.rb;
        const int g = nb(Ha * kvl * 32);
        switch (T) {
        case 1: kd_absorb<1><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        case 2: kd_absorb<2><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        case 3: kd_absorb<3><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        case 4: kd_absorb<4><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        case 5: kd_absorb<5><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        default: kd_absorb<6><<<g,256,0,s.st>>>(kw,d.qp,qpc,QH,d.qabs,nope,kvl,Ha,qh,pp,th,scale); break;
        }
    }
    else
        kb_absorb_bf16<<<dim3(nb(Ha * kvl), T), 256, 0, s.st>>>((const uint16_t*) kb.d, d.qp, qpc, QH, d.qabs, nope, kvl, Ha);
    if (qh) {
        const dim3 grid(nb(Ha * 576), T);
        if (Ha == 32) kb_mqa_queries<32><<<grid,256,0,s.st>>>(d.qabs,d.qp,qpc,QH,nope,th,0,scale,qh,pp);
        else if (Ha == 16) kb_mqa_queries<16><<<grid,256,0,s.st>>>(d.qabs,d.qp,qpc,QH,nope,th,0,scale,qh,pp);
        else kb_mqa_queries<8><<<grid,256,0,s.st>>>(d.qabs,d.qp,qpc,QH,nope,th,0,scale,qh,pp);
    }
    if (side || early) HY_CK(cudaStreamWaitEvent(s.st, d.attention_join));
    const int value_groups = T<=2 && std::getenv("HY4F_VALUE_GROUPS") ? 4 : 1;
    if (sw) {
        if (split && Hl == 16) {
            (value_groups==4 ? kd_mqa_dec<16,false,4> : kd_mqa_dec<16>)<<<dim3(NSP*value_groups, T), 256, mqd_smem<16>(), s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, 0, pp,
                                                                       win, 1, 0, scale, d.spm, d.spl, d.spacc, 0, 0, qh, false, 0, 0, nullptr, 0);
            if (std::getenv("HY4F_MQA_FOLD"))
                kd_mqa_fold<false><<<dim3(Hl,T),256,0,s.st>>>(d.spm,d.spl,d.spacc,NSP,Hl,
                    f32(s,P+"attn_sinks.weight")+s.g*Hl,d.o,nullptr,0,d.cb,s.g,S,nullptr);
            else kd_mqa_merge<<<dim3(Hl, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hl,
                                                        f32(s, P + "attn_sinks.weight") + s.g * Hl, d.o, nullptr, 0);
        } else {
            kb_attn2<<<dim3(Hl, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, nullptr, 0, pp, win, 1,
                                                    0, f32(s, P + "attn_sinks.weight") + s.g * Hl, scale, d.o, nullptr, Hl);
        }
    } else {
        if (!dense_dsa && !async_select && !early)
            d_indexer_select(s,d,P,il,pp,T,index_stride,QH,th,s.st,local_iq);
        if (!mtp && !dfs.empty() && df_slot[il] >= 0)   // (the DFly drafter reads this layer's top-k of every row)
            HY_CK(cudaMemcpyAsync(dfs[s.g].snap + (size_t) df_slot[il] * DT * n_cap, d.cells, sizeof(int) * T * n_cap,
                                  cudaMemcpyDeviceToDevice, s.st));
        if (sparse_dsa || (dense_dsa && !s.prefix_kv.empty())) {
            if (sparse_dsa)
                kd_mqa_dec<16,false,1,true><<<dim3(NSP,T,4),256,mqd_smem<16>(),s.st>>>(
                    d.qabs,d.qp,qpc,QH,nope,th,s.kv[il],d.cells,n_cap,pp,win,S,s.g,
                    scale,d.spm,d.spl,d.spacc,top_k,top_k/kpool-tail,qh,local_queries,sparse_base,sparse_pitch,s.prefix_kv.at(il),sparse_skip);
            else
                kd_mqa_dec<16,true,1,true><<<dim3(NSP,T,4),256,mqd_smem<16>(),s.st>>>(
                    d.qabs,d.qp,qpc,QH,nope,th,s.prefix_kv.at(il),nullptr,0,pp,win,S,s.g,
                    scale,d.spm,d.spl,d.spacc,0,0,qh,local_queries);
            kd_replica_merge<<<dim3(Hl,T),256,0,s.st>>>(d.spm,d.spl,d.spacc,NSP,T,
                f32(s,P+"attn_sinks.weight")+s.g*Hl,d.o);
        } else if (S > 1) {
            const bool contiguous_cb = split && Hd == 32 && S == 4 && std::getenv("HY4F_CB_CONTIG");
            if (split && Hd == 32) {
                (value_groups==4 ? (dense_dsa ? kd_mqa_dec<32,true,4> : kd_mqa_dec<32,false,4>) : (dense_dsa ? kd_mqa_dec<32, true> : kd_mqa_dec<32>))<<<dim3(NSP*value_groups, T), 256, mqd_smem<32>(), s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], d.cells,
                                                                           n_cap, pp, win, S, s.g, scale, d.spm, d.spl,
                                                                           d.spacc, direct_dsa && S == 4 && kpool == 4 ? top_k : 0, top_k / kpool - tail, qh, false, 0, 0, nullptr, 0);
                if (contiguous_cb) {
                    kd_mqa_fold<false,true><<<dim3(Hd,T),256,0,s.st>>>(d.spm,d.spl,d.spacc,NSP,Hd,nullptr,nullptr,
                                                                      d.part,CBW(),d.cb,s.g,S,nullptr);
                } else if (std::getenv("HY4F_HEAD_SEND")) {
                    kd_mqa_fold<true><<<dim3(Hd,T),256,0,s.st>>>(d.spm,d.spl,d.spacc,NSP,Hd,nullptr,nullptr,
                                                               d.part,CBW(),d.head_cb,s.g,S,d.cb_ctr);
                } else if (std::getenv("HY4F_MQA_FOLD")) {
                    kd_mqa_fold<false><<<dim3(Hd,T),256,0,s.st>>>(d.spm,d.spl,d.spacc,NSP,Hd,nullptr,nullptr,
                                                                d.part,CBW(),d.cb,s.g,S,nullptr);
                } else kd_mqa_merge<<<dim3(Hd, T), 256, 0, s.st>>>(d.spm, d.spl, d.spacc, NSP, Hd, nullptr, nullptr, d.part, CBW());
            } else {
                kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp,
                                                        win, S, s.g, nullptr, scale, nullptr, d.part, Hd);
            }
            const bool sent = !contiguous_cb && split && Hd==32 && std::getenv("HY4F_HEAD_SEND");
            PeersD cb = sent ? d.head_cb : d.cb;
            if (contiguous_cb) {
                // The following residual all-reduce keeps the source and single
                // destination slot alive until all previous readers are done.
                const size_t span = (size_t)T * Hl * CBW();
                for (int r=1; r<S; ++r) {
                    const int peer = (s.g+r) % S;
                    HY_CK(cudaMemcpyAsync(cb.buf[peer][0] + s.g * span, d.part + peer * span,
                                          span * sizeof(float), cudaMemcpyDeviceToDevice, s.st));
                }
            } else if (sent) { /* kd_mqa_fold published each remote head. */ }
            else if (std::getenv("HY4F_CB_DMA") && push_seq) {
                // This exchange is followed by a blocking residual reduction
                // on every rank. A single receive slot therefore suffices.
                const size_t slice = (size_t) Hl * CBW();
                for (int j=0;j<S;++j) cb.buf[j][1]=cb.buf[j][0];
                for (int r=1;r<S;++r) {
                    const int j=(s.g+r)%S;
                    cudaMemcpy3DPeerParms copy{};
                    copy.srcDevice=s.dev;
                    copy.dstDevice=sh[j].dev;
                    copy.srcPtr=make_cudaPitchedPtr(d.part+(size_t)j*slice,
                        (size_t)Hd*CBW()*sizeof(float),(size_t)Hd*CBW()*sizeof(float),T);
                    copy.dstPtr=make_cudaPitchedPtr(cb.buf[j][0]+(size_t)s.g*T*slice,
                        slice*sizeof(float),slice*sizeof(float),T);
                    copy.extent=make_cudaExtent(slice*sizeof(float),T,1);
                    HY_CK(cudaMemcpy3DPeerAsync(&copy,s.st));
                    kd_cb_publish<<<1,32,0,s.st>>>(cb.flag[j],s.g,d.cb_ctr);
                }
            } else if (push_seq)
                (push_one ? kd_push_seq1 : kd_push_seq)<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd,
                                                                                    CBW(), Hl, d.cb, s.g, S, d.cb_ctr, d.arrive + 8);
            else if (push_all)
                kd_push_all<<<ar_push_blocks * (S - 1), 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl,
                                                                        d.cb, s.g, S, ar_push_blocks, d.cb_ctr, d.arrive + 8);
            else
                for (int r = 1; r < S; ++r)
                    kd_push1<<<ar_push_blocks, 256, 0, s.st>>>(d.part, nullptr, T * Hl * CBW(), 1, T, Hd, CBW(), Hl, d.cb, s.g,
                                                               (s.g + r) % S, d.cb_ctr, d.arrive + 8);
            if (contiguous_cb)
                kd_cb_dma_tail<<<T * Hl,256,0,s.st>>>(d.part,T,Hd,kvl,CBW(),f32(s,P+"attn_sinks.weight"),
                                                     cb,s.g,S,d.cb_ctr,d.arrive+12,d.o);
            else kd_cb_tail<<<T * Hl, 256, 0, s.st>>>(d.part, T, Hd, kvl, CBW(), s.g * Hl, Hl, f32(s, P + "attn_sinks.weight"),
                                                 cb, s.g, S, d.cb_ctr, d.arrive + 12, d.o, sent);
        } else {
            kb_attn2<<<dim3(Hd, T), 256, 0, s.st>>>(d.qabs, d.qp, qpc, QH, nope, th, s.kv[il], nullptr, d.cells, n_cap, pp, win,
                                                    1, 0, f32(s, P + "attn_sinks.weight"), scale, d.o, nullptr, Hd);
        }
    }
    if (vb.packed_projection) {
        ProjectionArgs args{(const uint8_t*)vb.d, d.o, Hl, Hl * kvl, kvl};
        args.gate = d.xp + og;
        args.gate_stride = xpc;
        args.qout = d.xq;
        launch_projection<512>(T, args, s.st);
    } else if (vb.type == T_Q8_0) {
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
    if (!vb.packed_projection) k_q8<<<nb(T * Hl * vh), 256, 0, s.st>>>(d.a, T * Hl * vh, d.xq);
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
        if (mtp && fuse_rms_q81 && E == 3072) {
            kd_rms_q81<3><<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, d.x, d.xq);
        } else {
            if (mtp) kb_rms<<<T, 1024, 0, s.st>>>(d.mr, E, f32(s, P + "ffn_norm.weight"), E, eps, d.x, E);
            else d_norm_in(s, d, P, "hc_ffn", P + "ffn_norm.weight", 2 * hc, hc_mag, T);
            k_q8<<<nb(T * E), 256, 0, s.st>>>(d.x, T * E, d.xq);
        }
    }
    if (d.hc_side && !mtp && nx_after) d_hc_prepare(s,d,*nx_after,T);
    if (il < dense_lead) {
        const int fl = n_ff / S;
        dec_mmvq(s, P + "dense_gu", d.xq, d.ga, T);
        if (fuse_rms_q81 && fl % 32 == 0) {
            kd_ffn_q8<<<dim3(nb(fl), T), 256, 0, s.st>>>(nullptr, nullptr, nullptr, 0, 0, d.ga,
                                                         2 * fl, fl, nullptr, d.xq);
        } else {
            kb_swiglu_rows<<<dim3(nb(fl), T), 256, 0, s.st>>>(d.ga, 2 * fl, fl, T, d.shh);
            k_q8<<<nb(T * fl), 256, 0, s.st>>>(d.shh, T * fl, d.xq);
        }
        dec_mmvq(s, P + "ffn_down.weight", d.xq, d.ffo, T);
        d_ar(s, d, d.ffo, nullptr, P + "post_ffw_norm.weight", T, mtp, nx_after);
        return;
    }
    const auto& ownership = expert_bounds(il);
    const int K = n_used, e0 = ownership[s.g], e1 = ownership[s.g + 1], fl = n_ffx / S;
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
    const DW& rw = s.w.at(P + "ffn_gate_inp.weight");
    if (rw.type == T_ROUTER_PACKED)
        kd_router_gemv<<<dim3(nb(n_exp, 4), T), 128, 0, s.st>>>((const uint16_t*) rw.d, d.x, d.lg, E, n_exp);
    else
        kb_f32_gemv<<<dim3(nb(n_exp * 32), T), 256, 0, s.st>>>(f32(s, P + "ffn_gate_inp.weight"), d.x, E, d.lg, n_exp,
                                                                E, n_exp, 1.0f);
    const bool grouped = !std::getenv("HY4F_IDS_LOOP");
    const bool use_router16 = n_exp == 512 && K == 16 && !std::getenv("HY4F_OLD_ROUTER");
    const bool fuse_groups = use_router16 && T == 1 && grouped && std::getenv("HY4F_GROUP_MATCH");
    if (use_router16)
        kd_router16<<<T, 512, 0, s.st>>>(d.lg, f32(s, P + "exp_probs_b.bias"), w_scale, e0, e1, d.eloc, d.ew,
                                         fuse_groups ? d.mg_n : nullptr, d.mg_exp, d.mg_off, d.mg_slot);
    else {
        router(s, d.lg, f32(s, P + "exp_probs_b.bias"), T, d.eids, d.ew);
        kd_local_ids<<<1, 128, 0, s.st>>>(d.eids, d.eloc, T * K, e0, e1);
    }
    if (!d.side) {
        dec_mmvq(s, P + "xmoe", d.xq, d.xm, T);
        kd_q8_slots<<<dim3(nb(latent), T * K), 256, 0, s.st>>>(d.xm, xmc, latent, K, d.xlq);
    }
    const bool mp = T > 4;   // (more than 4 slots per expert possible: the multi-pass copy of the grouped kernel)
    if (grouped && !fuse_groups) {
        const int slots = T * K, threads = slots <= 64 ? 64 : 128;
        if (!std::getenv("HY4F_GROUP_MATCH"))
            kd_moe_groups<<<1, threads, 0, s.st>>>(d.eloc, slots, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot);
        else if (T == 1 && K <= 32)
            kd_groups_one<<<1, 32, 0, s.st>>>(d.eloc, slots, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot);
        else
            kd_groups_match<<<1, threads, (e1 - e0) * (threads / 32) * sizeof(unsigned), s.st>>>(
                d.eloc, slots, e1 - e0, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot);
    }
    auto ids_mmvq = [&](const DW& w, const void* xq, float* y, int n_in, int n_out) {
        const size_t eb = w.rb * (size_t) w.ne[1];
        if (w.type == T_Q8_0) {
            if (grouped)
                expert_q8::launch<true>(w.d, eb, d.mg_exp, 0, d.mg_n, d.mg_off, d.mg_slot, xq, false, y,
                                        n_in, n_out, 2 * moe_blocks, s.st);
            else
                expert_q8::launch<false>(w.d, eb, d.eloc, T * K, nullptr, nullptr, nullptr, xq, false, y,
                                         n_in, n_out, 2 * moe_blocks, s.st);
            HY_CK(cudaGetLastError());
            return;
        }
        if (grouped) {
            if (q3_pair_layout && q3_eligible(w)) {
                packed_q3::forward(w.d, eb, d.mg_n, d.mg_exp, d.mg_off, d.mg_slot, xq, y, n_out, moe_blocks, s.st);
                return;
            }
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

template<int T> void rerank_head(Shard& s,DShard& d,const float* x,int rows,bool two,bool ranked) {
    if (ranked) kd_head_rank_list<T><<<nb(rows),256,0,s.st>>>(d.logits,rows,d.cert);
    else kd_head_list<T><<<nb(rows),256,0,s.st>>>(d.logits,rows,nb(rows,cert_tile),two,d.cert);
    const auto* bf=static_cast<const uint16_t*>(s.w.at("output.weight").d);
    kd_head_rerank<T><<<128,128,0,s.st>>>(bf,x,E,rows,d.logits,d.cert);
}
void certified_head_rows(Shard& s,DShard& d,const float* x,int T,bool two,bool ranked,float* argv,int* argi) {
    const int rows=V/S;
    const DW& q=s.w.at("cert.output.weight");
    k_q8<<<nb(T*E),256,0,s.st>>>(x,T*E,d.xq);
    kd_head_input_norm<<<T,256,0,s.st>>>(x,d.xq,E,d.cert.input);
    kn::native_q8_0_mmvq(q.d,d.xq,d.logits,E,rows,T,s.st);
    if (ranked) {
        HY_CK(cudaMemsetAsync(d.cert.rank_hist,0,(size_t)T*cert_rank_bins*sizeof(unsigned),s.st));
        kd_head_rank_bounds<<<dim3(nb(rows,cert_tile),T),cert_tile,0,s.st>>>(d.logits,rows,d.cert);
        kd_head_rank_cut<<<T,32,0,s.st>>>(d.cert,head_candidates);
    } else {
        kd_head_bounds<<<dim3(nb(rows,cert_tile),T),cert_tile,0,s.st>>>(d.logits,rows,d.cert);
    }
    switch(T) {
    case 1: rerank_head<1>(s,d,x,rows,two,ranked); break;
    case 2: rerank_head<2>(s,d,x,rows,two,ranked); break;
    case 3: rerank_head<3>(s,d,x,rows,two,ranked); break;
    case 4: rerank_head<4>(s,d,x,rows,two,ranked); break;
    case 5: rerank_head<5>(s,d,x,rows,two,ranked); break;
    default: rerank_head<6>(s,d,x,rows,two,ranked); break;
    }
    kd_argmax<<<T,1024,0,s.st>>>(d.logits,rows,s.g*rows,argv,argi);
}

void verify_head_ops(Shard& s,DShard& d,int T,HeadMode mode) {
    if (mode!=FullHead) certified_head_rows(s,d,d.x,T,false,mode==RankedHead,d.argv,d.argi);
    else head_rows(s, d, d.x, T, d.argv, d.argi);
    const int rows = V / S;
    if (!compact_io) HY_CK(cudaMemcpy2DAsync(hlogT + (size_t) s.g * rows, sizeof(float) * V, d.logits, sizeof(float) * rows,
                            sizeof(float) * rows, T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargv + (size_t) s.g * DT, d.argv, sizeof(float) * T, cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hargi + (size_t) s.g * DT, d.argi, sizeof(int) * T, cudaMemcpyDeviceToHost, s.st));
}

void verify_ops(Shard& s, DShard& d, int T, bool dense_dsa = false, bool sparse_dsa = false, bool early_inputs = false) {
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
        // A single set of scratch buffers suffices even for adjacent Engram layers.
        for (size_t e = 0; e < eg_layers.size(); ++e) {
            if (eg_layers[e] != il) continue;
            if (d.engram_side) {
                if (il == 0) d_engram_fork(s, d, (int)e, il, pp, T);
                HY_CK(cudaStreamWaitEvent(s.st, d.engram_ready));
            } else {
                d_engram_prepare(s, d, (int)e, il, pp, T, s.st, d.xq);
            }
            d_engram_mix(s, d, il, T, d.engram_side != nullptr);
        }
        if (d.engram_side) {
            for (size_t e = 0; e < eg_layers.size(); ++e)
                if (eg_layers[e] == il + 1) d_engram_fork(s, d, (int)e, il + 1, pp, T);
        }
        d_attention(s, d, il, pp, T, false, ready, dense_dsa, nullptr, sparse_dsa, early_inputs);
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
    if (!separate_target_head()) verify_head_ops(s,d,T,FullHead);
}

void mtp_head_ops(Shard& s, DShard& d, HeadMode mode) {
    if (mode==RankedHead || (mode==GreedyHead && cert_mtp)) {
        certified_head_rows(s,d,d.mh,1,true,mode==RankedHead,d.argv+DT-1,d.argi+DT-1);
    } else if (mode==GreedyHead) {
        const auto& w = s.w.at(cert_head && draft_head_type==T_Q8_0 ? "cert.output.weight" : "mtp.output.weight");
        const int rows = V / S;
        k_q8<<<nb(E), 256, 0, s.st>>>(d.mh, E, d.xq);
        if (w.type == T_Q8_0) kn::native_q8_0_mmvq(w.d, d.xq, d.logits, E, rows, 1, s.st);
        else kn::native_q4_k_mmvq(w.d, d.xq, d.logits, E, rows, 1, s.st);
        kd_argmax<<<1, 1024, 0, s.st>>>(d.logits, rows, s.g * rows, d.argv + DT - 1, d.argi + DT - 1);
    } else {
        head_rows(s, d, d.mh, 1, d.argv + DT - 1, d.argi + DT - 1);
    }
    if (compact_io)
        kd_runner_up<<<1, 1024, 0, s.st>>>(d.logits, V / S, s.g * (V / S), d.argi + DT - 1, d.argv + DT);
    const int count = compact_io ? 2 : 1;
    HY_CK(cudaMemcpyAsync(hmargv + count * s.g, d.argv + DT - 1, count * sizeof(float), cudaMemcpyDeviceToHost, s.st));
    HY_CK(cudaMemcpyAsync(hmargi + s.g, d.argi + DT - 1, sizeof(int), cudaMemcpyDeviceToHost, s.st));
    if (!compact_io) HY_CK(cudaMemcpyAsync(hmlog + (size_t) s.g * (V / S), d.logits, sizeof(float) * (V / S), cudaMemcpyDeviceToHost, s.st));
}

void mtp_ops(Shard& s, DShard& d, int T, bool chain = false, bool dense_dsa = false, bool sparse_dsa = false) {
    HY_CK(cudaMemcpyAsync(d.dio_m, hio_m, sizeof(IoT), cudaMemcpyHostToDevice, s.st));
    const int* pp = &d.dio_m->pos;
    if (dense_dsa) kd_dense_cells<<<dim3(nb(n_cap), T), 256, 0, s.st>>>(pp, n_cap, d.cells);
    const std::string P = "blk." + std::to_string(L) + ".";
    const DW& emb = s.w.at("token_embd.weight");
    if (fuse_rms_q81 && E == 3072) {
        kd_rms_q81<3,true><<<dim3(T, 2), 1024, 0, s.st>>>(chain ? d.mh : d.h, E,
            f32(s, P + "nextn.hnorm.weight"), E, eps, nullptr, d.xq, (const uint8_t*)emb.d,
            emb.rb, d.dio_m->toks, f32(s, P + "nextn.enorm.weight"));
    } else {
        kd_embed_rows<<<dim3(nb(E), T), 256, 0, s.st>>>((const uint8_t*) emb.d, emb.rb, d.dio_m->toks, E, d.me);
        kb_rms<<<T, 1024, 0, s.st>>>(d.me, E, f32(s, P + "nextn.enorm.weight"), E, eps, d.cat, 2 * E);
        // h: the trunk rows of the last verify, or (a chained draft) the MTP's own last state (llama.cpp t_h_nextn)
        kb_rms<<<T, 1024, 0, s.st>>>(chain ? d.mh : d.h, E, f32(s, P + "nextn.hnorm.weight"), E, eps, d.cat + E, 2 * E);
        k_q8<<<nb(T * 2 * E), 256, 0, s.st>>>(d.cat, T * 2 * E, d.xq);
    }
    dec_mmvq(s, P + "nextn.eh_proj.weight", d.xq, d.mr, T);
    if (T > 1 && std::getenv("HY4F_MTP_LAST")) {
        DShard view = d;  // the final row will rebase the residual and projection pointers in this capture
        d_attention(s, view, L, pp, T, true, false, dense_dsa, &d.dio_m->last_pos, sparse_dsa);
        d_ffn(s, view, L, 1, true);
    } else {
        d_attention(s, d, L, pp, T, true, false, dense_dsa, nullptr, sparse_dsa);
        d_ffn(s, d, L, T, true);
    }
    kb_rms<<<1, 1024, 0, s.st>>>(d.mr + (size_t) (T - 1) * E, E, f32(s, P + "nextn.shared_head_norm.weight"), E, eps, d.mh, E);
    if (!separate_mtp_head()) mtp_head_ops(s,d,FullHead);
}

void dec_setup() {
    setup_sparse_replica();
    HY_CK(cudaHostAlloc((void**) &hio_v, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hio_m, sizeof(IoT), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hlogT, sizeof(float) * DT * V, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargv, sizeof(float) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hargi, sizeof(int) * S * DT, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargv, sizeof(float) * S * (compact_io ? 2 : 1), cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmargi, sizeof(int) * S, cudaHostAllocPortable));
    HY_CK(cudaHostAlloc((void**) &hmlog, sizeof(float) * V, cudaHostAllocPortable));
    std::memset(hio_v, 0, sizeof(IoT));
    std::memset(hio_m, 0, sizeof(IoT));
    if (compact_io && rank_geometry() && (cert_ranked || std::getenv("HY4F_RANKED_SAMPLE")))
        HY_CK(cudaHostAlloc(&rank_host,sizeof(HeadValue)*DT*S*head_candidates,cudaHostAllocPortable));
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
    for (auto& d : dsh) {
        d.ar=ar; d.cb=cb; d.head_cb=cb;
        for (int j=0;j<S;++j) d.head_cb.flag[j]=dsh[j].cb_flag+4;
    }
    if (df_nl) df_setup();
    kn::native_mmvq_set_multi_exact(false);   // (one weight read for the T columns; set before the captures)
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        HY_CK(cudaFuncSetAttribute(kd_ar_post, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) (sizeof(float) * DT * E)));
        if (std::getenv("HY4F_VALUE_GROUPS")) {
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<16,false,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)mqd_smem<16>()));
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<32,false,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)mqd_smem<32>()));
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<32,true,4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)mqd_smem<32>()));
        }
        if (sparse_limit)
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<16,false,1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)mqd_smem<16>()));
        if (replicate_prefix()) {
            HY_CK(cudaFuncSetAttribute(kd_mqa_dec<16,true,1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)mqd_smem<16>()));
        }
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
        // The short shape is instantiated initially. Keep the two deep shapes
        // host-side and instantiate one on demand at a context boundary.
        for (int mode : {0, 2, 1}) {
            if ((mode == 2 && !sparse_limit) || (mode == 1 && !short_graphs)) continue;
            const bool dense_dsa = mode == 1, sparse_dsa = mode == 2;
            for (int T = 1; T <= DT; ++T) {
                cudaGraph_t gr = nullptr;
                // (MMVQ with 5..8 columns: the upstream layout's 2-warp reduction is not bitwise equal to one column; the
                // verify rows must equal plain steps (spec == greedy), so T >= 5 takes the exact layout)
                kn::native_mmvq_set_multi_exact(T >= 5 && !std::getenv("HY4F_FAST_T56"));
                HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                const bool early_inputs=mode==0 && short_graphs && sparse_limit &&
                    std::getenv("HY4F_INDEX_INPUT") && std::getenv("HY4F_INDEX_OVERLAP");
                verify_ops(s, d, T, dense_dsa, sparse_dsa, early_inputs);
                HY_CK(cudaStreamEndCapture(s.st, &gr));
                kn::native_mmvq_set_multi_exact(false);
                d.gv[T].capture(gr, mode, short_graphs);
                if (has_mtp) {
                    HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                    mtp_ops(s, d, T, false, dense_dsa, sparse_dsa);
                    HY_CK(cudaStreamEndCapture(s.st, &gr));
                    d.gm[T].capture(gr, mode, short_graphs);
                    if (T == 1) {
                        HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                        mtp_ops(s, d, 1, true, dense_dsa, sparse_dsa);
                        HY_CK(cudaStreamEndCapture(s.st, &gr));
                        d.gc.capture(gr, mode, short_graphs);
                    }
                }
            }
        }
        if (separate_target_head()) {
            for (int T=1;T<=DT;++T) for (HeadMode mode : {FullHead,GreedyHead,RankedHead}) {
                if ((mode==GreedyHead && !cert_target) || (mode==RankedHead && !cert_ranked)) continue;
                cudaGraph_t gr=nullptr;
                HY_CK(cudaStreamBeginCapture(s.st,cudaStreamCaptureModeThreadLocal));
                verify_head_ops(s,d,T,mode);
                HY_CK(cudaStreamEndCapture(s.st,&gr));
                d.verify_heads[T][mode].capture(gr,0,false);
            }
        }
        if (has_mtp && separate_mtp_head()) {
            for (HeadMode mode : {FullHead,GreedyHead,RankedHead}) {
                if ((mode==GreedyHead && !cert_mtp && !draft_head_type) || (mode==RankedHead && !cert_ranked)) continue;
                cudaGraph_t gr = nullptr;
                HY_CK(cudaStreamBeginCapture(s.st, cudaStreamCaptureModeThreadLocal));
                mtp_head_ops(s,d,mode);
                HY_CK(cudaStreamEndCapture(s.st, &gr));
                d.mtp_heads[mode].capture(gr, 0, false);
            }
        }
        if (df_nl) df_capture(s);
        HY_CK(cudaDeviceSynchronize());
    }
}

float mtp_max(int rank) const { return hmargv[rank * (compact_io ? 2 : 1)]; }

// Keep the canonical pooled rows usable by prefill, MTP and host checkpoints.
// Speculative rows and the open 8-group tile are always read from that storage.
void prepare_packed_index(int pos, int T) {
    const int stable = (pos/32)*8;
    if (stable < packed_index_groups) packed_index_groups = -1;
    if (T<2 || T>4 || pos+T<=top_k || (packed_index_groups>=0 && stable<packed_index_next)) return;
    int count=0;
    for (int il=0;il<L;++il) count+=!swa[il];
    if (!count) return;
    size_t bytes=bsh.empty()?0:bsh.front().arena_bytes;
    for (const auto& b : bsh) bytes=std::min(bytes,b.arena_bytes);
    const int fit=int(std::min<size_t>(bytes/(size_t(count)*256),o.max_context/4))/8*8;
    const int groups=std::min(fit,stable);
    launch_parallel([&](Shard& s) {
        int slab=0;
        for (int il=0;il<L;++il) if (!swa[il]) {
            auto* dest=groups?reinterpret_cast<uint4*>(bsh[s.g].arena_base)+size_t(slab)*groups*16:nullptr;
            packed_index::pack_keys<<<std::max(1,(groups+15)/16),256,0,s.st>>>(s.pooled.at(il),dest,groups,dsh[s.g].index_cache+il);
            ++slab;
        }
    });
    packed_index_groups=groups;
    packed_index_next=groups==fit?INT_MAX:groups+2048;
}

void dec_launch(bool mtp, int T, bool chain, bool full_logits) {
    if (!mtp && dsh.front().index_cache) prepare_packed_index(hio_v->pos,T);
    const bool ranked=cert_ranked && rank_active && !full_logits;
    const bool narrow=greedy_draft_head && !full_logits && !rank_active;
    rank_from_mtp=mtp;
    rank_pruned_rows=ranked ? (1u<<(mtp ? 1 : T))-1 : 0;
    if (mtp) {
        mtp_full_row = !compact_io || full_logits;
        certified_mtp_active = ranked || (cert_mtp && narrow);
    }
    const int end = (mtp ? hio_m : hio_v)->pos + T;
    const int mode = short_graphs && end <= top_k ? 1 : sparse_limit && end <= sparse_limit ? 2 : 0;
    const bool restore = mode == 2 && !sparse_valid;
    if (sparse_limit) {
        sparse_high = std::max(sparse_high,std::min(end,sparse_limit));
        if (end > sparse_limit) sparse_valid = false;
    }
    if (restore) {
        launch_parallel([&](Shard& s) {
            for (const auto& layer : s.prefix_kv) {
                const int il = layer.first;
                for (int owner = 0; owner < S; ++owner) if (owner != s.g) {
                    const int count = (sparse_high + S - 1 - owner)/S;
                    if (count <= sparse_skip) continue;
                    const int base = replica_row_base(owner,s.g,sparse_base,sparse_pitch);
                    HY_CK(cudaMemcpyPeerAsync(s.kv[il]+(size_t)(base+sparse_skip)*KV_RB,s.dev,
                                              sh[owner].kv[il]+(size_t)sparse_skip*KV_RB,sh[owner].dev,
                                              (size_t)(count-sparse_skip)*KV_RB,s.st));
                }
            }
            HY_CK(cudaStreamSynchronize(s.st));
        });
        sparse_valid = true;
    }
    change_q3_layout(true);
    launch_parallel([&](Shard& s) {
        DShard& d = dsh[s.g];
        DecodeGraph& graph = chain ? d.gc : mtp ? d.gm[T] : d.gv[T];
        HY_CK(cudaGraphLaunch(graph.select(mode), s.st));
        if (mtp && separate_mtp_head()) {
            const HeadMode mode=ranked ? RankedHead : narrow && (cert_mtp || draft_head_type) ? GreedyHead : FullHead;
            HY_CK(cudaGraphLaunch(d.mtp_heads[mode].exec,s.st));
        }
        if (!mtp && separate_target_head()) {
            const HeadMode mode=ranked ? RankedHead : narrow && cert_target ? GreedyHead : FullHead;
            HY_CK(cudaGraphLaunch(d.verify_heads[T][mode].exec,s.st));
        }
        if (rank_active && !full_logits) {
            const int n=V/S, blocks=(n+1023)/1024, nr=mtp ? 1 : T;
            kd_rank_head<false><<<dim3(blocks,nr),256,0,s.st>>>(d.logits,nullptr,d.rank_tiles,n,blocks,s.g*n);
            kd_rank_head<true><<<dim3(1,nr),256,0,s.st>>>(nullptr,d.rank_tiles,d.rank_final,n,blocks,s.g*n);
            HY_CK(cudaMemcpy2DAsync(rank_host+s.g*head_candidates,sizeof(HeadValue)*S*head_candidates,
                                    d.rank_final,sizeof(HeadValue)*head_candidates,sizeof(HeadValue)*head_candidates,
                                    nr,cudaMemcpyDeviceToHost,s.st));
        }
        if (compact_io && full_logits) {
            const int rows = V / S;
            if (mtp)
                HY_CK(cudaMemcpyAsync(hmlog + (size_t)s.g * rows, d.logits, sizeof(float) * rows, cudaMemcpyDeviceToHost, s.st));
            else
                HY_CK(cudaMemcpy2DAsync(hlogT + (size_t)s.g * rows, sizeof(float) * V, d.logits, sizeof(float) * rows,
                                        sizeof(float) * rows, T, cudaMemcpyDeviceToHost, s.st));
        }
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
    packed_index_groups = -1;
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
