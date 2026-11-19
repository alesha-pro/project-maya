// hy4f_lane.inc - HY4F_CPU_LANE=1: the CPU lane of the expert offload (inside Model::Impl).
//
// A decode step's routed experts that live in host memory are computed by host threads instead of being read over
// PCIe by the device kernels.  Per shard: the device's k_split_lane writes the layer's host groups, their slots and
// the tokens' latent inputs into the host-mapped LaneMail and bumps `req`; this lane's boss thread sees it, runs the
// experts with Maya's native CPU kernels (the GGUF blocks as they are, ggml-cpu's arithmetic) over its crew, writes
// the slots' outputs into the mail and sets `ack`; the device's k_wait_lane copies them next to the VRAM experts'.
// The device computes its own experts meanwhile.  The prefill does not use the lane.
struct Lane {
    using clk = std::chrono::steady_clock;
    Impl* m = nullptr;
    int g = 0;
    LaneMail* mail = nullptr;
    std::thread boss;
    std::vector<std::thread> crew;
    std::atomic<bool> quit{false};
    struct Task {
        int group, r0, r1;
    };
    Task tasks[2048];
    std::atomic<uint64_t> ticket{0};   // (the phase's tasks << 32) | the next one to claim
    std::atomic<int> left{0};
    std::atomic<unsigned> epoch{0};
    int phase = 0;
    const strata::kernels::cpu::NativeFmt* fmt = nullptr;
    const uint8_t* rec[LANE_SLOTS];
    size_t half = 0;
    int nt[LANE_SLOTS];
    const void* act_p[LANE_SLOTS][LANE_T];
    float* ff_p[LANE_SLOTS][LANE_T];
    const void* hq_p[LANE_SLOTS][LANE_T];
    float* out_p[LANE_SLOTS][LANE_T];
    std::vector<uint8_t> act, hq;
    std::vector<float> ff;
    uint64_t n_req = 0, n_groups = 0, n_slots = 0;
    double busy = 0, t_prep = 0, t_gu = 0, t_qh = 0, t_dn = 0;   // (seconds: a request's stages)

    static void idle(clk::time_point last) {
        const auto dt = clk::now() - last;
        if (dt < std::chrono::milliseconds(2)) _mm_pause();
        else if (dt < std::chrono::seconds(1)) std::this_thread::sleep_for(std::chrono::microseconds(50));
        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // the rows about to be read: one prefetch a cache line, so the cold lines arrive with the core memory parallelism
    // instead of one demand miss after another (a 4 KiB page ends every hardware prefetch stream)
    static void fetch(const uint8_t* p, size_t n) {
        for (size_t i = 0; i < n; i += 64) _mm_prefetch(reinterpret_cast<const char*>(p + i), _MM_HINT_T1);
    }
    bool prefetch = true;
    bool parallel_quant = false;
    void run(const Task& t) {
        namespace kc = strata::kernels::cpu;
        const int gi = t.group;
        if (phase == 2) {
            for (int j = 0; j < nt[gi]; ++j)
                kc::native_quant_h(*fmt, ff_p[gi][j], const_cast<void*>(hq_p[gi][j]));
        } else if (phase == 0) {
            if (prefetch) {
                fetch(rec[gi] + (size_t) t.r0 * fmt->gu_row, (size_t) (t.r1 - t.r0) * fmt->gu_row);
                fetch(rec[gi] + half + (size_t) t.r0 * fmt->gu_row, (size_t) (t.r1 - t.r0) * fmt->gu_row);
            }
            kc::native_gu_rows_split(*fmt, rec[gi], rec[gi] + half, act_p[gi], nt[gi], ff_p[gi], t.r0, t.r1, m->clampv);
        } else {
            if (prefetch) fetch(rec[gi] + 2 * half + (size_t) t.r0 * fmt->d_row, (size_t) (t.r1 - t.r0) * fmt->d_row);
            kc::native_down_rows_split(*fmt, rec[gi] + 2 * half, hq_p[gi], nt[gi], out_p[gi], t.r0, t.r1);
        }
    }
    void drain() {
        for (;;) {
            const uint64_t t = ticket.fetch_add(1, std::memory_order_acq_rel);
            const uint32_t i = (uint32_t) t, n = (uint32_t) (t >> 32);
            if (i >= n) break;
            run(tasks[i]);
            left.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    void run_phase(int n) {
        left.store(n, std::memory_order_release);
        ticket.store((uint64_t) n << 32, std::memory_order_release);
        epoch.fetch_add(1, std::memory_order_acq_rel);
        drain();
        while (left.load(std::memory_order_acquire) > 0) _mm_pause();
    }
    void crew_loop() {
        unsigned seen = 0;
        auto last = clk::now();
        while (!quit.load(std::memory_order_relaxed)) {
            const unsigned e = epoch.load(std::memory_order_acquire);
            if (e == seen) {
                idle(last);
                continue;
            }
            seen = e;
            drain();
            last = clk::now();
        }
    }
    void serve() {
        namespace kc = strata::kernels::cpu;
        const int il = mail->layer, ng = mail->n_groups, K = m->n_used, n_ff = m->n_ffx, n_out = m->latent;
        if (ng <= 0 || ng > LANE_SLOTS || mail->n_slots > LANE_SLOTS) return;
        const OffLayer& ol = m->sh[g].off.at(il);
        fmt = &m->lane_fmt.at(il);
        const size_t st = ol.hybrid ? ol.cstride : ol.stride;
        half = (ol.hybrid ? ol.cgu_bytes : ol.gu_bytes) / 2;
        const uint8_t* host = ol.hybrid ? ol.chost : ol.host;
        const std::vector<int>& home = ol.hybrid ? ol.chome : ol.home;
        const auto c0 = clk::now();
        bool used[LANE_T] = {};
        for (int gi = 0; gi < ng; ++gi) {
            rec[gi] = host + (size_t) home[mail->gexp[gi]] * st;
            nt[gi] = mail->goff[gi + 1] - mail->goff[gi];
            for (int j = 0; j < nt[gi]; ++j) {
                const int si = mail->goff[gi] + j, t = mail->gslot[si] / K;
                used[t] = true;
                act_p[gi][j] = act.data() + (size_t) t * kc::kNativeActBytes;
                ff_p[gi][j] = ff.data() + (size_t) si * n_ff;
                hq_p[gi][j] = hq.data() + (size_t) si * kc::kNativeHBytes;
                out_p[gi][j] = mail->y[si];
            }
        }
        for (int t = 0; t < LANE_T; ++t)
            if (used[t]) kc::native_quant_act(*fmt, mail->x[t], act.data() + (size_t) t * kc::kNativeActBytes);
        int n = 0;
        phase = 0;
        for (int gi = 0; gi < ng; ++gi)
            for (int r = 0; r < n_ff; r += 128) tasks[n++] = {gi, r, std::min(r + 128, n_ff)};
        const auto c1 = clk::now();
        run_phase(n);
        const auto c2 = clk::now();
        if (parallel_quant && ng > 1) {
            phase = 2;
            for (int gi = 0; gi < ng; ++gi) tasks[gi] = {gi, 0, 0};
            run_phase(ng);
        } else {
            for (int gi = 0; gi < ng; ++gi)
                for (int j = 0; j < nt[gi]; ++j) kc::native_quant_h(*fmt, ff_p[gi][j], const_cast<void*>(hq_p[gi][j]));
        }
        const auto c3 = clk::now();
        n = 0;
        phase = 1;
        for (int gi = 0; gi < ng; ++gi)
            for (int r = 0; r < n_out; r += 256) tasks[n++] = {gi, r, std::min(r + 256, n_out)};
        run_phase(n);
        const auto c4 = clk::now();
        const auto sec = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
        t_prep += sec(c0, c1), t_gu += sec(c1, c2), t_qh += sec(c2, c3), t_dn += sec(c3, c4);
        n_groups += ng;
        n_slots += mail->n_slots;
    }
    void boss_loop() {
        unsigned last = 0;
        auto t_last = clk::now();
        while (!quit.load(std::memory_order_relaxed)) {
            const unsigned r = *(volatile const unsigned*) &mail->req;
            if (r == last) {
                idle(t_last);
                continue;
            }
            const auto a = clk::now();
            serve();
            last = r;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) &mail->ack = r;
            t_last = clk::now();
            busy += std::chrono::duration<double>(t_last - a).count();
            ++n_req;
        }
    }
};
std::vector<std::unique_ptr<Lane>> lanes;
std::map<int, strata::kernels::cpu::NativeFmt> lane_fmt;

void lane_start() {
    const char* e = std::getenv("HY4F_CPU_LANE");
    if (!e || !std::atoi(e) || !offload) return;
    namespace kc = strata::kernels::cpu;
    if (!kc::native_experts_available()) throw std::runtime_error("hy4f: HY4F_CPU_LANE needs a build with ggml-cpu");
    if (n_used * DT > LANE_SLOTS || DT > LANE_T || latent != LANE_DIM)
        throw std::runtime_error("hy4f: the CPU lane's mailbox does not fit this geometry");
    const int last = L + (has_mtp ? 1 : 0);
    for (int il = dense_lead; il < last; ++il) {
        const ExpertSrc src = expert_src(il, hybrid_layer(il));
        kc::NativeFmt f;
        std::string err;
        if (!kc::native_fmt(src.gu_type, src.dn_type, latent, n_ffx, f, err)) throw std::runtime_error("hy4f: CPU lane: " + err);
        if (f.act_bytes > kc::kNativeActBytes || f.h_bytes > kc::kNativeHBytes)
            throw std::runtime_error("hy4f: CPU lane: the activation does not fit");
        lane_fmt[il] = f;
    }
    const char* th = std::getenv("HY4F_CPU_THREADS");
    const int per = std::max(1, (th ? std::atoi(th) : 40) / S);
    for (int g = 0; g < S; ++g) {
        auto ln = std::make_unique<Lane>();
        ln->m = this;
        ln->g = g;
        HY_CK(cudaSetDevice(sh[g].dev));
        HY_CK(cudaHostAlloc((void**) &ln->mail, sizeof(LaneMail), cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(ln->mail, 0, sizeof(LaneMail));
        ln->act.resize((size_t) LANE_T * kc::kNativeActBytes);
        ln->ff.resize((size_t) LANE_SLOTS * n_ffx);
        ln->hq.resize((size_t) LANE_SLOTS * kc::kNativeHBytes);
        Lane* p = ln.get();
        if (const char* pf = std::getenv("HY4F_LANE_PREFETCH")) p->prefetch = std::atoi(pf) != 0;
        p->parallel_quant = S == 1 && std::getenv("HY4F_SINGLE_PAR_QUANT");
        p->boss = std::thread([p] { p->boss_loop(); });
        for (int i = 1; i < per; ++i) p->crew.emplace_back([p] { p->crew_loop(); });
        // HY4F_LANE_PIN=1: one online CPU a thread, from CPU 1 up (the first hardware thread of each core comes first
        // in the kernel's numbering here), shard after shard
        if (const char* pin = std::getenv("HY4F_LANE_PIN"); pin && std::atoi(pin)) {
            cpu_set_t online;
            CPU_ZERO(&online);
            sched_getaffinity(0, sizeof(online), &online);
            std::vector<int> cpus;
            for (int c = 1; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &online)) cpus.push_back(c);
            auto bind = [&](std::thread& th, int k) {
                if (cpus.empty()) return;
                cpu_set_t one;
                CPU_ZERO(&one);
                CPU_SET(cpus[(size_t) k % cpus.size()], &one);
                pthread_setaffinity_np(th.native_handle(), sizeof(one), &one);
            };
            bind(p->boss, g * per);
            for (int i = 1; i < per; ++i) bind(p->crew[(size_t) i - 1], g * per + i);
        }
        lanes.push_back(std::move(ln));
    }
    if (o.verbose) std::fprintf(stderr, "hy4f: CPU lane: %d threads a shard\n", per);
}
void lane_stop() {
    for (auto& ln : lanes) ln->quit.store(true);
    for (auto& ln : lanes) {
        if (ln->boss.joinable()) ln->boss.join();
        for (auto& t : ln->crew) t.join();
    }
}
