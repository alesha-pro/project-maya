// Optional full-context replication for selected trunk DSA layers. Allocate only
// after the real batch arena exists. The CP graph stays available on low memory.
int full_stride() const { return (int)((o.max_context + S - 1) / S); }
size_t full_foreign_bytes() const { return (size_t)(S - 1) * full_stride() * KV_RB; }

void discard_full_storage() {
    for (auto& s : sh) {
        if (!s.full_store) continue;
        cudaSetDevice(s.dev);
        cudaFree(s.full_store);
        s.full_store = nullptr;
        std::fill(std::begin(s.full_mirror),std::end(s.full_mirror),nullptr);
    }
}

bool ensure_full_storage() {
    if (full_ready) return true;
    if (full_disabled || !full_requested || !sparse_limit || bsh.empty() || !mbs.empty()) return false;
    const int count = full_requested;
    const size_t bytes = full_foreign_bytes();
    constexpr size_t reserve = 16u << 20;
    int found = 0;
    for (int il = 0; il < L && found < count; ++il)
        if (!swa[il]) full_layers[found++] = il;
    if (found != count) {
        full_disabled = true;
        return false;
    }
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        size_t available = 0, total = 0;
        HY_CK(cudaMemGetInfo(&available,&total));
        if (available < count * bytes + reserve) {
            std::fprintf(stderr,"hy4f: full DSA skipped on device %d, %.1f MiB available\n",s.dev,available/1048576.0);
            full_disabled = true;
            return false;
        }
    }
    for (auto& s : sh) {
        HY_CK(cudaSetDevice(s.dev));
        const auto status = cudaMalloc(&s.full_store,count * bytes);
        if (status == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            discard_full_storage();
            full_disabled = true;
            return false;
        }
        HY_CK(status);
        for (int i = 0; i < count; ++i) s.full_mirror[i] = s.full_store + i * bytes;
    }
    full_count = count;
    full_ready = true;
    std::fprintf(stderr,"hy4f: %d dedicated full DSA mirrors, %.1f MiB/device\n",count,count*bytes/1048576.0);
    return true;
}

void truncate_full_mirrors(int pos) {
    for (int i=0;i<full_count;++i) full_prefix[i]=std::min(full_prefix[i],pos);
}

uint8_t* full_layer_mirror(Shard& s,int il) const {
    for (int i=0;i<full_count;++i)
        if (il==full_layers[i]) return s.full_mirror[i];
    return nullptr;
}

void restore_full_mirrors(int pos) {
    if (std::all_of(full_prefix,full_prefix+full_count,[&](int end){return end==pos;})) return;
    launch_parallel([&](Shard& s) {
        for (int i=0;i<full_count;++i) {
            const int il=full_layers[i];
            for (int owner=0;owner<S;++owner) if (owner!=s.g) {
                const int first=(full_prefix[i]+S-1-owner)/S;
                const int last=(pos+S-1-owner)/S;
                if (first>=last) continue;
                const int base=replica_row_base(owner,s.g,0,full_stride());
                HY_CK(cudaMemcpyPeerAsync(s.full_mirror[i]+(size_t)(base+first)*KV_RB,s.dev,
                         sh[owner].kv.at(il)+(size_t)first*KV_RB,sh[owner].dev,(size_t)(last-first)*KV_RB,s.st));
            }
        }
        HY_CK(cudaStreamSynchronize(s.st));
    });
    std::fill_n(full_prefix,full_count,pos);
}
