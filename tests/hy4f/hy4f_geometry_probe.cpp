// tests/hy4f/hy4f_geometry_probe.cpp - the Hy4 flash port: read a hy_v4_flash GGUF (all shards) through Maya's GLM
// geometry route, check every tensor shape, and ask the forward gate.  No GPU, no weight read.
// usage: hy4f_geometry_probe <shard-00001.gguf> [more shards...]
#include "strata/core/glm_arch.hpp"
#include "strata/core/glm_model.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shard-00001.gguf> [more shards...]\n", argv[0]);
        return 2;
    }
    std::vector<std::unique_ptr<strata::GgufFile>> shards;
    try {
        for (int i = 1; i < argc; ++i) shards.emplace_back(new strata::GgufFile(argv[i]));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open: %s\n", e.what());
        return 1;
    }
    using strata::core::Glm5Geometry;
    Glm5Geometry geo;
    std::string err;
    if (!Glm5Geometry::from_gguf(*shards[0], geo, err)) {
        std::fprintf(stderr, "FAIL geometry: %s\n", err.c_str());
        return 1;
    }
    std::printf("arch: %s\n", strata::core::glm_arch_name(geo.arch));
    if (geo.arch != strata::core::GlmArch::Hy4Flash) {
        std::fprintf(stderr, "FAIL: not routed to hy-v4-flash\n");
        return 1;
    }
    std::printf("geometry: %s\n", strata::core::hy4f_geometry_summary(geo).c_str());
    std::string swa;
    for (int il = 0; il < geo.n_layers; ++il) swa += geo.is_swa(il) ? 'S' : 'D';
    std::printf("layers:   %s (D = DSA + indexer, S = sliding window)\n", swa.c_str());
    std::string eg;
    for (int l : geo.engram_layers) eg += (eg.empty() ? "" : ",") + std::to_string(l);
    std::printf("engram:   layers %s, ngram %d, pad %d, token_map %lld\n", eg.c_str(), geo.engram_ngram, geo.engram_pad,
                (long long) geo.engram_token_map);
    const auto find = [&](const std::string& n) -> const strata::TensorInfo* {
        for (const auto& s : shards)
            if (const strata::TensorInfo* t = s->find(n)) return t;
        return nullptr;
    };
    if (!strata::core::glm_geometry_check(geo, find, 0, 0, err)) {
        std::fprintf(stderr, "FAIL shapes: %s\n", err.c_str());
        return 1;
    }
    std::printf("shapes:   every tensor of %d layers + NextN + globals + Engram table checks out\n", geo.n_layers);
    if (strata::core::glm_forward_available(geo, err)) {
        std::fprintf(stderr, "FAIL: the forward gate let a hy-v4-flash geometry through\n");
        return 1;
    }
    std::printf("gate:     %s\n", err.c_str());
    std::printf("OK\n");
    return 0;
}
