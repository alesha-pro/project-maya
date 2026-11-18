// src/hy4f/hy4f.hpp - maya-hy4f: Tencent Hy4 flash (hy_v4_flash GGUF) on NVIDIA GPUs.
//
// v0 (this file's first cut): the correctness-first decode - one token through every layer, layer split over the
// visible GPUs, F32 activations and F32 latent KV, the routed / dense weights in their GGUF blocks on the device
// (Maya's native MMVQ), the Engram table in pinned host RAM read by the GPUs in place.  Math: docs/HY4F-SPEC.md of
// the hy4-flash project, checked against tools/hy4f_cpu_ref (the host F32 oracle) and llama.cpp's dumps.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4f {

struct Options {
    std::vector<std::string> shards;   // the GGUF shards (00001 first)
    std::vector<int> gpus;             // empty: every visible GPU
    std::vector<int> split;            // first layer of each later part (empty: by weight bytes)
    int64_t max_context = 8192;
    std::string dump_dir;              // seams per position: <dir>/<name>.f32 (appended token by token)
    bool verbose = false;
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Options& o, std::string& err);
    /// one token at position pos (pos must be the next position); logits (vocab) to the host when `logits` != null
    bool step(int32_t token, int64_t pos, float* logits, std::string& err);
    /// n tokens at positions pos .. pos + n - 1 in chunks of `chunk` (the batched path); the last token's logits
    bool prefill(const int32_t* toks, int n, int64_t pos, float* logits, int chunk, std::string& err);
    int vocab() const;
    std::string summary() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::hy4f
