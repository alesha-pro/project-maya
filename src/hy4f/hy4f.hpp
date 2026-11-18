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
    int vocab() const;
    std::string summary() const;

    // ---- serve support (maya-hy4f --serve, hy4f_serve.cpp; defined at the end of hy4f.cu) ----
    /// the next step's position: the cache holds positions 0 .. position() - 1
    int64_t position() const;
    int64_t max_context() const;
    /// keep the cache's first `pos` positions (the next step is at `pos`).  Every cache is indexed by position, so a
    /// later step finds the rows it reads - except in the two rings: the SWA layers' (the last `window` positions) and
    /// the indexer's (the current k-pool group).  A shorter prefix is kept where the rings still hold it (nothing was
    /// written past it that aliases a row it needs) or where mark()'s copy does, else the start of pos's indexer group
    /// where that is enough, else 0.  Returns the position kept (<= pos), -1 on a CUDA error (err).
    int64_t rewind(int64_t pos, std::string& err);
    /// back to position 0 (no copy: the caches are overwritten as the positions come again)
    void reset();
    /// copy both rings as they are now (the prompt read, before the answer): a later rewind to this position, or to
    /// the one before it, gets them back - the next turn of a chat whose template drops the answer's reasoning
    bool mark(std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
    struct Snap;                  // (mark()'s copy; shared_ptr: complete only in hy4f.cu's serve block)
    std::shared_ptr<Snap> snap_;
    int64_t hw_ = 0;              // positions ever written since the last reset (the rings may hold up to it)
};

}  // namespace strata::hy4f
