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
    std::string dfly;                  // the DFly2 drafter GGUF (scripts/dfly_to_gguf.py); empty: none.  Drops the MTP layer
                                       // (VRAM) unless HY4F_KEEP_MTP
    bool verbose = false;
};

/// embedding rows (an encoded image) that replace the token embeddings of toks[offset .. offset + n) in prefill();
/// rows: host [n][n_embd] floats.  The image token ids there are placeholders (120120); the Engram is masked on them.
struct EmbdSpan {
    int offset = 0, n = 0;
    const float* rows = nullptr;
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Options& o, std::string& err);
    /// one token at position pos (pos must be the next position); logits (vocab) to the host when `logits` != null
    bool step(int32_t token, int64_t pos, float* logits, std::string& err);
    /// n tokens at positions pos .. pos + n - 1 in chunks of `chunk` (the batched path); the last token's logits
    bool prefill(const int32_t* toks, int n, int64_t pos, float* logits, int chunk, std::string& err,
                 const std::vector<EmbdSpan>* spans = nullptr);
    /// the image encoder (mmproj GGUF): weights in pinned host memory, encodes inside the batch buffers' arena on one GPU
    bool load_vision(const std::string& mmproj, std::string& err);
    bool has_vision() const;
    /// an encoded image file (PNG / JPEG / ...: a path or the bytes) -> rows [n_tokens][n_embd] for an EmbdSpan; call
    /// between prefill()/verify() calls.  The sequence is ... 120118, n_tokens rows, 120119 ...
    bool encode_image(const std::string& path, std::vector<float>& rows, int& n_tokens, std::string& err);
    bool encode_image_mem(const uint8_t* bytes, size_t n, std::vector<float>& rows, int& n_tokens, std::string& err);
    /// T (1..4) tokens at positions pos .. pos+T-1 (pos <= the next position: a rejected tail is overwritten), logits
    /// of every row ([T][vocab], may be null) and their argmax; keeps the rows' final hidden states for mtp()
    bool verify(const int32_t* toks, int T, int64_t pos, float* logits, int32_t* argmax, std::string& err);
    /// the MTP draft layer over (the last verify's hidden rows 0..T-1, next_toks[0..T)) at positions mpos ..; the
    /// draft for the token after next_toks[T-1].  After prefill, row 0 holds the prompt's last state.
    bool mtp(const int32_t* next_toks, int T, int64_t mpos, int32_t* draft, std::string& err, float* logits = nullptr);
    /// (logits: the MTP head's full row [vocab], for drafts sampled from it)
    /// the next draft from the MTP's own last state (after mtp() or a previous mtp_chain()) and the previous draft
    /// token at position mpos
    bool mtp_chain(int32_t tok, int64_t mpos, int32_t* draft, std::string& err, float* logits = nullptr);
    bool has_mtp() const;
    /// DFly2 drafts: n (1..5) draft tokens after `bonus` (the token at position `anchor`, not yet in the cache), from the
    /// last verify's row `src_row` (the last accepted token: its final hidden state and its top-k cells at the drafter's
    /// target layers).  After prefill, verify the prompt's last token alone (row 0) first.  Writes the block's latent
    /// rows at anchor .. anchor+5 of the target's DSA caches (the next verify overwrites them).
    bool dfly(int src_row, int32_t bonus, int64_t anchor, int n, int32_t* drafts, std::string& err, float* logits = nullptr);
    bool has_dfly() const;
    int vocab() const;
    std::string summary() const;

    // ---- serve support (maya-hy4f --serve, hy4f_serve.cpp; defined at the end of hy4f.cu) ----
    /// the next step's position: the cache holds positions 0 .. position() - 1
    int64_t position() const;
    int64_t max_context() const;
    /// keep the cache's first `pos` positions (the next step is at `pos`).  Every cache is indexed by position, so a
    /// later step finds the rows it reads - except in the rings: the SWA layers' (window + RING_PAD rows) and the
    /// indexer's (the current k-pool group).  A shorter prefix is kept where the rings still hold it (nothing written
    /// past it aliases a row it needs) or where mark()'s copy does, else the start of pos's indexer group where that is
    /// enough, else 0.  The MTP layer's caches follow best effort (a stale row only costs draft acceptance).  Returns
    /// the position kept (<= pos), -1 on a CUDA error (err).
    int64_t rewind(int64_t pos, std::string& err);
    /// back to position 0 (no copy: the caches are overwritten as the positions come again)
    void reset();
    /// copy the rings (trunk and MTP layer) as they are now (the prompt read, before the answer): a later rewind to
    /// this position, or to the one before it, gets them back - the next turn of a chat whose template drops the
    /// answer's reasoning
    bool mark(std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
    struct Snap;                  // (mark()'s copy; shared_ptr: complete only in hy4f.cu's serve block)
    std::shared_ptr<Snap> snap_;
    int64_t hw_ = 0;              // positions ever written since the last reset (the rings may hold up to it)
};

}  // namespace strata::hy4f
