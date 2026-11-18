// src/hy4f/hy4f_vision.hpp - maya-hy4f vision: the Hy4 flash image encoder (mmproj GGUF, projector "hy_v4_flash") on one
// NVIDIA GPU.  Reference: llama.cpp + the AngelSlim patch (tools/mtmd/models/hy-v4-flash.cpp, tools/mtmd/clip.cpp,
// tools/mtmd/mtmd-image.cpp, tools/mtmd/mtmd.cpp), checked by tools/hy4f_vision_ref + hy4f_vision_test --ref.
//
// Pipeline (all of it matches llama.cpp, the float order aside):
//   decode     stb_image, forced to 3 channels (RGB8), like mtmd_helper_bitmap_init_from_buf
//   size       smart resize: align = patch 16 * merge 2 = 32, both edges rounded to a multiple of 32 (std::round),
//              then if w*h > max_pixels (4194304): edge = floor(edge / sqrt(h*w/max) / 32) * 32, elif < min_pixels (4096):
//              ceil(edge * sqrt(min/(h*w)) / 32) * 32  (mmproj keys clip.vision.image_{min,max}_pixels)
//   resize     Pillow-exact Lanczos-3 (fixed point, 22 bits), skipped when the size already matches
//   normalize  x = (u8 / 255 - 0.5) / 0.5
//   ViT        16x16 patch conv (+bias) + the learned 128x128 position grid bilinearly resampled to (ph, pw)
//              (sample point (i + 0.5) / ((n + 0.1) / 128) - 0.5, clamped), 27 pre-LN blocks: LN -> q,k,v (+bias),
//              16 heads x 72, full (non-causal, unmasked; config vision_full_attention=false notwithstanding, the
//              reference attends over all patches) softmax(qk / sqrt(72)) v -> out (+bias) -> residual; LN -> up 4304
//              (+bias) -> gelu (tanh) -> down (+bias) -> residual.  No pre/post LN, no rope in the encoder.
//   projector  RMSNorm (mm.pre_norm, eps 1e-5) -> 2x2 stride-2 conv 1152 -> 4608 (+bias) -> gelu (erf) -> 1x1 conv
//              4608 -> 3072 (+bias) -> one learned v.image_newline row appended after every output row.
// Output order: row-major over the merged grid with the newline last in each row:
//   token k = oy * (ow + 1) + ox, ox in [0, ow] (ox == ow is the newline), oy in [0, oh), ow = W/32, oh = H/32.
//   n_tokens = (ow + 1) * oh  (1120x640 -> 35x20 -> 720 tokens).
//
// ===================================== LM SPLICING CONTRACT (what the LM engine must do) =====================================
//  Token ids (config.json; the tokenizer text carries a revision hash, e.g. '<｜hy_image▁start:6124c78e｜>'):
//    image_start 120118, image_end 120119, image_pad 120120 (placeholder, never fed), image_newline 120121 (NOT fed:
//    the newline is the learned embedding row inside the encoder output), video_start/end/pad 120122/120123/120124.
//  Sequence (exactly what llama.cpp's mtmd builds for "...<__media__>..."):
//    [text ...] 120118 | n_tokens embedding rows from encode() (no token ids, the rows replace tok_embd) | 120119 [text ...]
//    i.e. the start/end are ordinary TEXT tokens embedded with tok_embd; only the rows between them are image embeddings.
//  Positions (llama.cpp reference): hy_v4_flash's decoder rope type is LLAMA_ROPE_TYPE_NORM (consecutive pairs), so mtmd
//    uses MTMD_POS_TYPE_NORMAL: image row i sits at position pos_0 + i (t = h = w = pos_0 + i, one plain 1-D rope
//    position), and the next text token (120119) is at pos_0 + n_tokens.  Output::pos / Output::n_pos carry exactly that
//    (PosMode::Sequential, the default).  The HF config does declare mrope_section [12,10,10] interleaved over the
//    64 rope dims, but the reference implementation does not use it; PosMode::Mrope2D gives the layout mtmd would use
//    for an M-RoPE decoder (t = pos_0, w = pos_0 + i % (ow+1), h = pos_0 + i / (ow+1), advance max(ow+1, oh)) for a
//    future HF-faithful experiment - do not mix it with parity checks against llama.cpp.
//  Engram: an embedding position has no n-gram.  At every image row the Engram contribution must be fully masked
//    (all columns invalid -> no delta added, exactly as if the module were skipped for that row; llama.cpp:
//    llama_model_hy_v4_flash::engram_hash, is_embd -> valid = 0).  Text tokens AFTER the image still hash normally:
//    their n-gram history is the previous token ids, and an image row in that history counts as token id -1
//    (llama.cpp stores an embedding row's KV cell token as hparams.ple_embd_token, which hy_v4_flash leaves at
//    LLAMA_TOKEN_NULL; -1 is outside engram_token_map -> "dead" -> the pad id), so 120119's history is
//    [pad, pad, ...] for the image rows before it.  The same holds for history before position 0.
//  Everything else (attention, HC, MoE, the k-pool indexer) treats image rows like any other position.
// ==========================================================================================================================
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::hy4f::vision {

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;  // interleaved RGB8, w*h*3
};

/// decode a PNG / JPEG / BMP / ... file to RGB8 (stb_image, 3 channels forced, like mtmd)
bool load_image(const std::string& path, Image& out, std::string& err);

enum class PosMode { Sequential, Mrope2D };

struct Pos {
    int32_t t, h, w;  // relative to the image's first position pos_0 (add pos_0)
};

/// positions of the n_tokens image rows (relative, pos_0 = 0) and the text position advance after the image
void image_positions(int ow, int oh, PosMode mode, std::vector<Pos>& pos, int& n_pos);

struct Output {
    int n_tokens = 0;          // (ow + 1) * oh
    int n_embd = 0;            // 3072
    int ow = 0, oh = 0;        // merged grid (without the newline column)
    int img_w = 0, img_h = 0;  // the resized image fed to the ViT (multiples of 32)
    const float* d_embd = nullptr;  // device [n_tokens][n_embd] F32 on the encoder's GPU; valid until the next encode()
    std::vector<float> embd;        // host copy [n_tokens][n_embd] when encode(..., to_host = true)
    std::vector<Pos> pos;           // per row, PosMode of the call
    int n_pos = 0;                  // text position advance after the image block
    double ms_preprocess = 0, ms_gpu = 0;  // host resize/normalize, GPU ViT+projector (synchronized)
};

struct Options {
    int gpu = 0;
    size_t attn_scratch_bytes = size_t(768) << 20;  // S (F32) + P (F16) per query chunk, all heads
    bool verbose = false;
};

class Encoder {
public:
    Encoder();
    ~Encoder();
    bool load(const std::string& mmproj_path, const Options& o, std::string& err);
    /// encode one image; the device buffers are reused across calls (grow only)
    bool encode(const Image& img, Output& out, std::string& err, bool to_host = true, PosMode mode = PosMode::Sequential);
    /// the resized size encode() will use for a w x h input
    void target_size(int w, int h, int& tw, int& th) const;
    int n_embd() const;

private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};

}  // namespace strata::hy4f::vision
