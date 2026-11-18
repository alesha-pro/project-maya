// hy4f_vision_ref - the llama.cpp (+ AngelSlim patch) reference for the maya-hy4f vision encoder.
//
// Loads the Hy4 flash LM in vocab-only mode (mtmd needs the vocabulary for the image start/end tokens and the
// decoder rope type), the mmproj, tokenizes "<__media__>" with one image and encodes the image chunk.  Writes:
//   <out>/embd.f32     [n_tokens][n_embd]  the image embeddings exactly as mtmd hands them to the LM
//   <out>/pos.i32      [n_tokens][4]       mtmd_image_tokens_get_decoder_pos(pos_0 = 0) (t, x, y, z)
//   <out>/meta.txt     n_tokens, n_embd, n_pos (the text position advance), the text chunks' token ids
//
// build (against the llama.cpp build tree, not part of the maya cmake):
//   L=/mnt/nvme/engines/llama-cpp/llama.cpp-hy4-flash-2cbca38
//   g++ -O2 -std=c++17 tools/hy4f_vision_ref.cpp -I$L/include -I$L/ggml/include -I$L/tools/mtmd \
//       -L$L/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o build/hy4f_vision_ref
// usage: hy4f_vision_ref <lm-00001.gguf> <mmproj.gguf> <image> <out_dir> [--cpu] [--threads N] [--dump REGEX]
//   --dump: also save the clip graph tensors whose name matches (cb_eval) to <out>/t_<name>.f32 (+ manifest.tsv)
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <regex>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct DumpCtx {
    std::string dir;
    std::regex re;
    std::FILE* manifest = nullptr;
    std::vector<uint8_t> buf;
};

static bool dump_cb(struct ggml_tensor* t, bool ask, void* ud) {
    auto* d = static_cast<DumpCtx*>(ud);
    const bool want = t->type == GGML_TYPE_F32 && std::regex_match(t->name, d->re);
    if (ask) return want;
    if (!want) return true;
    const size_t nb = ggml_nbytes(t);
    d->buf.resize(nb);
    ggml_backend_tensor_get(t, d->buf.data(), 0, nb);
    std::vector<float> out(ggml_nelements(t));
    // walk the (possibly strided) tensor in ne order
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    out[k++] = *(const float*) (d->buf.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
    std::string name = t->name;
    for (auto& c : name) if (c == '/' || c == ' ') c = '_';
    std::FILE* f = std::fopen((d->dir + "/t_" + name + ".f32").c_str(), "wb");
    std::fwrite(out.data(), 4, out.size(), f);
    std::fclose(f);
    std::fprintf(d->manifest, "%s\t%lld\t%lld\t%lld\t%lld\n", t->name, (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3]);
    std::fflush(d->manifest);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <lm-00001.gguf> <mmproj.gguf> <image> <out_dir> [--cpu] [--threads N]\n", argv[0]);
        return 2;
    }
    const std::string lm = argv[1], mmproj = argv[2], image = argv[3], out = argv[4];
    bool use_gpu = true;
    int threads = 16;
    std::string dump_re;
    for (int i = 5; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu")) use_gpu = false;
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump_re = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(lm.c_str(), mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", lm.c_str()); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = use_gpu;
    cp.n_threads = threads;
    cp.print_timings = true;
    cp.warmup = false;
    DumpCtx dc;
    if (!dump_re.empty()) {
        dc.dir = out;
        dc.re = std::regex(dump_re);
        dc.manifest = std::fopen((out + "/manifest.tsv").c_str(), "w");
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = &dc;
    }
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot load mmproj\n"); return 1; }

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, image.c_str(), false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { std::fprintf(stderr, "cannot load image\n"); return 1; }
    std::fprintf(stderr, "image %ux%u\n", mtmd_bitmap_get_nx(bw.bitmap), mtmd_bitmap_get_ny(bw.bitmap));

    std::string prompt = std::string("A") + mtmd_default_marker() + "B";
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bms[1] = {bw.bitmap};
    if (int r = mtmd_tokenize(ctx, chunks, &txt, bms, 1)) { std::fprintf(stderr, "tokenize failed %d\n", r); return 1; }

    std::string meta;
    char line[512];
    std::snprintf(line, sizeof line, "use_mrope %d\n", (int) mtmd_decode_use_mrope(ctx));
    meta += line;
    const size_t nc = mtmd_input_chunks_size(chunks);
    for (size_t c = 0; c < nc; ++c) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(ch, &n);
            meta += "text_chunk";
            for (size_t i = 0; i < n; ++i) meta += " " + std::to_string(t[i]);
            meta += "\n";
            continue;
        }
        const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
        const size_t nt = mtmd_image_tokens_get_n_tokens(it);
        const llama_pos npos = mtmd_input_chunk_get_n_pos(ch);
        const auto t0 = std::chrono::steady_clock::now();
        if (int r = mtmd_encode_chunk(ctx, ch)) { std::fprintf(stderr, "encode failed %d\n", r); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int ne = llama_model_n_embd_inp(model);
        if (ne <= 0) ne = 3072; // vocab-only load leaves the LM hparams unset; mm.2 out = 3072
        const float* e = mtmd_get_output_embd(ctx);
        std::FILE* f = std::fopen((out + "/embd.f32").c_str(), "wb");
        std::fwrite(e, sizeof(float), nt * (size_t) ne, f);
        std::fclose(f);
        std::vector<int32_t> pos(nt * 4);
        for (size_t i = 0; i < nt; ++i) {
            mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, 0, i);
            pos[i * 4 + 0] = p.t; pos[i * 4 + 1] = p.x; pos[i * 4 + 2] = p.y; pos[i * 4 + 3] = p.z;
        }
        f = std::fopen((out + "/pos.i32").c_str(), "wb");
        std::fwrite(pos.data(), sizeof(int32_t), pos.size(), f);
        std::fclose(f);
        std::snprintf(line, sizeof line, "image_chunk n_tokens %zu n_embd %d n_pos %d encode_ms %.1f\n", nt, ne, (int) npos, ms);
        meta += line;
        std::fprintf(stderr, "%s", line);
    }
    std::FILE* f = std::fopen((out + "/meta.txt").c_str(), "w");
    std::fputs(meta.c_str(), f);
    std::fclose(f);
    std::fputs(meta.c_str(), stdout);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
