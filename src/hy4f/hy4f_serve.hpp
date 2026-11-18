// src/hy4f/hy4f_serve.hpp - maya-hy4f --serve: Maya's engine protocol (serve/server.py's StrataEngine) on stdin / stdout.
#pragma once

#include "hy4f.hpp"

#include <string>
#include <vector>

namespace strata::hy4f {

struct ServeOptions {
    std::vector<int> eos{120025};   // the tokens that end an answer (<｜hy_end:opensource｜>)
    int chunk = 64;                 // prompt tokens per PP line (and per prefill call, once the engine reads batches)
    std::string model_name = "Hy4-flash";
};

/// runs until QUIT or the end of stdin; the model is loaded
int serve(Model& m, const ServeOptions& so);

}  // namespace strata::hy4f
