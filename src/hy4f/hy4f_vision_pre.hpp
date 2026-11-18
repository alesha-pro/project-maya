// src/hy4f/hy4f_vision_pre.hpp - internal: host preprocessing of the Hy4 flash vision encoder (hy4f_vision_pre.cpp).
#pragma once

#include "hy4f_vision.hpp"

namespace strata::hy4f::vision::detail {

/// mtmd's dyn_size target: both edges multiples of `align`, the pixel count clamped to [min_pixels, max_pixels]
void smart_size(int width, int height, int align, int min_pixels, int max_pixels, int& tw, int& th);
/// Pillow-exact Lanczos-3 resample (fixed point), as llama.cpp's img_tool::resize_pillow
void resize_lanczos(const Image& src, int tw, int th, Image& dst);

}  // namespace strata::hy4f::vision::detail
