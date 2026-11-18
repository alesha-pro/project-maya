"""serve/hy4_vision.py - pictures for Tencent Hy4 flash (maya-hy4f --mmproj runs the image encoder itself).

The template writes a picture as <|hy_image start|> <|hy_image pad|> <|hy_image end|> (120118, 120120, 120119); the
server turns the pad into the picture's n placeholder tokens, n = the encoder's rows for its size - the mmproj's smart
resize (both edges multiples of 32 = patch 16 x merge 2, the pixel count within [4096, 4194304]) to an ow x oh grid,
one learned newline row after each grid row: n = (ow + 1) * oh (src/hy4f/hy4f_vision.hpp).  The engine gets the
picture as decoded (EXIF orientation applied, transparent areas white, not resized: the encoder resizes it itself with
llama.cpp's Lanczos) in the same MMI1 file as MiniMax's (serve/mm3_vision.py), and checks n against its own rows.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
from dataclasses import dataclass

from serve import mm3_vision

MAX_RATIO = 200


@dataclass(frozen=True)
class Geometry:
    patch: int = 16
    merge: int = 2
    min_pixels: int = 4096
    max_pixels: int = 4194304
    image_token: int = 120120
    image_start: int = 120118
    image_end: int = 120119

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        """The engine's `INFO vision=hy4 vision_patch=... image_token=...` facts."""
        def num(k, d):
            v = info.get(k)
            return int(v) if isinstance(v, int) else d
        return cls(patch=num("vision_patch", 16), merge=num("vision_merge", 2),
                   min_pixels=num("vision_min_pixels", 4096), max_pixels=num("vision_max_pixels", 4194304),
                   image_token=num("image_token", 120120), image_start=num("image_start", 120118),
                   image_end=num("image_end", 120119))

    @property
    def factor(self) -> int:
        return self.patch * self.merge

    @property
    def max_pixels_total(self) -> int:
        return self.max_pixels

    @property
    def max_tokens(self) -> int:
        f = self.factor
        side = int(math.sqrt(self.max_pixels)) // f
        return (side + 1) * side


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def target_size(width: int, height: int, g: Geometry) -> tuple[int, int]:
    """hy4f_vision_pre.cpp's smart_size (llama.cpp's calc_size_preserved_ratio), its float32 steps included."""
    f = g.factor
    if max(height, width) / max(1, min(height, width)) > MAX_RATIO:
        raise ValueError(f"the image's aspect ratio is more than {MAX_RATIO}:1")

    def rnd(x):   # std::round(x / f) * f in float32 (halves away from zero)
        q = _f32(_f32(x) / f)
        return int(math.floor(q + 0.5)) * f

    w_bar, h_bar = max(f, rnd(width)), max(f, rnd(height))
    if h_bar * w_bar > g.max_pixels:
        beta = _f32(math.sqrt(_f32(_f32(_f32(float(height)) * width) / g.max_pixels)))
        h_bar = max(f, int(math.floor(_f32(_f32(height / beta) / f))) * f)
        w_bar = max(f, int(math.floor(_f32(_f32(width / beta) / f))) * f)
    elif h_bar * w_bar < g.min_pixels:
        beta = _f32(math.sqrt(_f32(g.min_pixels / _f32(_f32(float(height)) * width))))
        h_bar = int(math.ceil(_f32(_f32(height * beta) / f))) * f
        w_bar = int(math.ceil(_f32(_f32(width * beta) / f))) * f
    return w_bar, h_bar


def image_tokens(width: int, height: int, g: Geometry) -> int:
    tw, th = target_size(width, height, g)
    ow, oh = tw // g.factor, th // g.factor
    return (ow + 1) * oh


@dataclass
class Picture(mm3_vision.Picture):
    @property
    def tokens(self) -> int:
        """What the picture adds to the prompt the template rendered (its start / end tokens are there already):
        count_tokens adds tokens - 1 for the one pad token it replaces."""
        return self.n


def prepare(data: bytes, g: Geometry) -> Picture:
    """A picture's bytes (any format Pillow reads) -> its pixels as viewers see them (not resized)."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ValueError("pictures need Pillow (python -m pip install pillow)") from None
    try:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            im = ImageOps.exif_transpose(src)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
            rgb = im.tobytes()
    except Exception as e:  # noqa: BLE001 - whatever Pillow raises for a file it cannot read
        raise ValueError(f"the image could not be read ({e})") from None
    if im.width * im.height > (1 << 27):
        raise ValueError(f"the image is too large ({im.width}x{im.height})")
    return Picture(rgb, im.width, im.height, image_tokens(im.width, im.height, g))


class Vision(mm3_vision.Vision):
    """maya-hy4f's pictures: the decode (cached by the picture's bytes), the pad -> placeholders splice, the GENI file
    (MMI1, written by mm3_vision.Vision.write)."""

    def picture(self, source: str) -> Picture:
        data = self._load(source)
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            pic = self.cache.get(key)
            if pic is None:
                pic = prepare(data, self.g)
                self.cache[key] = pic
                while len(self.cache) > 16:             # oldest first
                    self.cache.pop(next(iter(self.cache)))
            return pic

    def expand(self, ids: list[int], pictures: list[Picture]) -> tuple[list[int], list[int]]:
        """Each pad token the template wrote (between its start and end tokens) -> the picture's n placeholders;
        -> (ids, the positions of the pictures' first placeholders)."""
        g = self.g
        out, starts, k = [], [], 0
        for t in ids:
            if t == g.image_token:
                if k >= len(pictures) or not out or out[-1] != g.image_start:
                    raise ValueError("the prompt and its images do not match")
                starts.append(len(out))
                out += [g.image_token] * pictures[k].n
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError("the prompt and its images do not match")
        return out, starts
