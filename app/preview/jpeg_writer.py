"""Resize + JPEG encoding helpers shared by the cache tier and the export stage."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.arrays import Array


def resize_long_edge(arr: Array, long_edge: int) -> Array:
    """Lanczos-downscale so the long edge is ``long_edge`` px. ``0`` (or any
    non-positive value) means native resolution; never upscales."""
    if long_edge <= 0:
        return arr
    h, w = arr.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_edge:
        return arr
    scale = long_edge / cur_long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


def write_jpeg(
    arr: Array,
    dest: Path,
    quality: int = 92,
    *,
    progressive: bool = True,
    subsampling: int | None = None,
) -> None:
    """Encode ``arr`` (HxWx3 uint8 RGB) to ``dest``, creating parent dirs.

    ``subsampling`` follows Pillow: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0; ``None``
    leaves the choice to libjpeg.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    extra: dict[str, int] = {} if subsampling is None else {"subsampling": subsampling}
    Image.fromarray(arr).save(
        dest, format="JPEG", quality=quality, optimize=True, progressive=progressive, **extra
    )


def decode_jpeg_bytes(jpeg_bytes: bytes) -> Array:
    """Decode JPEG bytes to HxWx3 uint8 RGB, applying EXIF orientation.

    The embedded thumbnail LibRaw hands back for a portrait RAW is stored in the
    sensor's landscape frame with an Orientation tag; without honouring it the
    grid thumb renders sideways while the developed preview (already upright) is
    correct. ``exif_transpose`` rotates the pixels and drops the tag.
    """
    with Image.open(io.BytesIO(jpeg_bytes)) as opened:
        img = ImageOps.exif_transpose(opened) or opened
        return np.asarray(img.convert("RGB"))
