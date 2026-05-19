"""JPEG encoding helpers for the cache tier."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def resize_long_edge(arr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = arr.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_edge:
        return arr
    scale = long_edge / cur_long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


def write_jpeg(arr: np.ndarray, dest: Path, quality: int = 92) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    img.save(dest, format="JPEG", quality=quality, optimize=True, progressive=True)


def decode_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
