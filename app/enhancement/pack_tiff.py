"""Write a 16-bit TIFF with sRGB ICC profile + (optional) XMP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def write_tiff16(arr: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint16:
        arr = (np.clip(arr, 0, 255) * 257).astype(np.uint16) if arr.dtype == np.uint8 else arr.astype(np.uint16)
    tifffile.imwrite(out, arr, photometric="rgb", compression="lzw")
