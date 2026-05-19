"""Final-stage Lanczos resize back to the user's target resolution.

`settings.enhance_target_res` accepts:
    "native"        -> resize to the source RAW's native (w, h)
    "200%"          -> 2x native (any "<int>%" works)
    "WIDTHxHEIGHT"  -> explicit pixel size, e.g. "3840x2160"
"""

from __future__ import annotations

import numpy as np

from app.config import settings
from app.enhancement.downsample import lanczos_resize


def _parse_target(spec: str, native: tuple[int, int]) -> tuple[int, int]:
    native_w, native_h = native
    s = spec.strip().lower()
    if s == "native":
        return native_w, native_h
    if s.endswith("%"):
        pct = float(s[:-1]) / 100.0
        return max(1, int(round(native_w * pct))), max(1, int(round(native_h * pct)))
    if "x" in s:
        w_str, h_str = s.split("x", 1)
        return max(1, int(w_str)), max(1, int(h_str))
    raise ValueError(f"unrecognised enhance_target_res: {spec!r}")


def upsample_final(arr: np.ndarray, native_size: tuple[int, int]) -> np.ndarray:
    target = _parse_target(settings.enhance_target_res, native_size)
    h, w = arr.shape[:2]
    if (w, h) == target:
        return arr
    return lanczos_resize(arr, target)
