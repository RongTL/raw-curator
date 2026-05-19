"""Coarse exposure flags from the thumbnail histogram."""

from __future__ import annotations

import numpy as np

_BLOWN_THRESH = 0.05
_CRUSHED_THRESH = 0.05


def exposure_flag(rgb: np.ndarray) -> tuple[str, float]:
    gray = rgb.mean(axis=-1) if rgb.ndim == 3 else rgb
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = float(hist.sum() or 1)
    blown = hist[-2:].sum() / total
    crushed = hist[:2].sum() / total
    mean = float(gray.mean())

    if blown > _BLOWN_THRESH and crushed > _CRUSHED_THRESH:
        flag = "high_contrast_clipped"
    elif blown > _BLOWN_THRESH:
        flag = "overexposed"
    elif crushed > _CRUSHED_THRESH:
        flag = "underexposed"
    elif mean < 32:
        flag = "very_dark"
    elif mean > 224:
        flag = "very_bright"
    else:
        flag = "ok"
    return flag, mean
