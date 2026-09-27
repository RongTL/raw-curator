"""Lanczos resize helpers for the hybrid AI pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.arrays import Array


def lanczos_resize(arr: Array, target: tuple[int, int]) -> Array:
    w, h = target
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))


def scale(arr: Array, factor: float) -> Array:
    h, w = arr.shape[:2]
    return lanczos_resize(arr, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))))
