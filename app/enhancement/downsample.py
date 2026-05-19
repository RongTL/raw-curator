"""Lanczos resize helpers for the hybrid AI pipeline."""

from __future__ import annotations

import numpy as np
from PIL import Image


def lanczos_resize(arr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    w, h = target
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))


def scale(arr: np.ndarray, factor: float) -> np.ndarray:
    h, w = arr.shape[:2]
    return lanczos_resize(arr, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))))
