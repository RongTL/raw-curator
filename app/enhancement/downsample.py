"""Lanczos resize helpers for the hybrid AI pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.arrays import Array


def lanczos_resize(arr: Array, target: tuple[int, int]) -> Array:
    w, h = target
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.Resampling.LANCZOS))


def scale(arr: Array, factor: float) -> Array:
    h, w = arr.shape[:2]
    return lanczos_resize(arr, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))))


def resize_float(arr: Array, target: tuple[int, int]) -> Array:
    """Lanczos resize for float32 HxWx3 images in [0, 1]; `target` is (w, h) like `lanczos_resize`.

    PIL cannot build an image from float32 HxWx3, so we go through OpenCV, which resizes
    float arrays without quantising to 8 bits. Returns float32 HxWx3.
    """
    w, h = target
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
