"""Perceptual + difference hash on the thumbnail."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image


def phash(rgb: np.ndarray, size: int = 8) -> str:
    return str(imagehash.phash(Image.fromarray(rgb), hash_size=size))


def dhash(rgb: np.ndarray, size: int = 8) -> str:
    return str(imagehash.dhash(Image.fromarray(rgb), hash_size=size))


def hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
