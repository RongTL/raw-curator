"""Perceptual + difference hash on the thumbnail."""

from __future__ import annotations

import imagehash
from PIL import Image

from app.arrays import Array


def phash(rgb: Array, size: int = 8) -> str:
    return str(imagehash.phash(Image.fromarray(rgb), hash_size=size))


def dhash(rgb: Array, size: int = 8) -> str:
    return str(imagehash.dhash(Image.fromarray(rgb), hash_size=size))


def hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
