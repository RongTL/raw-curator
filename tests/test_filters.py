"""Phase 3 unit tests for cheap CPU filters."""

from __future__ import annotations

import numpy as np

from app.filters.blur import laplacian_variance
from app.filters.exposure import exposure_flag
from app.filters.phash import dhash, hamming, phash


def test_blur_on_uniform_image_is_low() -> None:
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert laplacian_variance(arr) < 1.0


def test_exposure_overexposed() -> None:
    arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    flag, mean = exposure_flag(arr)
    assert flag in {"overexposed", "very_bright"}
    assert mean > 200


def test_phash_identical_images_zero_hamming() -> None:
    arr = np.random.RandomState(0).randint(0, 255, (128, 128, 3), dtype=np.uint8)
    h1 = phash(arr)
    h2 = phash(arr.copy())
    assert hamming(h1, h2) == 0


def test_dhash_works_on_uniform() -> None:
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert len(dhash(arr)) > 0
