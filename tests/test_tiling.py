"""Overlap-blended tiling: no hard steps at window boundaries, exact for a consistent model."""

from __future__ import annotations

import numpy as np

from app.arrays import Array
from app.enhancement.tiling import tiled_apply


def _up2(t: Array) -> Array:
    return np.repeat(np.repeat(t, 2, axis=0), 2, axis=1)


def test_feather_removes_per_tile_offset_steps() -> None:
    calls = {"n": 0}

    def fn(t: Array) -> Array:  # alternate windows come back 40 levels brighter
        calls["n"] += 1
        return np.clip(_up2(t).astype(np.int32) + (40 if calls["n"] % 2 else 0), 0, 255).astype(
            np.uint8
        )

    flat = np.full((200, 200, 3), 100, dtype=np.uint8)
    out = tiled_apply(flat, fn, tile=64, overlap=16, scale=2)
    assert out.shape == (400, 400, 3)
    assert calls["n"] == 16  # starts 0, 48, 96, 136 on each axis
    assert np.abs(np.diff(out.astype(np.int16), axis=0)).max() <= 2
    assert np.abs(np.diff(out.astype(np.int16), axis=1)).max() <= 2


def test_consistent_model_round_trips_within_rounding() -> None:
    img = np.random.default_rng(0).integers(0, 256, size=(130, 97, 3), dtype=np.uint8)
    out = tiled_apply(img, _up2, tile=64, overlap=16, scale=2)
    assert out.shape == (260, 194, 3)
    assert np.abs(out.astype(np.int16) - _up2(img).astype(np.int16)).max() <= 1


def test_image_smaller_than_tile_is_one_call() -> None:
    calls = {"n": 0}

    def fn(t: Array) -> Array:
        calls["n"] += 1
        return _up2(t)

    out = tiled_apply(np.full((40, 50, 3), 7, dtype=np.uint8), fn, tile=64, overlap=16, scale=2)
    assert calls["n"] == 1
    assert out.shape == (80, 100, 3)
    assert int(out.min()) == 7 == int(out.max())
