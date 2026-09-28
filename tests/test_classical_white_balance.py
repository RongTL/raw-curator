"""Unit tests for gray-world white balance, including neutral-pixel gating.

`gray_world(neutral_only=True)` must estimate the cast from near-neutral
pixels only, so a strongly chromatic region (e.g. a saturated sky) cannot
drag the correction the way a whole-frame mean would.
"""

from __future__ import annotations

import numpy as np

from app.enhancement.classical.white_balance import gray_world


def test_gray_world_neutral_only_ignores_saturated_sky() -> None:
    # Top half: a saturated blue sky (chromatic — excluded by the neutral mask).
    # Bottom half: a near-neutral grey wall carrying a mild warm cast (r > g > b)
    # tight enough (spread < 8% of max) that the mask keeps it.
    img = np.zeros((40, 40, 3), dtype=np.float32)
    img[:20] = (0.2, 0.35, 0.9)  # saturated sky, excluded by the neutral mask
    img[20:] = (0.52, 0.50, 0.48)  # near-neutral warm grey, kept by the mask

    out = gray_world(img, target_rg=1.0, target_bg=1.0, strength=1.0, neutral_only=True)
    wall = out[20:].reshape(-1, 3).mean(axis=0)
    # Gains come from the wall alone, so its warm cast is neutralised: r/g, b/g -> 1.
    assert abs(wall[0] / wall[1] - 1.0) < 1e-3
    assert abs(wall[2] / wall[1] - 1.0) < 1e-3

    # A whole-frame gray-world lets the blue sky bias the gains, leaving the wall
    # un-neutralised — exactly the failure neutral_only avoids.
    whole = gray_world(img, target_rg=1.0, target_bg=1.0, strength=1.0, neutral_only=False)
    wall_whole = whole[20:].reshape(-1, 3).mean(axis=0)
    assert abs(wall_whole[0] / wall_whole[1] - 1.0) > 0.1


def test_gray_world_neutral_only_falls_back_when_too_few_neutrals() -> None:
    # Almost the whole frame is chromatic; the neutral mask covers < 2%, so the
    # step falls back to the whole-frame mean instead of dividing by ~zero.
    img = np.full((40, 40, 3), (0.2, 0.35, 0.9), dtype=np.float32)
    out = gray_world(img, strength=1.0, neutral_only=True)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
