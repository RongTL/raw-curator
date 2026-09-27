"""Overlap-blended tiling for full-frame neural models.

Real-ESRGAN's own tiler crops the padding away and abuts tiles; per-tile mean
drift then shows as a grid in skies and other smooth gradients. This runs the
model on overlapping windows and cross-fades them with a linear feather so no
hard tile boundary survives. Windows are exactly ``tile`` px, so peak memory
equals the un-feathered tiler's.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.arrays import Array


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(n: int, overlap: int, *, first: bool, last: bool) -> Array:
    """Per-pixel weight along one axis: 1 inside, a linear fade over ``overlap`` px on each
    side that has a neighbour. Two neighbours' fades sum to exactly 1 across the overlap."""
    w = np.ones(n, dtype=np.float32)
    k = min(overlap, n)
    if k > 0:
        r = np.arange(1, k + 1, dtype=np.float32) / (k + 1)
        if not first:
            w[:k] = r
        if not last:
            w[n - k :] = np.minimum(w[n - k :], r[::-1])
    return w


def tiled_apply(
    rgb: Array,
    fn: Callable[[Array], Array],
    *,
    tile: int,
    overlap: int,
    scale: int,
) -> Array:
    """Apply ``fn`` (HxWx3 uint8 -> (H*scale)x(W*scale)x3 uint8) over ``tile``-px windows
    overlapping by ``overlap`` px; overlaps are cross-faded with a linear feather."""
    h, w = rgb.shape[:2]
    stride = max(1, tile - overlap)
    acc = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    wsum = np.zeros((h * scale, w * scale, 1), dtype=np.float32)
    for y0 in _starts(h, tile, stride):
        for x0 in _starts(w, tile, stride):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            out = fn(rgb[y0:y1, x0:x1]).astype(np.float32)
            oh, ow = out.shape[:2]
            wy = _ramp(oh, overlap * scale, first=y0 == 0, last=y1 == h)
            wx = _ramp(ow, overlap * scale, first=x0 == 0, last=x1 == w)
            wgt = (wy[:, None] * wx[None, :])[..., None]
            ys, xs = slice(y0 * scale, y0 * scale + oh), slice(x0 * scale, x0 * scale + ow)
            acc[ys, xs] += out * wgt
            wsum[ys, xs] += wgt
    return np.clip(np.rint(acc / np.maximum(wsum, 1e-6)), 0, 255).astype(np.uint8)
