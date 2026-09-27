"""Colour-space helpers shared across the enhancement engine.

Every module used to carry its own copy of the Rec.709 luma weights and a
BT.601 YCbCr transform; this is the single definition.
"""

from __future__ import annotations

import numpy as np

from app.arrays import Array

# Rec.709 / sRGB luminance weights.
LUMA_R = 0.2126
LUMA_G = 0.7152
LUMA_B = 0.0722


def luma(rgb: Array) -> Array:
    """Rec.709 luminance of an HxWx3 array; float32 in, float32 out (same scale)."""
    return (LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]).astype(np.float32)


def rgb_to_ycbcr(rgb: Array) -> Array:
    """BT.601 RGB -> YCbCr on float RGB in [0, 1]; Cb/Cr centred at 0.5."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)
