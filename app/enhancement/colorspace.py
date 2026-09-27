"""Colour-space helpers shared across the enhancement engine.

Every module used to carry its own copy of the Rec.709 luma weights and a
BT.601 YCbCr transform; this is the single definition. It also holds the
Rec.2020 <-> sRGB gamut matrices and the sRGB transfer functions used at the
AI / JPEG boundary.
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


# Rec.2020 (D65) <-> sRGB (D65) linear-light 3x3 matrices (BT.2087 / IEC 61966-2-1).
REC2020_TO_SRGB = np.array(
    [
        [1.6604910, -0.5876411, -0.0728499],
        [-0.1245505, 1.1328999, -0.0083494],
        [-0.0181508, -0.1005789, 1.1187297],
    ],
    dtype=np.float32,
)
SRGB_TO_REC2020 = np.array(
    [
        [0.6274040, 0.3292820, 0.0433136],
        [0.0690970, 0.9195400, 0.0113612],
        [0.0163916, 0.0880132, 0.8955950],
    ],
    dtype=np.float32,
)


def encode_srgb(lin: Array) -> Array:
    """Linear light -> sRGB-encoded, both float in [0, 1] (clips)."""
    f = np.clip(lin, 0.0, 1.0).astype(np.float32)
    p = np.power(f, 1.0 / 2.4)
    # ``p + 0.055 * (p - 1)`` is algebraically ``1.055 * p - 0.055`` but stays exactly 1.0
    # at white in float32 (the textbook form lands one ulp under) and is closer to the
    # float64 reference across the curve.
    return np.where(f <= 0.0031308, 12.92 * f, p + 0.055 * (p - 1.0)).astype(np.float32)


def decode_srgb(enc: Array) -> Array:
    """sRGB-encoded -> linear light, both float in [0, 1] (clips)."""
    f = np.clip(enc, 0.0, 1.0).astype(np.float32)
    return np.where(f <= 0.04045, f / 12.92, np.power((f + 0.055) / 1.055, 2.4)).astype(np.float32)


def rec2020_to_srgb_linear(lin: Array) -> Array:
    """Linear Rec.2020 -> linear sRGB primaries on HxWx3 float; no clipping."""
    return (lin @ REC2020_TO_SRGB.T).astype(np.float32)


def srgb_to_rec2020_linear(lin: Array) -> Array:
    """Linear sRGB -> linear Rec.2020 primaries on HxWx3 float; no clipping."""
    return (lin @ SRGB_TO_REC2020.T).astype(np.float32)


def linear_rec2020_to_srgb_u8(lin: Array) -> Array:
    """The AI / JPEG boundary: gamut-map by clipping, sRGB-encode, quantise."""
    return (encode_srgb(rec2020_to_srgb_linear(lin)) * 255.0 + 0.5).astype(np.uint8)


def srgb_u8_to_linear_rec2020(u8: Array) -> Array:
    """Inverse of the AI / JPEG boundary: uint8 sRGB -> float32 linear Rec.2020."""
    return srgb_to_rec2020_linear(decode_srgb(u8.astype(np.float32) / 255.0))
