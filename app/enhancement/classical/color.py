"""Color corrections (spec §3.3 / §3.4) in float32 RGB [0,1].

Saturation is scaled around per-pixel luminance so colors don't bleed
into highlights. A skin-tone mask in YCbCr suppresses the desaturation
on faces — important when the engine pulls saturation down on a photo
that has both an oversaturated sunset and a person in frame.
"""

from __future__ import annotations

import numpy as np

_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722

# YCbCr skin-tone range per spec §3.2, normalized to [0,1].
_SKIN_CB_LO = 77.0 / 255.0
_SKIN_CB_HI = 127.0 / 255.0
_SKIN_CR_LO = 133.0 / 255.0
_SKIN_CR_HI = 173.0 / 255.0


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    return np.stack([Y, Cb, Cr], axis=-1)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycbcr = _rgb_to_ycbcr(rgb)
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]
    in_cb = ((Cb >= _SKIN_CB_LO) & (Cb <= _SKIN_CB_HI)).astype(np.float32)
    in_cr = ((Cr >= _SKIN_CR_LO) & (Cr <= _SKIN_CR_HI)).astype(np.float32)
    return in_cb * in_cr


def adjust_saturation(
    rgb: np.ndarray,
    factor: float,
    protect_skin: bool = True,
) -> np.ndarray:
    """Scale saturation by `factor` around luma; 1.0 is identity."""
    if abs(factor - 1.0) < 1e-4:
        return rgb
    L = (_LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1] + _LUMA_B * rgb[..., 2])[..., None]
    scale = float(factor)
    if protect_skin:
        m = _skin_mask(rgb)[..., None]
        scale_arr = 1.0 * m + scale * (1.0 - m)
        out = L + (rgb - L) * scale_arr
    else:
        out = L + (rgb - L) * scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)
