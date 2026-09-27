"""Color corrections (spec §3.3 / §3.4) in float32 RGB [0,1].

Saturation is scaled around per-pixel luminance so colors don't bleed
into highlights. A skin-tone mask in YCbCr suppresses the desaturation
on faces — important when the engine pulls saturation down on a photo
that has both an oversaturated sunset and a person in frame.
"""

from __future__ import annotations

import numpy as np

from app.enhancement.colorspace import luma, rgb_to_ycbcr

# YCbCr skin-tone range per spec §3.2, normalized to [0,1].
_SKIN_CB_LO = 77.0 / 255.0
_SKIN_CB_HI = 127.0 / 255.0
_SKIN_CR_LO = 133.0 / 255.0
_SKIN_CR_HI = 173.0 / 255.0


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycbcr = rgb_to_ycbcr(rgb)
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]
    in_cb = ((cb >= _SKIN_CB_LO) & (cb <= _SKIN_CB_HI)).astype(np.float32)
    in_cr = ((cr >= _SKIN_CR_LO) & (cr <= _SKIN_CR_HI)).astype(np.float32)
    return in_cb * in_cr


def adjust_saturation(
    rgb: np.ndarray,
    factor: float,
    protect_skin: bool = True,
) -> np.ndarray:
    """Scale saturation by `factor` around luma; 1.0 is identity."""
    if abs(factor - 1.0) < 1e-4:
        return rgb
    lum = luma(rgb)[..., None]
    scale = float(factor)
    if protect_skin:
        m = _skin_mask(rgb)[..., None]
        scale_arr = 1.0 * m + scale * (1.0 - m)
        out = lum + (rgb - lum) * scale_arr
    else:
        out = lum + (rgb - lum) * scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)
