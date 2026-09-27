"""Shared luma / YCbCr helpers for the enhancement engine."""

from __future__ import annotations

import numpy as np

from app.enhancement.colorspace import luma, rgb_to_ycbcr


def test_luma_uses_rec709_weights() -> None:
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])
    out = luma(rgb.astype(np.float32))
    assert out.shape == (1, 4)
    assert np.allclose(out[0], [0.2126, 0.7152, 0.0722, 1.0], atol=1e-6)


def test_rgb_to_ycbcr_neutral_gray_is_centred() -> None:
    gray = np.full((2, 2, 3), 0.5, dtype=np.float32)
    ycc = rgb_to_ycbcr(gray)
    assert ycc.shape == (2, 2, 3)
    assert np.allclose(ycc[..., 0], 0.5, atol=1e-6)
    assert np.allclose(ycc[..., 1:], 0.5, atol=1e-6)
