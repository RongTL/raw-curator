"""Shared luma / YCbCr helpers for the enhancement engine."""

from __future__ import annotations

import numpy as np

from app.enhancement.colorspace import (
    decode_srgb,
    encode_srgb,
    linear_rec2020_to_srgb_u8,
    linear_srgb_to_oklab,
    luma,
    rec2020_to_srgb_linear,
    rgb_to_ycbcr,
    srgb_to_rec2020_linear,
    srgb_u8_to_linear_rec2020,
)


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


def test_srgb_transfer_round_trip_and_anchors() -> None:
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32).reshape(1, 11, 1).repeat(3, axis=2)
    assert np.allclose(decode_srgb(encode_srgb(x)), x, atol=1e-6)
    assert (
        abs(
            float(encode_srgb(np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32))[0, 0, 0]) - 0.4613
        )
        < 1e-3
    )
    assert float(encode_srgb(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))[0, 0, 0]) == 1.0


def test_gamut_matrices_are_inverses_and_keep_white() -> None:
    white = np.ones((1, 1, 3), dtype=np.float32)
    assert np.allclose(rec2020_to_srgb_linear(white), white, atol=2e-3)
    rng = np.random.default_rng(1).random((4, 5, 3), dtype=np.float32)
    assert np.allclose(srgb_to_rec2020_linear(rec2020_to_srgb_linear(rng)), rng, atol=1e-4)


def test_oklab_white_is_neutral_and_red_is_chromatic() -> None:
    white = np.ones((1, 1, 3), dtype=np.float32)
    lab = linear_srgb_to_oklab(white)
    assert abs(float(lab[0, 0, 0]) - 1.0) < 1e-3  # L ~ 1
    assert abs(float(lab[0, 0, 1])) < 1e-3  # a ~ 0
    assert abs(float(lab[0, 0, 2])) < 1e-3  # b ~ 0
    red = np.zeros((1, 1, 3), dtype=np.float32)
    red[..., 0] = 1.0
    lab_r = linear_srgb_to_oklab(red)
    chroma = float(np.hypot(lab_r[0, 0, 1], lab_r[0, 0, 2]))
    assert chroma > 0.2


def test_u8_boundary_round_trip_is_within_one_code() -> None:
    lin = np.random.default_rng(2).random((6, 6, 3), dtype=np.float32) * 0.8 + 0.05
    u8 = linear_rec2020_to_srgb_u8(lin)
    assert u8.dtype == np.uint8
    back = srgb_u8_to_linear_rec2020(u8)
    assert np.abs(linear_rec2020_to_srgb_u8(back).astype(int) - u8.astype(int)).max() <= 1
