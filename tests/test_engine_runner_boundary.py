"""The AI boundary converts linear Rec.2020 float to display sRGB uint8 and back."""

from __future__ import annotations

import numpy as np
import pytest

from app.enhancement.engine import runner


def test_boundary_encodes_linear_to_display_srgb() -> None:
    mid_grey_linear = np.full((2, 2, 3), 0.18, dtype=np.float32)
    u8 = runner._to_u8(mid_grey_linear)
    assert u8.dtype == np.uint8
    assert 115 <= int(u8[0, 0, 0]) <= 120  # 0.18 linear ~= 118/255 in sRGB, not 46/255


def test_boundary_round_trip_is_close_in_linear() -> None:
    from app.enhancement.colorspace import srgb_to_rec2020_linear

    # Build the input from sRGB-linear values so every colour is inside the sRGB gamut;
    # out-of-gamut Rec.2020 colours are clipped at the 8-bit boundary by design.
    lin = srgb_to_rec2020_linear(np.random.default_rng(3).random((4, 4, 3), dtype=np.float32) * 0.9)
    back = runner._from_u8(runner._to_u8(lin))
    assert np.abs(back - lin).max() < 0.01


def test_identity_model_leaves_the_float_image_untouched() -> None:
    lin = np.random.default_rng(4).random((8, 8, 3), dtype=np.float32) * 0.9
    out = runner.apply_ai_delta(lin, lambda u8: u8)
    assert np.allclose(out, lin, atol=1e-6)  # quantisation cancels: base keeps full precision


def test_delta_is_scaled_by_strength() -> None:
    lin = np.full((4, 4, 3), 0.2, dtype=np.float32)
    brighter = lambda u8: np.clip(u8.astype(int) + 40, 0, 255).astype(np.uint8)  # noqa: E731
    full = runner.apply_ai_delta(lin, brighter, strength=1.0)
    half = runner.apply_ai_delta(lin, brighter, strength=0.5)
    assert np.allclose(half - lin, (full - lin) * 0.5, atol=1e-4)


def test_x2_model_output_is_merged_onto_an_upsampled_base() -> None:
    lin = np.random.default_rng(5).random((6, 6, 3), dtype=np.float32) * 0.5
    x2 = lambda u8: np.repeat(np.repeat(u8, 2, axis=0), 2, axis=1)  # noqa: E731
    out = runner.apply_ai_delta(lin, x2, scale=2)
    assert out.shape == (12, 12, 3)


def test_resize_float_preserves_a_constant_image() -> None:
    from app.enhancement.downsample import resize_float

    const = np.full((6, 6, 3), 0.37, dtype=np.float32)
    out = resize_float(const, (12, 12))
    assert out.shape == (12, 12, 3)
    assert np.allclose(out, 0.37, atol=1e-6)


def test_scale2_requires_the_model_to_upscale() -> None:
    lin = np.random.default_rng(6).random((6, 6, 3), dtype=np.float32) * 0.5
    with pytest.raises(ValueError):
        runner.apply_ai_delta(lin, lambda u8: u8, scale=2)
