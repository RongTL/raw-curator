"""The AI boundary converts linear Rec.2020 float to display sRGB uint8 and back."""

from __future__ import annotations

import numpy as np

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
