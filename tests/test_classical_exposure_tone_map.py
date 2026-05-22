"""Regression tests for the May 2026 enhancement-chain dark-output incident.

Two bugs in the classical pipeline combined to collapse mid-toned inputs to a
near-black "mostly black with a sliver of white" output, and the enhance_only
deletion path then destroyed the source RAW. These tests pin the fixes.
"""

from __future__ import annotations

import numpy as np

from app.enhancement.classical import exposure, tone_map


def _midtone_image(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.10, 0.60, size=(64, 64, 3)).astype(np.float32)


def test_highlight_recover_preserves_shadows_and_midtones() -> None:
    """Pixels below `knee` must pass through unchanged.

    Before the fix, `highlight_recover` returned `knee + rolled` for every
    pixel, and `rolled = 0` below knee, so the entire shadow/midtone range
    collapsed to a constant value of `knee` (= 0.78 in production).
    """
    img = _midtone_image()
    out = exposure.highlight_recover(img, amount=0.5, knee=0.78)

    below_mask = img <= 0.78
    assert below_mask.all(), "synthetic input should be entirely below knee"
    np.testing.assert_allclose(out[below_mask], img[below_mask], atol=1e-6)


def test_highlight_recover_remaps_highlights() -> None:
    """Pixels above `knee` get a soft roll-off into [knee, 1.0)."""
    ramp = np.linspace(0.0, 1.0, num=256, dtype=np.float32).reshape(16, 16, 1)
    img = np.broadcast_to(ramp, (16, 16, 3)).copy()
    out = exposure.highlight_recover(img, amount=0.5, knee=0.78)

    high_mask = img > 0.78
    assert high_mask.any(), "test setup must include highlights above knee"
    assert (out[high_mask] >= 0.78).all(), "highlights stay at or above knee"
    assert (out[high_mask] <= 1.0).all(), "highlights stay clipped to <= 1.0"
    # The roll-off should be monotone non-decreasing relative to input.
    flat_in = img[high_mask].ravel()
    flat_out = out[high_mask].ravel()
    order = np.argsort(flat_in)
    assert np.all(np.diff(flat_out[order]) >= -1e-6)


def test_global_compress_does_not_collapse_narrow_plateau() -> None:
    """A flat-ish image should not be hyper-stretched to [0, 1].

    Before the fix, `global_compress` re-normalized its output by the actual
    (min, max) of the compressed buffer. On narrow-range inputs, that
    stretched e.g. [0.715, 0.833] across [0, 1], slamming the mean to ~0.
    """
    rng = np.random.default_rng(0)
    plateau = np.full((32, 32, 3), 0.80, dtype=np.float32)
    plateau += rng.normal(0, 0.01, size=plateau.shape).astype(np.float32)
    plateau = np.clip(plateau, 0.0, 1.0)
    out = tone_map.global_compress(plateau, strength=0.5)

    # Compression legitimately darkens midtones (0.80 -> ~0.74 at strength=0.5).
    # The pre-fix bug would have collapsed the plateau to mean ~0.04 — a 0.76
    # shift — so a 0.10 tolerance still catches any regression of that class.
    assert abs(float(out.mean()) - float(plateau.mean())) < 0.10, (
        f"mean shifted from {float(plateau.mean()):.3f} to {float(out.mean()):.3f}"
    )
    assert float(out.std()) < 0.05, "narrow input must produce a narrow output"


def test_global_compress_preserves_endpoints() -> None:
    """The sigmoid maps 0 -> 0 and 1 -> 1 regardless of strength."""
    edges = np.array([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]], dtype=np.float32)
    for s in (0.1, 0.5, 0.9, 1.0):
        out = tone_map.global_compress(edges, strength=s)
        np.testing.assert_allclose(out[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[1], 1.0, atol=1e-6)


def test_chain_preserves_mean_on_normal_input() -> None:
    """End-to-end repro: a mid-toned image should not lose 80%+ of its mean
    after passing through highlight_recover -> global_compress.

    This is the exact failure mode of the May 2026 incident — the on-disk
    broken IMG_2593.tif had mean 0.056 from an input mean of 0.30.
    """
    img = _midtone_image()
    input_mean = float(img.mean())
    after_hl = exposure.highlight_recover(img, amount=0.15, knee=0.78)
    after_gc = tone_map.global_compress(after_hl, strength=0.53)
    output_mean = float(after_gc.mean())
    ratio = output_mean / input_mean
    assert ratio > 0.5, (
        f"chain darkened mean from {input_mean:.3f} to {output_mean:.3f} "
        f"(ratio {ratio:.2f}) — bug regression"
    )
