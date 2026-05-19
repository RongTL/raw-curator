"""Unit tests for tone_balance.recover_backlit."""

from __future__ import annotations

import numpy as np

from app.enhancement import tone_balance


def _backlit_image(h: int = 96, w: int = 96) -> np.ndarray:
    """Dark central subject on a bright background — canonical backlit scene.

    Subject occupies ~28% of the frame (deep shadows), background ~72%
    (bright highlights), no midtones — triggers the bimodal-histogram gate.
    """
    img = np.full((h, w, 3), 230, dtype=np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 25
    return img


def _balanced_image(h: int = 96, w: int = 96) -> np.ndarray:
    """Midtone-heavy image — no shadow density, no highlight density,
    should never trigger the backlit detector."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)


def _high_contrast_landscape(h: int = 96, w: int = 96) -> np.ndarray:
    """Bright sky + shadowed ground + midtone foliage.

    Regression case: this is high-contrast but NOT backlit (midtones are
    present). The tightened detector must not false-trigger on it.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 3] = 220  # sky
    img[h // 3 : 2 * h // 3] = 120  # midtone foliage
    img[2 * h // 3 :] = 50  # shadowed ground
    return img


def test_is_backlit_true_for_dark_subject_on_bright_bg() -> None:
    img = _backlit_image()
    luma = tone_balance._rgb_to_luma(img)
    assert tone_balance.is_backlit(luma) is True


def test_is_backlit_false_for_midtone_image() -> None:
    img = _balanced_image()
    luma = tone_balance._rgb_to_luma(img)
    assert tone_balance.is_backlit(luma) is False


def test_is_backlit_false_for_high_contrast_landscape() -> None:
    """Bright sky + dark ground but plenty of midtones — not backlit."""
    img = _high_contrast_landscape()
    luma = tone_balance._rgb_to_luma(img)
    assert tone_balance.is_backlit(luma) is False


def test_recover_backlit_lifts_shadows_when_triggered() -> None:
    img = _backlit_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.5)
    subject_before = img[32, 32].mean()
    subject_after = out[32, 32].mean()
    assert subject_after > subject_before + 5, (
        f"shadow lift should brighten subject; before={subject_before} after={subject_after}"
    )


def test_recover_backlit_protects_highlights() -> None:
    img = _backlit_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.6, highlight_protect=0.5)
    bg_before = int(img[0, 0].mean())
    bg_after = int(out[0, 0].mean())
    assert abs(bg_after - bg_before) <= 8, (
        f"highlights should be protected; before={bg_before} after={bg_after}"
    )


def test_recover_backlit_noop_when_not_backlit() -> None:
    img = _balanced_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.5)
    assert np.array_equal(img, out)


def test_recover_backlit_zero_lift_is_identity() -> None:
    img = _backlit_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.0)
    assert np.array_equal(img, out)


def test_recover_backlit_force_runs_on_balanced_image() -> None:
    img = _balanced_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.5, force=True)
    assert not np.array_equal(img, out)


def test_recover_backlit_output_is_uint8_and_bounded() -> None:
    img = _backlit_image()
    out = tone_balance.recover_backlit(img, shadow_lift=0.8, force=True)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 255


def test_recover_backlit_rejects_non_rgb() -> None:
    bad = np.zeros((10, 10), dtype=np.uint8)
    try:
        tone_balance.recover_backlit(bad, force=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-RGB input")
