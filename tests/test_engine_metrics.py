"""Unit tests for the Auto Enhancement Engine measurement layer.

Tests build synthetic images and assert each metric responds in the
expected direction. Absolute thresholds are intentionally loose so the
tests stay stable across cv2 versions.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.enhancement.engine.metrics import (
    color_metrics,
    dynamic_range_metrics,
    exposure_metrics,
    measure_all,
    noise_metrics,
    sharpness_metrics,
    to_float01,
)
from app.enhancement.engine.scoring import score_report


def _gray(value: float, h: int = 256, w: int = 256) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.float32)


def _noisy(value: float, sigma: float, h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), value, dtype=np.float32)
    return np.clip(base + rng.normal(0, sigma, base.shape).astype(np.float32), 0.0, 1.0)


def test_exposure_metrics_on_midgray() -> None:
    m = exposure_metrics(_gray(0.5))
    assert m["shadow_clip"] == 0.0
    assert m["highlight_clip"] == 0.0
    assert 0.0 <= m["midtone_ratio"] <= 1.0


def test_exposure_metrics_flag_clipped_shadows() -> None:
    m = exposure_metrics(_gray(0.0))
    assert m["shadow_clip"] > 0.5


def test_exposure_metrics_flag_clipped_highlights() -> None:
    m = exposure_metrics(_gray(1.0))
    assert m["highlight_clip"] > 0.5


def test_dynamic_range_high_on_bimodal() -> None:
    img = np.zeros((128, 128, 3), dtype=np.float32)
    img[:, :64] = 0.0
    img[:, 64:] = 1.0
    m = dynamic_range_metrics(img)
    assert m["dr_p95_p5"] > 200.0


def test_dynamic_range_low_on_flat() -> None:
    m = dynamic_range_metrics(_gray(0.5))
    assert m["dr_p95_p5"] < 5.0


def test_color_metrics_detect_warm_cast() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[..., 0] = 0.7
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.3
    m = color_metrics(rgb)
    assert m["rg_ratio"] > 1.2
    assert m["bg_ratio"] < 0.9


def test_color_metrics_neutral_is_neutral() -> None:
    m = color_metrics(_gray(0.5))
    assert 0.95 < m["rg_ratio"] < 1.05
    assert 0.95 < m["bg_ratio"] < 1.05


def test_sharpness_low_on_blur() -> None:
    m = sharpness_metrics(_gray(0.5))
    assert m["lap_var"] < 1.0


def test_sharpness_higher_on_edges() -> None:
    img = np.zeros((128, 128, 3), dtype=np.float32)
    img[:, 64:] = 1.0
    m = sharpness_metrics(img)
    assert m["lap_var"] > 100.0


def test_noise_increases_with_sigma() -> None:
    quiet = noise_metrics(_gray(0.5))
    loud = noise_metrics(_noisy(0.5, sigma=0.10))
    assert loud["luma_noise"] > quiet["luma_noise"]


def test_to_float01_accepts_common_dtypes() -> None:
    u8 = np.array([[[0, 128, 255]]], dtype=np.uint8)
    f = to_float01(u8)
    assert f.dtype == np.float32
    assert f.max() == pytest.approx(1.0, abs=1e-3)
    u16 = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    f16 = to_float01(u16)
    assert f16.max() == pytest.approx(1.0, abs=1e-3)


def test_measure_all_returns_full_keyset() -> None:
    img = _noisy(0.5, sigma=0.02)
    m = measure_all(img)
    expected = {
        "mean_luma", "shadow_clip", "highlight_clip", "midtone_ratio", "midtone_deviation",
        "dr_p95_p5", "local_dr_mean",
        "rg_ratio", "bg_ratio", "avg_saturation", "oversat_ratio", "skin_hue_var",
        "lap_var", "edge_density", "hf_energy",
        "luma_noise", "chroma_noise",
    }
    assert expected.issubset(m.keys())


def test_score_report_q_in_range() -> None:
    img = _noisy(0.5, sigma=0.02)
    m = measure_all(img)
    r = score_report(m)
    assert 0.0 <= r.score_q <= 100.0
    for s in (r.score_exposure, r.score_dynamic_range, r.score_color, r.score_sharpness, r.score_noise):
        assert 0.0 <= s <= 100.0
