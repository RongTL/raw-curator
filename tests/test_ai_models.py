"""Model wrappers load once and degrade to identity when weights are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.enhancement.denoise import ScunetModel
from app.enhancement.face_restore import CodeFormerModel
from app.enhancement.upscale import RealEsrganModel

RGB = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("cls", [ScunetModel, CodeFormerModel])
def test_missing_weights_means_identity(
    cls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with cls() as m:
        assert m.available is False
        assert m.apply(RGB, strength=0.8, weight=0.8) is RGB


def test_realesrgan_without_weights_falls_back_to_lanczos_x2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with RealEsrganModel() as m:
        out = m.apply(RGB, fidelity=0.7)
    assert out.shape == (64, 96, 3)


def test_zero_strength_short_circuits_without_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with ScunetModel() as m:
        assert m.apply(RGB, strength=0.0) is RGB
