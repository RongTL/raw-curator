"""Model-weight locations and VRAM-fit knobs come from Settings, not hardcoded paths/env."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.enhancement import weights


def test_weight_paths_follow_settings_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    assert weights.scunet_weights() == tmp_path / "scunet_color_real_psnr.pth"
    assert weights.realesrgan_weights() == tmp_path / "RealESRGAN_x2plus.pth"
    assert weights.codeformer_weights() == tmp_path / "CodeFormer/weights/CodeFormer/codeformer.pth"
    assert weights.codeformer_facelib_dir() == tmp_path / "CodeFormer/weights/facelib"


def test_scunet_tiling_reads_tile_size_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from app.enhancement.denoise import _tiled_forward

    monkeypatch.setattr(settings, "scunet_tile", 64)
    monkeypatch.setattr(settings, "scunet_tile_pad", 0)
    calls: list[tuple[int, int]] = []

    def identity(t: object) -> object:
        calls.append(tuple(t.shape[2:]))  # type: ignore[attr-defined]
        return t

    x = torch.zeros((1, 3, 128, 128))
    out = _tiled_forward(identity, x)
    assert out.shape == x.shape
    assert len(calls) == 4  # 128/64 = 2 tiles per axis


def test_codeformer_long_edge_cap_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    from app.enhancement.face_restore import _cap_long_edge

    monkeypatch.setattr(settings, "codeformer_max_long_edge", 100)
    rgb = np.zeros((300, 600, 3), dtype=np.uint8)
    capped = _cap_long_edge(rgb)
    assert max(capped.shape[:2]) == 100
    assert capped.shape[:2] == (50, 100)

    small = np.zeros((40, 80, 3), dtype=np.uint8)
    assert _cap_long_edge(small) is small
