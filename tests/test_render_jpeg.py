from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.enhancement import render_jpeg


def test_render_paths_named_by_hash(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(render_jpeg.settings, "cache", tmp_path)
    p = render_jpeg.render_paths("a" * 32)
    assert p["before"] == tmp_path / "enhanced" / f"{'a' * 32}.before.jpg"
    assert p["after_full"] == tmp_path / "enhanced" / f"{'a' * 32}.after.full.jpg"


def test_write_render_caps_long_edge(tmp_path: Path) -> None:
    lin = np.zeros((100, 400, 3), dtype=np.float32)
    dest = tmp_path / "r.jpg"
    render_jpeg.write_render(lin, dest, long_edge=200, quality=90)
    with Image.open(dest) as im:
        assert max(im.size) == 200


def test_write_render_native_when_long_edge_zero(tmp_path: Path) -> None:
    lin = np.zeros((100, 400, 3), dtype=np.float32)
    dest = tmp_path / "r.jpg"
    render_jpeg.write_render(lin, dest, long_edge=0, quality=90)
    with Image.open(dest) as im:
        assert im.size == (400, 100)
