"""Shared JPEG/resize helpers used by both the cache tier and the export stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

from app.preview.jpeg_writer import resize_long_edge, write_jpeg


def test_resize_long_edge_noop_when_below_target() -> None:
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_long_edge(arr, long_edge=500) is arr


def test_resize_long_edge_zero_means_native() -> None:
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    assert resize_long_edge(arr, long_edge=0) is arr


def test_resize_long_edge_scales_to_target() -> None:
    arr = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out = resize_long_edge(arr, long_edge=1000)
    assert out.shape[:2] == (500, 1000)


def test_write_jpeg_honours_chroma_subsampling(tmp_path: Path) -> None:
    arr = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    dest_420 = tmp_path / "a.jpg"
    dest_444 = tmp_path / "b.jpg"
    write_jpeg(arr, dest_420, quality=90, subsampling=2)
    write_jpeg(arr, dest_444, quality=90, subsampling=0)
    with Image.open(dest_420) as im:
        assert JpegImagePlugin.get_sampling(im) == 2
    with Image.open(dest_444) as im:
        assert JpegImagePlugin.get_sampling(im) == 0


def test_write_jpeg_creates_parent_dirs(tmp_path: Path) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    dest = tmp_path / "nested" / "deeper" / "x.jpg"
    write_jpeg(arr, dest, quality=80)
    assert dest.exists()
