"""Tests for the JPEG export step (library RAWs + exported TIFFs -> JPEG)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.export import jpeg_job
from app.export.jpeg_writer import (
    RAW_EXTENSIONS,
    TIFF_EXTENSIONS,
    _load_tiff_as_rgb8,
    _resize_long_edge,
    convert_tiff_to_jpeg,
    is_raw,
    is_tiff,
)


def test_is_raw_recognises_common_formats() -> None:
    assert is_raw(Path("IMG_0001.CR3"))
    assert is_raw(Path("DSC_0042.nef"))
    assert is_raw(Path("foo.arw"))
    assert not is_raw(Path("foo.tif"))
    assert not is_raw(Path("foo.jpg"))


def test_is_tiff_recognises_both_extensions() -> None:
    assert is_tiff(Path("a.tif"))
    assert is_tiff(Path("a.TIFF"))
    assert not is_tiff(Path("a.cr3"))


def test_extension_sets_have_no_overlap() -> None:
    assert RAW_EXTENSIONS.isdisjoint(TIFF_EXTENSIONS)


def test_resize_long_edge_noop_when_below_target() -> None:
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    out = _resize_long_edge(arr, long_edge=500)
    assert out.shape == arr.shape


def test_resize_long_edge_noop_when_zero() -> None:
    arr = np.zeros((3000, 4000, 3), dtype=np.uint8)
    out = _resize_long_edge(arr, long_edge=0)
    assert out.shape == arr.shape


def test_resize_long_edge_scales_to_target() -> None:
    arr = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out = _resize_long_edge(arr, long_edge=1000)
    assert max(out.shape[:2]) == 1000
    assert out.shape[:2] == (500, 1000)


def test_load_tiff_as_rgb8_handles_uint16(tmp_path: Path) -> None:
    arr16 = np.full((10, 10, 3), 32768, dtype=np.uint16)
    tiff = tmp_path / "in.tif"
    tifffile.imwrite(tiff, arr16, photometric="rgb")
    out = _load_tiff_as_rgb8(tiff)
    assert out.dtype == np.uint8
    assert out.shape == (10, 10, 3)
    assert out[0, 0, 0] == 32768 >> 8


def test_load_tiff_as_rgb8_drops_alpha(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 3] = 255
    tiff = tmp_path / "rgba.tif"
    tifffile.imwrite(tiff, arr)
    out = _load_tiff_as_rgb8(tiff)
    assert out.shape == (4, 4, 3)


def test_load_tiff_as_rgb8_promotes_grayscale(tmp_path: Path) -> None:
    arr = np.full((6, 6), 128, dtype=np.uint8)
    tiff = tmp_path / "gray.tif"
    tifffile.imwrite(tiff, arr)
    out = _load_tiff_as_rgb8(tiff)
    assert out.shape == (6, 6, 3)


def test_convert_tiff_to_jpeg_round_trip(tmp_path: Path) -> None:
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    arr = np.stack([arr, arr, arr], axis=-1)
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.jpg"
    tifffile.imwrite(src, arr, photometric="rgb")
    with patch("app.export.jpeg_writer._copy_exif"):
        convert_tiff_to_jpeg(src, dst, quality=92, long_edge=0, progressive=True)
    assert dst.exists() and dst.stat().st_size > 0
    decoded = np.asarray(Image.open(dst).convert("RGB"))
    assert decoded.shape == arr.shape


def test_dest_for_uses_configured_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_subdir", "jpeg")
    assert jpeg_job._dest_for(Path("/x/y/IMG_0001.CR3")) == tmp_path / "jpeg" / "IMG_0001.jpg"
    assert jpeg_job._dest_for(Path("/x/y/IMG_0001.tif")) == tmp_path / "jpeg" / "IMG_0001.jpg"


def test_list_candidates_partitions_by_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "library").mkdir()
    (tmp_path / "exported").mkdir()
    raw = tmp_path / "library" / "A.CR3"
    raw.write_bytes(b"\x00")
    tif = tmp_path / "exported" / "B.tif"
    tif.write_bytes(b"\x00")
    junk = tmp_path / "exported" / "B.cr3"  # ignored: RAW in exported/
    junk.write_bytes(b"\x00")
    (tmp_path / "library" / "ignore.jpg").write_bytes(b"\x00")

    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)

    assert jpeg_job._list_candidates("library") == [raw]
    assert jpeg_job._list_candidates("exported") == [tif]
    assert jpeg_job._list_candidates("all") == [raw, tif]


def test_run_jpeg_export_rejects_invalid_source() -> None:
    with pytest.raises(ValueError):
        jpeg_job.run_jpeg_export(source="bogus")


def test_run_jpeg_export_handles_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    jpeg_job.run_jpeg_export(source="all")
    assert not (tmp_path / "jpeg").exists() or not any((tmp_path / "jpeg").iterdir())


def test_run_jpeg_export_skips_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "exported").mkdir()
    arr = np.full((8, 8, 3), 200, dtype=np.uint8)
    src = tmp_path / "exported" / "A.tif"
    tifffile.imwrite(src, arr, photometric="rgb")

    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_subdir", "jpeg")
    monkeypatch.setattr(jpeg_job.settings, "jpeg_quality", 90)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_long_edge", 0)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_progressive", True)

    out_dir = tmp_path / "jpeg"
    out_dir.mkdir()
    sentinel = out_dir / "A.jpg"
    sentinel.write_bytes(b"existing")

    with patch("app.export.jpeg_writer._copy_exif"):
        jpeg_job.run_jpeg_export(source="exported", overwrite=False)

    assert sentinel.read_bytes() == b"existing"

    with patch("app.export.jpeg_writer._copy_exif"):
        jpeg_job.run_jpeg_export(source="exported", overwrite=True)

    assert sentinel.read_bytes() != b"existing"
    assert sentinel.stat().st_size > 0
