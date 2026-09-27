"""Tests for the JPEG export step (library RAWs + exported TIFFs -> JPEG)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.export import jpeg_job
from app.export.jpeg_writer import convert_image_to_jpeg, is_convertible


def test_is_convertible_accepts_every_supported_kind() -> None:
    assert is_convertible(Path("IMG_0001.CR3"))
    assert is_convertible(Path("a.TIFF"))
    assert is_convertible(Path("a.jpg"))
    assert is_convertible(Path("a.png"))
    assert not is_convertible(Path("a.txt"))


def test_convert_tiff_to_jpeg_round_trip(tmp_path: Path) -> None:
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    arr = np.stack([arr, arr, arr], axis=-1)
    src = tmp_path / "src.tif"
    dst = tmp_path / "out.jpg"
    tifffile.imwrite(src, arr, photometric="rgb")
    with patch("app.export.jpeg_writer._copy_exif"):
        convert_image_to_jpeg(src, dst, quality=92, long_edge=0, progressive=True)
    assert dst.exists() and dst.stat().st_size > 0
    decoded = np.asarray(Image.open(dst).convert("RGB"))
    assert decoded.shape == arr.shape


def test_convert_jpeg_without_resize_copies_bytes(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(src, format="JPEG")
    dst = tmp_path / "out" / "copy.jpg"
    convert_image_to_jpeg(src, dst, quality=50, long_edge=0, progressive=False)
    assert dst.read_bytes() == src.read_bytes()


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
    junk = tmp_path / "exported" / "B.cr3"  # ignored: only enhanced TIFFs live in exported/
    junk.write_bytes(b"\x00")
    jpg = tmp_path / "library" / "keep.jpg"  # library/ may hold any supported kind
    jpg.write_bytes(b"\x00")

    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)

    assert jpeg_job._list_candidates("library") == [raw, jpg]
    assert jpeg_job._list_candidates("exported") == [tif]
    assert jpeg_job._list_candidates("all") == [raw, jpg, tif]


def test_dest_for_preserves_subfolders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_subdir", "jpeg")
    raw = tmp_path / "library" / "2025" / "wedding" / "IMG_0001.CR3"
    tif = tmp_path / "exported" / "2025" / "wedding" / "IMG_0001.tif"
    assert jpeg_job._dest_for(raw) == tmp_path / "jpeg" / "2025" / "wedding" / "IMG_0001.jpg"
    assert jpeg_job._dest_for(tif) == tmp_path / "jpeg" / "2025" / "wedding" / "IMG_0001.jpg"


def test_list_candidates_recurses_subfolders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "library" / "2025" / "trip"
    nested.mkdir(parents=True)
    raw = nested / "A.CR3"
    raw.write_bytes(b"\x00")
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    assert jpeg_job._list_candidates("library") == [raw]


def test_run_jpeg_export_rejects_invalid_source() -> None:
    with pytest.raises(ValueError):
        jpeg_job.run_jpeg_export(source="bogus")


def test_run_jpeg_export_handles_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
