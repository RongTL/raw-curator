"""Master TIFF: 16-bit, ICC embedded when available, EXIF carried over from the RAW."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.enhancement.pack_tiff import copy_metadata, write_tiff16

ICC = Path("tests/fixtures/linear_rec2020.icc").read_bytes()


def test_write_tiff16_embeds_icc_when_given(tmp_path: Path) -> None:
    arr = np.full((3, 3, 3), 0.5, dtype=np.float32)
    write_tiff16(arr, tmp_path / "m.tif", icc=ICC)
    with tifffile.TiffFile(tmp_path / "m.tif") as tf:
        page = tf.pages[0]
        assert page.dtype == np.uint16
        assert bytes(page.tags[34675].value) == ICC


def test_write_tiff16_without_icc_writes_untagged(tmp_path: Path) -> None:
    write_tiff16(np.zeros((2, 2, 3), dtype=np.uint16), tmp_path / "u.tif")
    with tifffile.TiffFile(tmp_path / "u.tif") as tf:
        assert 34675 not in tf.pages[0].tags


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not installed")
def test_copy_metadata_carries_exif_and_forces_upright(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    Image.new("RGB", (8, 8)).save(src)
    subprocess.run(
        ["exiftool", "-overwrite_original", "-Make=Canon", "-Orientation#=6", str(src)],
        check=True,
        capture_output=True,
    )
    dest = tmp_path / "m.tif"
    write_tiff16(np.zeros((2, 2, 3), dtype=np.uint16), dest)
    assert copy_metadata(src, dest) is True
    out = subprocess.run(
        ["exiftool", "-s", "-s", "-s", "-Make", "-Orientation#", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert out == ["Canon", "1"]
