"""Unit tests for the format-aware decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.ingest import decode
from app.ingest import extensions as ext


def test_classify_raw_extensions() -> None:
    assert decode.classify_kind(Path("a.cr2")) == decode.FileKind.RAW
    assert decode.classify_kind(Path("a.CR2")) == decode.FileKind.RAW
    assert decode.classify_kind(Path("a.CR3")) == decode.FileKind.RAW
    assert decode.classify_kind(Path("a.nef")) == decode.FileKind.RAW
    assert decode.classify_kind(Path("a.arw")) == decode.FileKind.RAW


def test_classify_image_extensions() -> None:
    assert decode.classify_kind(Path("a.jpg")) == decode.FileKind.JPEG
    assert decode.classify_kind(Path("a.JPEG")) == decode.FileKind.JPEG
    assert decode.classify_kind(Path("a.tif")) == decode.FileKind.TIFF
    assert decode.classify_kind(Path("a.tiff")) == decode.FileKind.TIFF
    assert decode.classify_kind(Path("a.png")) == decode.FileKind.PNG


def test_classify_prefers_content_over_extension(tmp_path: Path) -> None:
    """A JPEG saved with a RAW extension (e.g. a Picasa/Google Photos export
    renamed to .CR2) must classify as JPEG, not RAW — content beats extension."""
    arr = np.full((12, 16, 3), 200, dtype=np.uint8)
    src = tmp_path / "IMG_misnamed.CR2"
    Image.fromarray(arr).save(src, format="JPEG", quality=90)
    assert decode.classify_kind(src) == decode.FileKind.JPEG


def test_classify_png_with_raw_extension(tmp_path: Path) -> None:
    arr = np.full((8, 8, 3), 50, dtype=np.uint8)
    src = tmp_path / "shot.nef"
    Image.fromarray(arr).save(src, format="PNG")
    assert decode.classify_kind(src) == decode.FileKind.PNG


def test_classify_nonexistent_raw_falls_back_to_extension() -> None:
    """When the file can't be read, the extension still decides, so genuine
    RAW paths keep classifying as RAW."""
    assert decode.classify_kind(Path("missing.cr2")) == decode.FileKind.RAW


def test_decode_preview_jpeg_with_raw_extension(tmp_path: Path) -> None:
    """A misnamed JPEG must decode via the JPEG path instead of crashing in LibRaw."""
    arr = np.full((10, 14, 3), 128, dtype=np.uint8)
    src = tmp_path / "IMG_x.CR2"
    Image.fromarray(arr).save(src, format="JPEG", quality=92)
    out = decode.decode_preview(src)
    assert out.dtype == np.uint8
    assert out.shape == (10, 14, 3)


def test_classify_unknown() -> None:
    assert decode.classify_kind(Path("a.txt")) is None
    assert decode.classify_kind(Path("noext")) is None
    assert decode.classify_kind(Path("a.gif")) is None


def test_heic_classification_respects_pillow_heif() -> None:
    result = decode.classify_kind(Path("a.heic"))
    if decode.heic_available():
        assert result == decode.FileKind.HEIC
    else:
        assert result is None


def test_all_supported_exts_is_union() -> None:
    exts = decode.all_supported_exts()
    assert ".cr2" in exts
    assert ".cr3" in exts
    assert ".jpg" in exts
    assert ".tif" in exts
    assert ".png" in exts
    if decode.heic_available():
        assert ".heic" in exts


def test_extension_sets_pairwise_disjoint() -> None:
    sets = [ext.RAW_EXTS, ext.JPEG_EXTS, ext.TIFF_EXTS, ext.HEIC_EXTS, ext.PNG_EXTS]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert a.isdisjoint(b), f"{a} ∩ {b} non-empty"


def test_decode_preview_jpeg(tmp_path: Path) -> None:
    arr = np.full((20, 30, 3), 128, dtype=np.uint8)
    arr[5:15, 10:20] = 255
    src = tmp_path / "x.jpg"
    Image.fromarray(arr).save(src, format="JPEG", quality=95)
    out = decode.decode_preview(src)
    assert out.dtype == np.uint8
    assert out.shape == (20, 30, 3)


def test_decode_preview_tiff_uint16(tmp_path: Path) -> None:
    arr = np.full((10, 10, 3), 32768, dtype=np.uint16)
    src = tmp_path / "x.tif"
    tifffile.imwrite(src, arr, photometric="rgb")
    out = decode.decode_preview(src)
    assert out.dtype == np.uint8
    assert out.shape == (10, 10, 3)
    assert out[0, 0, 0] == 32768 >> 8


def test_decode_preview_png(tmp_path: Path) -> None:
    arr = np.full((8, 12, 3), 64, dtype=np.uint8)
    src = tmp_path / "x.png"
    Image.fromarray(arr).save(src, format="PNG")
    out = decode.decode_preview(src)
    assert out.shape == (8, 12, 3)


def test_decode_preview_unsupported_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "x.txt"
    bogus.write_text("hi")
    with pytest.raises(ValueError):
        decode.decode_preview(bogus)


def test_extract_thumb_bytes_returns_none_for_non_raw(tmp_path: Path) -> None:
    jpg = tmp_path / "x.jpg"
    Image.new("RGB", (10, 10)).save(jpg)
    assert decode.extract_thumb_bytes(jpg) is None


def test_jpeg_exif_orientation_is_applied(tmp_path: Path) -> None:
    """If EXIF says rotate, the decoded array should already be rotated."""
    arr = np.zeros((20, 40, 3), dtype=np.uint8)
    arr[:, :20] = 255
    img = Image.fromarray(arr)
    src = tmp_path / "rot.jpg"
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation=6 -> rotate 90 CW
    img.save(src, format="JPEG", exif=exif.tobytes())
    out = decode.decode_preview(src)
    assert out.shape == (40, 20, 3)


def test_load_tiff_rgb8_handles_uint16(tmp_path: Path) -> None:
    arr16 = np.full((10, 10, 3), 32768, dtype=np.uint16)
    tiff = tmp_path / "in.tif"
    tifffile.imwrite(tiff, arr16, photometric="rgb")
    out = decode.load_tiff_rgb8(tiff)
    assert out.dtype == np.uint8
    assert out.shape == (10, 10, 3)
    assert out[0, 0, 0] == 32768 >> 8


def test_load_tiff_rgb8_drops_alpha(tmp_path: Path) -> None:
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 3] = 255
    tiff = tmp_path / "rgba.tif"
    tifffile.imwrite(tiff, arr)
    assert decode.load_tiff_rgb8(tiff).shape == (4, 4, 3)


def test_load_tiff_rgb8_promotes_grayscale(tmp_path: Path) -> None:
    arr = np.full((6, 6), 128, dtype=np.uint8)
    tiff = tmp_path / "gray.tif"
    tifffile.imwrite(tiff, arr)
    assert decode.load_tiff_rgb8(tiff).shape == (6, 6, 3)


ICC = Path("tests/fixtures/linear_rec2020.icc").read_bytes()


def test_icc_description_parses_darktable_profile() -> None:
    assert decode.icc_description(ICC) == "Linear Rec2020 RGB"
    assert decode.icc_description(b"not an icc") is None


def test_load_tiff_rgb8_converts_linear_rec2020_masters(tmp_path: Path) -> None:
    lin = np.full((4, 4, 3), int(0.18 * 65535), dtype=np.uint16)
    tifffile.imwrite(
        tmp_path / "lin.tif", lin, photometric="rgb", extratags=[(34675, "B", len(ICC), ICC, True)]
    )
    out = decode.load_tiff_rgb8(tmp_path / "lin.tif")
    assert (
        out.dtype == np.uint8 and 115 <= int(out[0, 0, 1]) <= 120
    )  # encoded, not bit-shifted (46)


def test_load_tiff_rgb8_keeps_untagged_tiffs_as_srgb(tmp_path: Path) -> None:
    srgb = np.full((4, 4, 3), 30000, dtype=np.uint16)
    tifffile.imwrite(tmp_path / "srgb.tif", srgb, photometric="rgb")
    assert int(decode.load_tiff_rgb8(tmp_path / "srgb.tif")[0, 0, 0]) == 30000 >> 8


def test_icc_description_bounds_a_malformed_tag_count() -> None:
    import time

    icc = bytearray(132)  # header only, no room for any tag records
    icc[128:132] = (2**31).to_bytes(4, "big")  # absurd tag count
    start = time.perf_counter()
    result = decode.icc_description(bytes(icc))
    assert result is None
    assert time.perf_counter() - start < 1.0  # bounded, not 2**31 iterations
