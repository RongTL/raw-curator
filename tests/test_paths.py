"""Unit tests for relative_subpath path-mirroring helper."""

from __future__ import annotations

import logging
from pathlib import Path

from app.paths import relative_subpath


def test_nested_under_incoming() -> None:
    photos = Path("/data/photos")
    src = photos / "incoming" / "2025" / "wedding" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("2025/wedding/IMG_1234.CR3")


def test_nested_under_library() -> None:
    photos = Path("/data/photos")
    src = photos / "library" / "2025" / "wedding" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("2025/wedding/IMG_1234.CR3")


def test_nested_under_exported() -> None:
    photos = Path("/data/photos")
    src = photos / "exported" / "a" / "b" / "IMG.tif"
    assert relative_subpath(src, photos) == Path("a/b/IMG.tif")


def test_file_directly_at_root_has_no_subfolder() -> None:
    photos = Path("/data/photos")
    src = photos / "incoming" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("IMG_1234.CR3")


def test_unknown_root_falls_back_to_basename_and_warns(caplog) -> None:
    photos = Path("/data/photos")
    src = Path("/somewhere/else/IMG_9999.CR3")
    with caplog.at_level(logging.WARNING):
        result = relative_subpath(src, photos)
    assert result == Path("IMG_9999.CR3")
    assert any("IMG_9999.CR3" in r.getMessage() for r in caplog.records)


def test_suffix_swap_builds_expected_output_path() -> None:
    photos = Path("/data/photos")
    src = photos / "library" / "2025" / "trip" / "IMG.CR3"
    rel = relative_subpath(src, photos)
    assert (photos / "exported" / rel.with_suffix(".tif")) == (
        photos / "exported" / "2025" / "trip" / "IMG.tif"
    )
    assert (photos / "jpeg" / rel.with_suffix(".jpg")) == (
        photos / "jpeg" / "2025" / "trip" / "IMG.jpg"
    )
