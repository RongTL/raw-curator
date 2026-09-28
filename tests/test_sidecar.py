"""User sidecars follow darktable's <name>.<ext>.xmp convention, mirrored by subfolder."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.enhancement.sidecar import resolve_xmp


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    photos = tmp_path / "photos"
    xmp = tmp_path / "xmp"
    src = photos / "incoming" / "trip" / "IMG_0001.CR3"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"")
    return photos, xmp, src


def test_mirrored_darktable_name_wins(tmp_path: Path) -> None:
    photos, xmp, src = _tree(tmp_path)
    want = xmp / "trip" / "IMG_0001.CR3.xmp"
    want.parent.mkdir(parents=True)
    want.write_text("")
    (xmp / "IMG_0001.xmp").write_text("")  # legacy also present
    assert resolve_xmp(src, photos_root=photos, xmp_root=xmp) == want


def test_legacy_stem_lookup_still_works_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    photos, xmp, src = _tree(tmp_path)
    xmp.mkdir()
    legacy = xmp / "IMG_0001.xmp"
    legacy.write_text("")
    with caplog.at_level(logging.WARNING):
        assert resolve_xmp(src, photos_root=photos, xmp_root=xmp) == legacy
    assert "legacy" in caplog.text


def test_baseline_used_when_no_user_sidecar(tmp_path: Path) -> None:
    photos, xmp, src = _tree(tmp_path)
    base = tmp_path / "base.xmp"
    base.write_text("")
    assert resolve_xmp(src, photos_root=photos, xmp_root=xmp, baseline=base) == base
    assert (
        resolve_xmp(src, photos_root=photos, xmp_root=xmp, baseline=tmp_path / "missing.xmp")
        is None
    )


def test_default_baseline_path_points_at_the_shipped_sidecar() -> None:
    from app.enhancement.sidecar import BASELINE_XMP

    assert BASELINE_XMP.name == "raw-curator-base.xmp" and BASELINE_XMP.exists()
