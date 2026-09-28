from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.export import export_job
from app.models import Decision, Photo


def _jpeg(path: Path, size=(8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(path, "JPEG")


@pytest.fixture
def wired(tmp_db, tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    monkeypatch.setattr(export_job.settings, "photos", photos)
    monkeypatch.setattr(export_job.settings, "cache", tmp_path / "cache")
    monkeypatch.setattr(export_job.settings, "jpeg_subdir", "jpeg")
    monkeypatch.setattr(export_job.settings, "jpeg_quality", 90)
    monkeypatch.setattr(export_job.settings, "jpeg_long_edge", 0)
    import contextlib

    @contextlib.contextmanager
    def _scope():
        yield tmp_db

    monkeypatch.setattr(export_job, "session_scope", _scope)
    return photos


def test_enhanced_keep_raw_moves_to_library(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths

    raw = wired / "incoming" / "IMG.CR3"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=1))
    tmp_db.commit()

    export_job.run_export()

    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert (wired / "library" / "IMG.CR3").exists()  # archived
    assert not raw.exists()  # moved out of incoming
    assert tmp_db.get(Decision, "h" * 32).applied == 1


def test_enhanced_no_keep_deletes_raw_after_jpeg(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths

    raw = wired / "incoming" / "IMG.CR3"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=0))
    tmp_db.commit()

    export_job.run_export()

    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert not raw.exists()  # deleted (after JPEG)
    assert not (wired / "library" / "IMG.CR3").exists()


def test_original_uses_before_render(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths

    raw = wired / "incoming" / "IMG.CR3"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["before_full"], size=(10, 10))
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="original", keep_raw=1))
    tmp_db.commit()

    export_job.run_export()
    assert (wired / "jpeg" / "IMG.jpg").exists()


def test_discard_produces_nothing(wired, tmp_db) -> None:
    raw = wired / "incoming" / "IMG.CR3"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="discard"))
    tmp_db.commit()

    export_job.run_export()
    assert not (wired / "jpeg").exists()
    assert raw.exists()  # untouched
    assert tmp_db.get(Decision, "h" * 32).applied == 0


def test_second_run_skips_applied(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths

    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(
        Photo(hash="h" * 32, source_path=str(wired / "library" / "IMG.CR3"), file_kind="raw")
    )
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=1, applied=1))
    tmp_db.commit()
    export_job.run_export()  # applied → skipped, no crash on missing incoming RAW
    assert not (wired / "jpeg" / "IMG.jpg").exists()


def test_non_raw_original_uses_source(wired, tmp_db) -> None:
    src = wired / "incoming" / "IMG.JPG"
    _jpeg(src)
    tmp_db.add(Photo(hash="h" * 32, source_path=str(src), file_kind="jpeg"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="original", keep_raw=1))
    tmp_db.commit()
    export_job.run_export()
    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert (wired / "library" / "IMG.JPG").exists()  # keep_raw archives the original file


def test_encode_failure_leaves_raw(wired, tmp_db, monkeypatch) -> None:
    from app.enhancement.render_jpeg import render_paths

    raw = wired / "incoming" / "IMG.CR3"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=0))
    tmp_db.commit()

    def _boom(*a, **k):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(export_job, "convert_image_to_jpeg", _boom)

    summary = export_job.run_export()
    assert summary.failed == 1
    assert raw.exists()  # NOT deleted — encode failed before retention
    assert tmp_db.get(Decision, "h" * 32).applied == 0
