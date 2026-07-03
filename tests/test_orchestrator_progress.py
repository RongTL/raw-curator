"""Stage progress derived from DB + filesystem state."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.models import Cluster, Decision, Photo
from app.orchestrator import progress


@pytest.fixture
def env(tmp_db, tmp_path: Path, monkeypatch):
    @contextmanager
    def fake_scope():
        yield tmp_db

    fake_settings = SimpleNamespace(photos=tmp_path / "photos", jpeg_subdir="jpeg")
    (tmp_path / "photos" / "incoming").mkdir(parents=True)
    monkeypatch.setattr(progress, "session_scope", fake_scope)
    monkeypatch.setattr(progress, "settings", fake_settings)
    return tmp_db, fake_settings


def test_filter_progress_counts_blur_var(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3", blur_var=10.0))
    sess.add(Photo(hash="b" * 32, source_path="/x/b.cr3"))
    sess.flush()
    assert progress.stage_progress("filter") == (1, 2)


def test_score_progress_requires_technical_and_aesthetic(env) -> None:
    sess, _ = env
    sess.add(
        Photo(hash="a" * 32, source_path="/x/a.cr3", technical_score=0.7, aesthetic_score=5.0)
    )
    sess.add(Photo(hash="b" * 32, source_path="/x/b.cr3", technical_score=0.4))
    sess.flush()
    assert progress.stage_progress("score") == (1, 2)


def test_ingest_progress_counts_incoming_files(env) -> None:
    sess, settings = env
    (settings.photos / "incoming" / "a.cr3").write_bytes(b"x")
    (settings.photos / "incoming" / "b.cr3").write_bytes(b"x")
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3"))
    sess.flush()
    assert progress.stage_progress("ingest") == (1, 2)


def test_ingest_progress_ignores_unsupported_files(env) -> None:
    sess, settings = env
    (settings.photos / "incoming" / "a.cr3").write_bytes(b"x")
    (settings.photos / "incoming" / "b.mov").write_bytes(b"x")
    (settings.photos / "incoming" / "a.xmp").write_bytes(b"x")
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3"))
    sess.flush()
    assert progress.stage_progress("ingest") == (1, 1)


def test_cluster_progress_flips_when_clusters_exist(env) -> None:
    sess, _ = env
    assert progress.stage_progress("cluster") == (0, 1)
    sess.add(Cluster(kind="burst"))
    sess.flush()
    assert progress.stage_progress("cluster") == (1, 1)


def test_submit_progress_counts_applied_vs_decided(env) -> None:
    sess, _ = env
    rows = [("yes", 1), ("no", 0), ("undecided", 0)]
    for i, (selected, applied) in enumerate(rows):
        h = str(i) * 32
        sess.add(Photo(hash=h, source_path=f"/x/{i}.cr3"))
        sess.add(Decision(photo_hash=h, selected=selected, applied=applied))
    sess.flush()
    assert progress.stage_progress("submit") == (1, 2)


def test_enhance_progress_counts_tiffs_vs_eligible(env) -> None:
    sess, settings = env
    for i, action in enumerate(["keep_and_enhance", "enhance_only", "none"]):
        h = str(i) * 32
        sess.add(Photo(hash=h, source_path=f"/x/{i}.cr3", file_kind="raw"))
        sess.add(Decision(photo_hash=h, selected="yes", action=action))
    sess.flush()
    exported = settings.photos / "exported"
    exported.mkdir(parents=True)
    (exported / "0.tif").write_bytes(b"x")
    assert progress.stage_progress("enhance") == (1, 2)


def test_enhance_progress_excludes_non_raw_sources(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3", file_kind="raw"))
    sess.add(Decision(photo_hash="a" * 32, selected="yes", action="keep_and_enhance"))
    sess.add(Photo(hash="b" * 32, source_path="/x/b.heic", file_kind="heic"))
    sess.add(Decision(photo_hash="b" * 32, selected="no", action="enhance_only"))
    sess.flush()
    assert progress.stage_progress("enhance") == (0, 1)


def test_export_jpeg_counts_distinct_destinations(env) -> None:
    _, settings = env
    # A "yes" photo: RAW kept in library/ AND its TIFF in exported/ share one
    # jpeg destination, so total must be 1, not 2.
    (settings.photos / "library").mkdir()
    (settings.photos / "library" / "a.cr3").write_bytes(b"x")
    (settings.photos / "exported").mkdir()
    (settings.photos / "exported" / "a.tif").write_bytes(b"x")
    (settings.photos / "jpeg").mkdir()
    (settings.photos / "jpeg" / "a.jpg").write_bytes(b"x")
    assert progress.stage_progress("export-jpeg") == (1, 1)


def test_batch_summary_shape(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3"))
    sess.add(Decision(photo_hash="a" * 32, selected="yes"))
    sess.flush()
    summary = progress.batch_summary()
    assert summary == {"photos": 1, "decided": 1, "incoming_files": 0}


def test_batch_summary_counts_only_supported_incoming(env) -> None:
    _, settings = env
    (settings.photos / "incoming" / "a.cr3").write_bytes(b"x")
    (settings.photos / "incoming" / "b.mov").write_bytes(b"x")
    assert progress.batch_summary()["incoming_files"] == 1


def test_unknown_stage_is_indeterminate(env) -> None:
    assert progress.stage_progress("nope") == (0, 0)


def test_missing_tables_returns_indeterminate(tmp_path: Path, monkeypatch) -> None:
    # A session.db with no tables (serve before reset/alembic) must degrade to
    # the indeterminate shape instead of raising on every poll.
    engine = make_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    make_session = sessionmaker(bind=engine, future=True)

    @contextmanager
    def fake_scope():
        sess = make_session()
        try:
            yield sess
        finally:
            sess.close()

    fake_settings = SimpleNamespace(photos=tmp_path / "photos", jpeg_subdir="jpeg")
    monkeypatch.setattr(progress, "session_scope", fake_scope)
    monkeypatch.setattr(progress, "settings", fake_settings)
    try:
        assert progress.stage_progress("filter") == (0, 0)
        assert progress.batch_summary() == {"photos": 0, "decided": 0, "incoming_files": 0}
    finally:
        engine.dispose()
