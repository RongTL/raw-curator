"""end_session wipes exactly the per-batch state: DB, cache tiers, and output trees."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.config import settings
from app.orchestrator.reset import end_session


@pytest.fixture
def batch_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    cache = tmp_path / "cache"
    photos = tmp_path / "photos"
    monkeypatch.setattr(settings, "cache", cache)
    monkeypatch.setattr(settings, "photos", photos)
    monkeypatch.setattr(settings, "jpeg_subdir", "jpeg")
    (cache / "previews").mkdir(parents=True)
    (cache / "thumbs").mkdir()
    (cache / "previews" / "a.jpg").write_bytes(b"x")
    (cache / "session.db").write_bytes(b"x")
    (cache / "session.db-wal").write_bytes(b"x")
    for sub in ("incoming", "library", "exported", "jpeg"):
        (photos / sub / "trip").mkdir(parents=True)
        (photos / sub / "trip" / "IMG.dat").write_bytes(b"x")
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    return {"cache": cache, "photos": photos, "_calls": calls}  # type: ignore[dict-item]


def test_end_session_wipes_db_cache_and_output_trees(batch_dirs: dict[str, Path]) -> None:
    end_session(force=True)
    cache, photos = batch_dirs["cache"], batch_dirs["photos"]
    assert not (cache / "session.db").exists()
    assert not (cache / "session.db-wal").exists()
    assert list((cache / "previews").iterdir()) == []
    for sub in ("library", "exported", "jpeg"):
        assert (photos / sub).is_dir()
        assert list((photos / sub).iterdir()) == [], sub


def test_end_session_leaves_incoming_alone(batch_dirs: dict[str, Path]) -> None:
    end_session(force=True)
    assert (batch_dirs["photos"] / "incoming" / "trip" / "IMG.dat").exists()


def test_end_session_reruns_migrations(batch_dirs: dict[str, Path]) -> None:
    end_session(force=True)
    calls = batch_dirs["_calls"]
    assert calls and calls[-1][:3] == ["alembic", "upgrade", "head"]  # type: ignore[index]
