"""Derive per-stage progress and batch summary from DB + filesystem state.

Read-only: keeps pipeline job code untouched. Values are display-only,
best-effort; a `total` of 0 means indeterminate. Directory scans are cheap
relative to the 1-3 s UI poll (local SSD, one batch of photos).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

from sqlalchemy import Select, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.ingest.extensions import ALL_SUPPORTED_EXTS, JPEG_EXTS, TIFF_EXTS
from app.models import Cluster, Decision, Photo
from app.paths import relative_subpath

log = logging.getLogger(__name__)

ENHANCE_ACTIONS = ("keep_and_enhance", "enhance_only")

_warned_no_schema = False


def _warn_no_schema_once(exc: OperationalError) -> None:
    """Warn once per process; a schema-less session.db 500ing every poll is noise."""
    global _warned_no_schema
    if not _warned_no_schema:
        log.warning("progress query failed (DB schema missing? run `make reset`): %s", exc)
        _warned_no_schema = True


def _iter_files(root: Path, suffixes: frozenset[str] | None = None) -> Iterator[Path]:
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if suffixes is not None and p.suffix.lower() not in suffixes:
            continue
        yield p


def _count_files(root: Path, suffixes: frozenset[str] | None = None) -> int:
    return sum(1 for _ in _iter_files(root, suffixes))


def _count(sess: Session, stmt: Select[tuple[int]]) -> int:
    return sess.scalar(stmt) or 0


def _ingest_progress() -> tuple[int, int]:
    total = _count_files(settings.photos / "incoming", ALL_SUPPORTED_EXTS)
    with session_scope() as sess:
        done = _count(sess, select(func.count()).select_from(Photo))
    return done, total


def _filter_progress() -> tuple[int, int]:
    with session_scope() as sess:
        photos = _count(sess, select(func.count()).select_from(Photo))
        done = _count(
            sess, select(func.count()).select_from(Photo).where(Photo.blur_var.is_not(None))
        )
    return done, photos


def _score_progress() -> tuple[int, int]:
    # Face detection isn't tracked here; score "done" ~= CLIP + IQA done.
    with session_scope() as sess:
        photos = _count(sess, select(func.count()).select_from(Photo))
        done = _count(
            sess,
            select(func.count())
            .select_from(Photo)
            .where(Photo.technical_score.is_not(None), Photo.aesthetic_score.is_not(None)),
        )
    return done, photos


def _cluster_progress() -> tuple[int, int]:
    with session_scope() as sess:
        clusters = _count(sess, select(func.count()).select_from(Cluster))
    return (1 if clusters else 0), 1


def _submit_progress() -> tuple[int, int]:
    with session_scope() as sess:
        applied = _count(
            sess, select(func.count()).select_from(Decision).where(Decision.applied == 1)
        )
        decided = _count(
            sess,
            select(func.count())
            .select_from(Decision)
            .where(Decision.selected != "undecided"),
        )
    return applied, decided


def _enhance_progress() -> tuple[int, int]:
    # Enhance only develops RAW sources; non-RAW decisions never produce a TIFF.
    tiffs = _count_files(settings.photos / "exported", TIFF_EXTS)
    with session_scope() as sess:
        eligible = _count(
            sess,
            select(func.count())
            .select_from(Decision)
            .join(Photo, Photo.hash == Decision.photo_hash)
            .where(Decision.action.in_(ENHANCE_ACTIONS), Photo.file_kind == "raw"),
        )
    return min(tiffs, eligible), eligible


def _export_jpeg_progress() -> tuple[int, int]:
    # A kept RAW in library/ and its enhanced TIFF in exported/ share one jpeg
    # destination; count distinct destination stems, not source files.
    source_roots: tuple[tuple[Path, frozenset[str] | None], ...] = (
        (settings.photos / "library", None),
        (settings.photos / "exported", TIFF_EXTS),
    )
    destinations = {
        relative_subpath(p, settings.photos).with_suffix("")
        for root, suffixes in source_roots
        for p in _iter_files(root, suffixes)
    }
    jpegs = _count_files(settings.photos / settings.jpeg_subdir, JPEG_EXTS)
    total = len(destinations)
    return min(jpegs, total), total


_STAGE_HANDLERS: dict[str, Callable[[], tuple[int, int]]] = {
    "ingest": _ingest_progress,
    "filter": _filter_progress,
    "score": _score_progress,
    "cluster": _cluster_progress,
    "submit": _submit_progress,
    "enhance": _enhance_progress,
    "export-jpeg": _export_jpeg_progress,
}


def stage_progress(stage: str) -> tuple[int, int]:
    """(current, total) for display. total == 0 means indeterminate."""
    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        return 0, 0
    try:
        return handler()
    except OperationalError as exc:
        _warn_no_schema_once(exc)
        return 0, 0


def batch_summary() -> dict[str, int]:
    try:
        with session_scope() as sess:
            photos = _count(sess, select(func.count()).select_from(Photo))
            decided = _count(
                sess,
                select(func.count())
                .select_from(Decision)
                .where(Decision.selected != "undecided"),
            )
    except OperationalError as exc:
        _warn_no_schema_once(exc)
        return {"photos": 0, "decided": 0, "incoming_files": 0}
    return {
        "photos": photos,
        "decided": decided,
        "incoming_files": _count_files(settings.photos / "incoming", ALL_SUPPORTED_EXTS),
    }
