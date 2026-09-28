"""Enhancement entrypoint — selects candidates, then delegates to the batch runner.

For every photo whose decision action is in {keep_and_enhance, enhance_only},
`run_enhancement` warms up the GPU worker and hands the candidate snapshots to
`app.enhancement.batch.run_batch`, which executes the Auto Enhancement Engine
step-major over the whole batch (develop+plan+pre-AI per photo, then each AI
model loaded once over the photos that need it, then post-AI+verify+write).

This module keeps the pieces the batch runner reuses: candidate selection
(`_candidates`), the persistence helpers (`persist_report`/`persist_plan`/
`persist_verdict`), the RAW-deletion gate (`may_delete_source`), the developed-
TIFF loader (`_load_linear_float`), and the shared data records
(`PhotoCandidate`, `EnhanceSummary`). `run_enhancement` imports `batch` lazily
because `batch` imports from this module.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import tifffile
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.arrays import Array
from app.db import session_scope
from app.decision.rules import ENHANCE_ACTIONS
from app.enhancement.engine.metrics import as_linear_float01
from app.enhancement.engine.plan import EnhancementPlan, QualityReport, plan_to_json
from app.enhancement.verify import Verdict
from app.models import Decision, Face, Photo, PhotoQualityReport
from app.workers.gpu_worker import warmup

FaceBox = tuple[int, int, int, int]  # x, y, w, h in preview pixels


@dataclass(frozen=True)
class PhotoCandidate:
    """Detached snapshot of a photo row, safe to use after the session closes."""

    hash: str
    source_path: str
    file_kind: str | None
    action: str
    preview_path: str | None = None
    iso: int | None = None
    camera_make: str | None = None
    camera_body: str | None = None
    lens: str | None = None
    aperture: float | None = None
    focal_length: float | None = None


def _candidates() -> list[tuple[PhotoCandidate, list[FaceBox]]]:
    snapshots: list[tuple[PhotoCandidate, list[FaceBox]]] = []
    with session_scope() as sess:
        rows = sess.execute(
            select(
                Photo.hash,
                Photo.source_path,
                Photo.file_kind,
                Decision.action,
                Photo.preview_path,
                Photo.iso,
                Photo.camera_make,
                Photo.camera_body,
                Photo.lens,
                Photo.aperture,
                Photo.focal_length,
            )
            .join(Decision, Photo.hash == Decision.photo_hash)
            .where(Decision.action.in_(ENHANCE_ACTIONS))
        ).all()
        faces_by_hash: dict[str, list[FaceBox]] = {}
        face_rows = sess.execute(
            select(Face.photo_hash, Face.bbox_x, Face.bbox_y, Face.bbox_w, Face.bbox_h)
        ).all()
        for digest, x, y, w, h in face_rows:
            faces_by_hash.setdefault(digest, []).append((int(x), int(y), int(w), int(h)))
        for (
            digest,
            source_path,
            file_kind,
            action,
            preview_path,
            iso,
            camera_make,
            camera_body,
            lens,
            aperture,
            focal_length,
        ) in rows:
            snapshots.append(
                (
                    PhotoCandidate(
                        hash=digest,
                        source_path=source_path,
                        file_kind=file_kind,
                        action=action,
                        preview_path=preview_path,
                        iso=iso,
                        camera_make=camera_make,
                        camera_body=camera_body,
                        lens=lens,
                        aperture=aperture,
                        focal_length=focal_length,
                    ),
                    faces_by_hash.get(digest, []),
                )
            )
    return snapshots


def persist_report(sess: Session, photo_hash: str, report: QualityReport) -> None:
    """Upsert the quality_reports row for this photo from a QualityReport.

    Column names mirror the dataclass fields one-to-one, so the mapping is
    derived rather than spelled out. Wiped by `make reset`.
    """
    row = sess.get(PhotoQualityReport, photo_hash)
    if row is None:
        row = PhotoQualityReport(photo_hash=photo_hash)
        sess.add(row)
    for f in fields(report):
        setattr(row, f.name, getattr(report, f.name))
    # session_scope() sessions are autoflush=False; flush so persist_plan/persist_verdict's
    # sess.get(...) in the same transaction sees this row instead of raising "run report first".
    sess.flush()


def persist_plan(sess: Session, photo_hash: str, plan: EnhancementPlan) -> None:
    row = sess.get(PhotoQualityReport, photo_hash)
    if row is None:
        raise ValueError(f"persist_report must run before persist_plan for {photo_hash}")
    row.plan_json = plan_to_json(plan)


def persist_verdict(sess: Session, photo_hash: str, verdict: Verdict) -> None:
    row = sess.get(PhotoQualityReport, photo_hash)
    if row is None:
        raise ValueError(f"no quality report for {photo_hash}")
    row.verify_json = verdict.to_json()
    row.score_q_after = verdict.q_after
    row.degraded = verdict.degraded


def may_delete_source(photo: PhotoCandidate, out: Path, verdict: Verdict) -> bool:
    return photo.action == "enhance_only" and out.exists() and not verdict.degraded


def _load_linear_float(tiff_path: Path) -> Array:
    """Load darktable's 16-bit linear Rec.2020 TIFF as float32 in [0, 1]."""
    arr = tifffile.imread(str(tiff_path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {arr.shape} from {tiff_path}")
    return as_linear_float01(arr)


def preview_size(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as im:
        return im.size


@dataclass(frozen=True)
class EnhanceSummary:
    enhanced: int = 0
    skipped: int = 0  # non-RAW or missing source; nothing written
    failed: int = 0  # raised mid-chain; source left untouched, batch continued


def run_enhancement() -> EnhanceSummary:
    from app.enhancement.batch import run_batch  # local: batch imports this module

    warmup()
    return run_batch(_candidates())
