"""Schema-tolerant review lookups: enhanced badges degrade on a pre-0004 session DB
(one created before the plan/verify migration added score_q_after/plan_json/etc.)."""

from __future__ import annotations

from dataclasses import fields

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.routes._review_data import (
    enhanced_by_hash,
    face_counts_by_hash,
    quality_report_or_none,
)
from app.enhancement.engine.plan import QualityReport
from app.models import Face, Photo, PhotoQualityReport

# The metric columns are NOT NULL; fill them so a row can be inserted.
_METRICS = {f.name: 0.0 for f in fields(QualityReport)}


def _photo(sess: Session, h: str = "p1") -> None:
    sess.add(Photo(hash=h, source_path=f"/data/photos/incoming/{h}.CR3"))
    sess.flush()


def _add_report(sess: Session, h: str, **over: object) -> None:
    sess.add(PhotoQualityReport(photo_hash=h, **{**_METRICS, **over}))


def test_enhanced_by_hash_returns_after_scores(tmp_db: Session) -> None:
    _photo(tmp_db, "p1")
    _add_report(tmp_db, "p1", score_q_after=77.0, degraded=1)
    tmp_db.flush()
    assert enhanced_by_hash(tmp_db) == {"p1": (77.0, True)}


def test_enhanced_by_hash_empty_when_columns_missing(tmp_db: Session) -> None:
    _photo(tmp_db, "p1")
    _add_report(tmp_db, "p1", score_q_after=77.0, degraded=0)
    tmp_db.commit()
    # Simulate a pre-0004 DB: the enhanced columns don't exist yet.
    tmp_db.execute(text("ALTER TABLE quality_reports DROP COLUMN score_q_after"))
    tmp_db.commit()
    assert enhanced_by_hash(tmp_db) == {}


def test_quality_report_or_none_guards_missing_columns(tmp_db: Session) -> None:
    _photo(tmp_db, "p1")
    _add_report(tmp_db, "p1")
    tmp_db.commit()
    assert quality_report_or_none(tmp_db, "p1") is not None
    tmp_db.execute(text("ALTER TABLE quality_reports DROP COLUMN plan_json"))
    tmp_db.commit()
    assert quality_report_or_none(tmp_db, "p1") is None


def test_face_counts_by_hash_counts_per_photo(tmp_db: Session) -> None:
    _photo(tmp_db, "p1")
    tmp_db.add(Face(photo_hash="p1", bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1, det_score=0.9))
    tmp_db.add(Face(photo_hash="p1", bbox_x=2, bbox_y=2, bbox_w=1, bbox_h=1, det_score=0.8))
    tmp_db.flush()
    assert face_counts_by_hash(tmp_db) == {"p1": 2}
