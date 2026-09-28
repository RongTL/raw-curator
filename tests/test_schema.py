"""Verify the DB schema comes up clean after a fresh migration."""

from __future__ import annotations

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.models import SessionMeta

EXPECTED_TABLES = {
    "session_meta",
    "clusters",
    "photos",
    "photo_embeddings",
    "faces",
    "cluster_members",
    "decisions",
    "quality_reports",
}


def test_all_tables_exist(tmp_db: Session) -> None:
    insp = inspect(tmp_db.bind)
    tables = set(insp.get_table_names())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"missing tables: {missing}"


def test_session_meta_can_be_inserted(tmp_db: Session) -> None:
    tmp_db.add(SessionMeta(id=1))
    tmp_db.commit()
    row = tmp_db.execute(select(SessionMeta)).scalar_one()
    assert row.id == 1
    assert row.started_at is not None


def test_quality_reports_has_plan_and_verify_columns(tmp_db: Session) -> None:
    cols = {c["name"] for c in inspect(tmp_db.bind).get_columns("quality_reports")}
    assert {
        "plan_json",
        "verify_json",
        "score_q_after",
        "degraded",
        "neutral_fraction",
        "lap_var_top",
    } <= cols


def test_photo_phash_index(tmp_db: Session) -> None:
    insp = inspect(tmp_db.bind)
    idx = {i["name"] for i in insp.get_indexes("photos")}
    assert "ix_photos_phash" in idx
    assert "ix_photos_captured_at" in idx


def test_decision_has_export_choice_and_keep_raw(tmp_db: Session) -> None:
    from app.models import Decision, Photo

    # FK: decisions.photo_hash -> photos.hash (foreign_keys=ON), so seed the parent row.
    tmp_db.add(Photo(hash="a" * 32, source_path="/data/photos/incoming/a.cr3"))
    tmp_db.add(Decision(photo_hash="a" * 32))
    tmp_db.flush()
    row = tmp_db.get(Decision, "a" * 32)
    assert row is not None
    assert row.export_choice == "undecided"
    assert bool(row.keep_raw) is True
