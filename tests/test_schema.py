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


def test_photo_phash_index(tmp_db: Session) -> None:
    insp = inspect(tmp_db.bind)
    idx = {i["name"] for i in insp.get_indexes("photos")}
    assert "ix_photos_phash" in idx
    assert "ix_photos_captured_at" in idx
