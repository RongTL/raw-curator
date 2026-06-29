"""Unit tests for stage_all bulk-decision helper."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.decision.bulk import stage_all
from app.models import Decision, Photo


def _add_photo(sess: Session, h: str) -> None:
    sess.add(Photo(hash=h, source_path=f"/data/photos/incoming/{h}.CR3"))


def test_stage_all_marks_undecided_photos_no(tmp_db: Session) -> None:
    for h in ("a", "b", "c"):
        _add_photo(tmp_db, h)
    tmp_db.flush()

    staged = stage_all(tmp_db, "no")

    assert staged == 3
    for h in ("a", "b", "c"):
        dec = tmp_db.get(Decision, h)
        assert dec is not None and dec.selected == "no"


def test_stage_all_updates_pending_decision(tmp_db: Session) -> None:
    _add_photo(tmp_db, "a")
    tmp_db.add(Decision(photo_hash="a", selected="yes", applied=0))
    tmp_db.flush()

    staged = stage_all(tmp_db, "no")

    assert staged == 1
    assert tmp_db.get(Decision, "a").selected == "no"


def test_stage_all_skips_applied_decisions(tmp_db: Session) -> None:
    _add_photo(tmp_db, "a")
    tmp_db.add(Decision(photo_hash="a", selected="yes", applied=1))
    tmp_db.flush()

    staged = stage_all(tmp_db, "no")

    assert staged == 0
    assert tmp_db.get(Decision, "a").selected == "yes"


def test_stage_all_returns_zero_for_empty_batch(tmp_db: Session) -> None:
    assert stage_all(tmp_db, "no") == 0


def test_stage_all_rejects_invalid_selected(tmp_db: Session) -> None:
    with pytest.raises(ValueError):
        stage_all(tmp_db, "maybe")
