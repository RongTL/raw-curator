"""Unit tests for stage_all bulk-decision helper."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.decision.bulk import stage_all, stage_cluster
from app.models import Cluster, Decision, Photo


def _add_photo(sess: Session, h: str) -> None:
    sess.add(Photo(hash=h, source_path=f"/data/photos/incoming/{h}.CR3"))


def _add_clustered(sess: Session, h: str, cluster_id: int, *, recommended: bool = False) -> None:
    if sess.get(Cluster, cluster_id) is None:
        sess.add(Cluster(id=cluster_id, kind="burst"))
        sess.flush()
    sess.add(
        Photo(
            hash=h,
            source_path=f"/data/photos/incoming/{h}.CR3",
            cluster_id=cluster_id,
            is_recommended=1 if recommended else 0,
        )
    )


def test_stage_all_marks_undecided_photos_no(tmp_db: Session) -> None:
    for h in ("a", "b", "c"):
        _add_photo(tmp_db, h)
    tmp_db.flush()

    staged = stage_all(tmp_db, "no")
    tmp_db.flush()  # autoflush=False: flush so the get()s below see the new rows

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


def test_stage_all_mixed_batch_counts_only_non_applied(tmp_db: Session) -> None:
    _add_photo(tmp_db, "missing")  # no decision row
    _add_photo(tmp_db, "pending")
    tmp_db.add(Decision(photo_hash="pending", selected="yes", applied=0))
    _add_photo(tmp_db, "applied")
    tmp_db.add(Decision(photo_hash="applied", selected="yes", applied=1))
    tmp_db.flush()

    staged = stage_all(tmp_db, "no")
    tmp_db.flush()  # autoflush=False: flush so the get()s below see the new row

    assert staged == 2  # missing (created) + pending (updated); applied skipped
    assert tmp_db.get(Decision, "missing").selected == "no"
    assert tmp_db.get(Decision, "pending").selected == "no"
    assert tmp_db.get(Decision, "applied").selected == "yes"


def test_stage_cluster_keep_recommended_keeps_best_rejects_rest(tmp_db: Session) -> None:
    _add_clustered(tmp_db, "rec", 5, recommended=True)
    _add_clustered(tmp_db, "dup1", 5)
    _add_clustered(tmp_db, "dup2", 5)
    _add_clustered(tmp_db, "other", 6, recommended=True)  # a different cluster
    tmp_db.flush()

    staged = stage_cluster(tmp_db, 5, "keep_recommended")
    tmp_db.flush()

    assert staged == 3
    assert tmp_db.get(Decision, "rec").selected == "yes"
    assert tmp_db.get(Decision, "dup1").selected == "no"
    assert tmp_db.get(Decision, "dup2").selected == "no"
    assert tmp_db.get(Decision, "other") is None  # other cluster untouched


def test_stage_cluster_reject_all_then_keep_all(tmp_db: Session) -> None:
    _add_clustered(tmp_db, "a", 7, recommended=True)
    _add_clustered(tmp_db, "b", 7)
    tmp_db.flush()

    assert stage_cluster(tmp_db, 7, "reject_all") == 2
    tmp_db.flush()
    assert tmp_db.get(Decision, "a").selected == "no"
    assert tmp_db.get(Decision, "b").selected == "no"

    assert stage_cluster(tmp_db, 7, "keep_all") == 2
    assert tmp_db.get(Decision, "a").selected == "yes"


def test_stage_cluster_skips_applied(tmp_db: Session) -> None:
    _add_clustered(tmp_db, "a", 8, recommended=True)
    tmp_db.add(Decision(photo_hash="a", selected="yes", applied=1))
    tmp_db.flush()

    assert stage_cluster(tmp_db, 8, "reject_all") == 0
    assert tmp_db.get(Decision, "a").selected == "yes"


def test_stage_cluster_rejects_invalid_mode(tmp_db: Session) -> None:
    with pytest.raises(ValueError):
        stage_cluster(tmp_db, 1, "bogus")


def test_stage_cluster_rejects_empty_cluster(tmp_db: Session) -> None:
    _add_clustered(tmp_db, "a", 9, recommended=True)
    tmp_db.flush()
    with pytest.raises(ValueError):
        stage_cluster(tmp_db, 999, "reject_all")
