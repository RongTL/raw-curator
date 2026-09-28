from __future__ import annotations

import pytest

from app.decision.bulk import stage_all, stage_cluster
from app.models import Cluster, Decision, Photo


def _photo(sess, h, cluster_id=None, recommended=0):
    sess.add(
        Photo(
            hash=h,
            source_path=f"/data/photos/incoming/{h}.CR3",
            file_kind="raw",
            cluster_id=cluster_id,
            is_recommended=recommended,
        )
    )


def test_stage_all_sets_export_choice(tmp_db) -> None:
    _photo(tmp_db, "a" * 32)
    _photo(tmp_db, "b" * 32)
    tmp_db.flush()
    assert stage_all(tmp_db, export_choice="enhanced") == 2
    tmp_db.flush()  # autoflush=False: flush so the get() below sees the new rows
    assert tmp_db.get(Decision, "a" * 32).export_choice == "enhanced"


def test_stage_all_sets_keep_raw_only(tmp_db) -> None:
    _photo(tmp_db, "a" * 32)
    tmp_db.flush()
    stage_all(tmp_db, keep_raw=False)
    tmp_db.flush()  # autoflush=False: flush so the get() below sees the new row
    row = tmp_db.get(Decision, "a" * 32)
    assert bool(row.keep_raw) is False
    assert row.export_choice == "undecided"  # untouched


def test_stage_all_skips_applied(tmp_db) -> None:
    _photo(tmp_db, "a" * 32)
    tmp_db.flush()
    tmp_db.add(Decision(photo_hash="a" * 32, export_choice="original", applied=1))
    tmp_db.flush()
    assert stage_all(tmp_db, export_choice="enhanced") == 0
    assert tmp_db.get(Decision, "a" * 32).export_choice == "original"


def test_stage_all_rejects_bad_choice(tmp_db) -> None:
    with pytest.raises(ValueError):
        stage_all(tmp_db, export_choice="bogus")


def test_stage_cluster_keep_recommended(tmp_db) -> None:
    tmp_db.add(Cluster(id=1, kind="burst"))
    tmp_db.flush()
    _photo(tmp_db, "a" * 32, cluster_id=1, recommended=1)
    _photo(tmp_db, "b" * 32, cluster_id=1, recommended=0)
    tmp_db.flush()
    assert stage_cluster(tmp_db, 1, "keep_recommended") == 2
    tmp_db.flush()  # autoflush=False: flush so the get()s below see the new rows
    assert tmp_db.get(Decision, "a" * 32).export_choice == "enhanced"
    assert tmp_db.get(Decision, "b" * 32).export_choice == "discard"
