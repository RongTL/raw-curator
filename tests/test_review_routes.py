"""Review API (queue / cluster / photo / decide) against a seeded temp DB."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import cluster, decide, photo, queue
from app.db import make_engine
from app.models import Base, Cluster, ClusterMember, Decision, Face, Photo

Scope = Callable[[], AbstractContextManager[Session]]


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Scope]:
    """A seeded temp DB; yields the session scope so tests can add rows."""
    engine = make_engine(f"sqlite:///{tmp_path / 'review.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True)

    @contextmanager
    def scope() -> Iterator[Session]:
        sess = factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    for mod in (cluster, decide, photo, queue):
        monkeypatch.setattr(mod, "session_scope", scope)

    with scope() as sess:
        c = Cluster(kind="burst", size=2)
        sess.add(c)
        sess.flush()
        # Inserted in reverse so the table's natural order (ccc, bbb, aaa) matches
        # none of the sort options; a sort that silently falls through shows up.
        sess.add_all(
            [
                Photo(hash="ccc", source_path="/x/C.jpg", file_kind="jpeg"),
                Photo(
                    hash="bbb",
                    source_path="/x/B.CR3",
                    file_kind="raw",
                    captured_at=datetime(2026, 1, 1, 9),
                    technical_score=0.5,
                    cluster_id=c.id,
                ),
                Photo(
                    hash="aaa",
                    source_path="/x/A.CR3",
                    file_kind="raw",
                    captured_at=datetime(2026, 1, 1, 10),
                    technical_score=0.9,
                    aesthetic_score=8.0,
                    cluster_id=c.id,
                    is_recommended=1,
                ),
            ]
        )
        sess.flush()  # FK targets first: SQLite enforces foreign_keys=ON
        sess.add_all(
            [
                ClusterMember(cluster_id=c.id, photo_hash="aaa", rank=0),
                ClusterMember(cluster_id=c.id, photo_hash="bbb", rank=1),
                Decision(
                    photo_hash="aaa", selected="yes", export_choice="enhanced", stars=4, favorite=1
                ),
                Face(photo_hash="aaa", bbox_x=1, bbox_y=2, bbox_w=3, bbox_h=4, det_score=0.9),
            ]
        )

    try:
        yield scope
    finally:
        engine.dispose()


@pytest.fixture
def client(db: Scope) -> TestClient:
    app = FastAPI()
    app.include_router(queue.router, prefix="/api/queue")
    app.include_router(photo.router, prefix="/api/photo")
    app.include_router(cluster.router, prefix="/api/cluster")
    app.include_router(decide.router, prefix="/api/decide")
    return TestClient(app)


def _hashes(client: TestClient, query: str = "") -> list[str]:
    r = client.get(f"/api/queue/{query}")
    assert r.status_code == 200, r.text
    return [i["hash"] for i in r.json()]


def test_queue_default_sort_is_capture_time(client: TestClient) -> None:
    assert _hashes(client) == _hashes(client, "?sort=captured") == ["bbb", "aaa", "ccc"]


def test_queue_sorted_by_technical_score_with_decisions(client: TestClient) -> None:
    items = client.get("/api/queue/?sort=score").json()
    assert [i["hash"] for i in items] == ["aaa", "bbb", "ccc"]  # None sorts last
    assert items[0]["decision"] == {
        "export_choice": "enhanced",
        "keep_raw": True,
        "stars": 4,
        "favorite": True,
        "applied": False,
        "note": None,
    }
    assert items[1]["decision"] is None
    assert items[0]["filename"] == "A.CR3"


def test_queue_sorted_by_capture_time_puts_undated_last(client: TestClient) -> None:
    assert _hashes(client, "?sort=captured") == ["bbb", "aaa", "ccc"]


def test_queue_capture_time_ties_break_on_source_path(client: TestClient, db: Scope) -> None:
    # Same second as "aaa" but a path that sorts before it: bursts keep shooting order.
    with db() as sess:
        sess.add(
            Photo(
                hash="ddd",
                source_path="/x/0.CR3",
                file_kind="raw",
                captured_at=datetime(2026, 1, 1, 10),
            )
        )
    assert _hashes(client, "?sort=captured") == ["bbb", "ddd", "aaa", "ccc"]


def test_queue_sorted_by_filename_uses_full_source_path(client: TestClient, db: Scope) -> None:
    # "0.CR3" would sort first by bare filename; by full path its subfolder keeps it
    # grouped after the top-level files, matching the ingest walk order.
    with db() as sess:
        sess.add(Photo(hash="ddd", source_path="/x/sub/0.CR3", file_kind="raw"))
    assert _hashes(client, "?sort=filename") == ["aaa", "bbb", "ccc", "ddd"]


def test_queue_rejects_unknown_sort(client: TestClient) -> None:
    assert client.get("/api/queue/?sort=bogus").status_code == 400


def test_clusters_list_members_by_rank_plus_unclustered_bucket(client: TestClient) -> None:
    out = client.get("/api/cluster/").json()
    assert [c["kind"] for c in out] == ["burst", "unclustered"]
    burst = out[0]
    assert [p["hash"] for p in burst["photos"]] == ["aaa", "bbb"]
    assert [p["rank"] for p in burst["photos"]] == [0, 1]
    assert burst["photos"][0]["is_recommended"] is True
    assert [p["hash"] for p in out[1]["photos"]] == ["ccc"]
    assert out[1]["id"] == -1


def test_cluster_detail_and_404(client: TestClient) -> None:
    out = client.get("/api/cluster/1").json()
    assert out["kind"] == "burst"
    assert [m["hash"] for m in out["members"]] == ["aaa", "bbb"]
    assert client.get("/api/cluster/999").status_code == 404


def test_photo_detail_and_404(client: TestClient) -> None:
    out = client.get("/api/photo/aaa").json()
    assert out["faces"] == [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9}]
    assert out["decision"]["stars"] == 4
    assert out["quality_report"] is None
    assert out["cluster_id"] == 1
    assert client.get("/api/photo/nope").status_code == 404


def test_decide_stages_and_lists_pending(client: TestClient) -> None:
    assert client.post(
        "/api/decide/", json={"photo_hash": "bbb", "export_choice": "original"}
    ).json() == {"ok": True}
    pending = {d["photo_hash"]: d for d in client.get("/api/decide/pending").json()}
    assert pending["bbb"]["export_choice"] == "original"
    assert pending["aaa"]["stars"] == 4
    assert (
        client.post(
            "/api/decide/", json={"photo_hash": "zzz", "export_choice": "enhanced"}
        ).status_code
        == 404
    )


def test_pending_excludes_undecided_decisions(client: TestClient) -> None:
    # A row that exists but is undecided (e.g. marked then reset) is not pending.
    client.post("/api/decide/", json={"photo_hash": "bbb", "export_choice": "enhanced"})
    client.post("/api/decide/", json={"photo_hash": "bbb", "export_choice": "undecided"})
    hashes = {d["photo_hash"] for d in client.get("/api/decide/pending").json()}
    assert "bbb" not in hashes  # undecided must not count as pending
    assert "aaa" in hashes  # the enhanced decision still counts


def test_decide_sets_export_choice_and_keep_raw(client: TestClient) -> None:
    r = client.post(
        "/api/decide/", json={"photo_hash": "bbb", "export_choice": "original", "keep_raw": False}
    )
    assert r.status_code == 200
    # Verify via /pending (this task) — the photo serializer does not emit the
    # new decision fields until Task 13.
    pending = {d["photo_hash"]: d for d in client.get("/api/decide/pending").json()}
    assert pending["bbb"]["export_choice"] == "original"
    assert pending["bbb"]["keep_raw"] is False


def test_decide_rejects_unknown_export_choice(client: TestClient) -> None:
    r = client.post("/api/decide/", json={"photo_hash": "bbb", "export_choice": "bogus"})
    assert r.status_code == 400


def test_decide_all_enhanced(client: TestClient) -> None:
    r = client.post("/api/decide/all", json={"export_choice": "enhanced"})
    assert r.json()["staged"] >= 1


def test_render_url_before_after() -> None:
    from app.api.routes._urls import render_url

    assert render_url("abc", "before") == "/cache/enhanced/abc.before.jpg"
    assert render_url("abc", "after") == "/cache/enhanced/abc.after.jpg"
