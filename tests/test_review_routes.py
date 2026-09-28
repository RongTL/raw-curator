"""Review API (queue / cluster / photo / decide) against a seeded temp DB."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import cluster, decide, photo, queue
from app.db import make_engine
from app.models import Base, Cluster, ClusterMember, Decision, Face, Photo

Scope = Callable[[], object]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
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
        sess.add_all(
            [
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
                Photo(
                    hash="bbb",
                    source_path="/x/B.CR3",
                    file_kind="raw",
                    captured_at=datetime(2026, 1, 1, 9),
                    technical_score=0.5,
                    cluster_id=c.id,
                ),
                Photo(hash="ccc", source_path="/x/C.jpg", file_kind="jpeg"),
            ]
        )
        sess.flush()  # FK targets first: SQLite enforces foreign_keys=ON
        sess.add_all(
            [
                ClusterMember(cluster_id=c.id, photo_hash="aaa", rank=0),
                ClusterMember(cluster_id=c.id, photo_hash="bbb", rank=1),
                Decision(photo_hash="aaa", selected="yes", stars=4, favorite=1),
                Face(photo_hash="aaa", bbox_x=1, bbox_y=2, bbox_w=3, bbox_h=4, det_score=0.9),
            ]
        )

    app = FastAPI()
    app.include_router(queue.router, prefix="/api/queue")
    app.include_router(photo.router, prefix="/api/photo")
    app.include_router(cluster.router, prefix="/api/cluster")
    app.include_router(decide.router, prefix="/api/decide")
    try:
        yield TestClient(app)
    finally:
        engine.dispose()


def test_queue_sorted_by_technical_score_with_decisions(client: TestClient) -> None:
    items = client.get("/api/queue/").json()
    assert [i["hash"] for i in items] == ["aaa", "bbb", "ccc"]
    assert items[0]["decision"] == {
        "selected": "yes",
        "stars": 4,
        "favorite": True,
        "applied": False,
        "action": "none",
        "note": None,
    }
    assert items[1]["decision"] is None
    assert items[0]["filename"] == "A.CR3"


def test_queue_sorted_by_capture_time(client: TestClient) -> None:
    items = client.get("/api/queue/?sort=captured").json()
    assert [i["hash"] for i in items] == ["ccc", "bbb", "aaa"]  # None sorts first


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
    assert client.post("/api/decide/", json={"photo_hash": "bbb", "selected": "no"}).json() == {
        "ok": True
    }
    pending = {d["photo_hash"]: d for d in client.get("/api/decide/pending").json()}
    assert pending["bbb"]["selected"] == "no"
    assert pending["aaa"]["stars"] == 4
    assert (
        client.post("/api/decide/", json={"photo_hash": "zzz", "selected": "yes"}).status_code
        == 404
    )


def test_pending_excludes_undecided_decisions(client: TestClient) -> None:
    # A row that exists but is undecided (e.g. marked then reset) is not pending.
    client.post("/api/decide/", json={"photo_hash": "bbb", "selected": "yes"})
    client.post("/api/decide/", json={"photo_hash": "bbb", "selected": "undecided"})
    hashes = {d["photo_hash"] for d in client.get("/api/decide/pending").json()}
    assert "bbb" not in hashes  # undecided must not count as pending
    assert "aaa" in hashes  # the yes decision still counts
