"""Pipeline + system routes, with the real runner (python -c) and mocked DB."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import pipeline, system
from app.orchestrator.runner import JobRunner
from app.orchestrator.stages import StageDef

PY = (sys.executable, "-c")


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> Iterator[TestClient]:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    fake_stages = (
        StageDef("ingest", "Ingest", ("print('ok')",), 1),
        StageDef("slow", "Slow", ("import time; time.sleep(30)",), 1),
        StageDef("fail", "Fail", ("import sys; sys.exit(2)",), 2),
    )
    monkeypatch.setattr(pipeline, "STAGES", fake_stages)
    monkeypatch.setattr(pipeline, "STAGE_BY_NAME", {s.name: s for s in fake_stages})
    monkeypatch.setattr(pipeline, "get_runner", lambda: runner)

    from app.orchestrator.autorun import AutoRun

    auto = AutoRun(runner)
    monkeypatch.setattr(pipeline, "get_autorun", lambda: auto)
    monkeypatch.setattr(pipeline.progress, "stage_progress", lambda s: (0, 0))
    monkeypatch.setattr(
        pipeline.progress,
        "batch_summary",
        lambda: {"photos": 0, "decided": 0, "incoming_files": 0},
    )

    app = FastAPI()
    app.include_router(pipeline.router, prefix="/api/pipeline")
    app.include_router(system.router, prefix="/api/system")
    # Context manager keeps a single portal event loop across requests, so the
    # runner's subprocess + reader task stay attached to a live loop.
    with TestClient(app) as test_client:
        yield test_client


def wait_for_state(client: TestClient, stage: str, state: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get("/api/pipeline/status").json()
        entry = next(s for s in payload["stages"] if s["name"] == stage)
        if entry["state"] == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"stage {stage} never reached {state}")


def test_unknown_stage_is_422(client: TestClient) -> None:
    assert client.post("/api/pipeline/run/bogus").status_code == 422


def test_run_stage_and_status(client: TestClient) -> None:
    assert client.post("/api/pipeline/run/ingest").json() == {"started": "ingest"}
    wait_for_state(client, "ingest", "done")
    payload = client.get("/api/pipeline/status").json()
    entry = next(s for s in payload["stages"] if s["name"] == "ingest")
    assert entry["exit_code"] == 0
    assert entry["started_at"] is not None


def test_busy_returns_409(client: TestClient) -> None:
    client.post("/api/pipeline/run/slow")
    assert client.post("/api/pipeline/run/ingest").status_code == 409
    assert client.post("/api/pipeline/cancel").json() == {"cancelled": True}
    wait_for_state(client, "slow", "cancelled")


def test_failed_stage_reports_exit_code(client: TestClient) -> None:
    client.post("/api/pipeline/run/fail")
    wait_for_state(client, "fail", "failed")
    entry = next(
        s for s in client.get("/api/pipeline/status").json()["stages"] if s["name"] == "fail"
    )
    assert entry["exit_code"] == 2


def test_logs_incremental(client: TestClient) -> None:
    client.post("/api/pipeline/run/ingest")
    wait_for_state(client, "ingest", "done")
    first = client.get("/api/pipeline/logs?after=0").json()
    assert first["lines"] == ["ok"]
    again = client.get(f"/api/pipeline/logs?after={first['next']}").json()
    assert again["lines"] == []


def test_auto_leg_validation(client: TestClient) -> None:
    assert client.post("/api/pipeline/auto/9").status_code == 422


def test_reset_requires_token(client: TestClient, monkeypatch) -> None:
    called = {"n": 0}

    async def fake_reset() -> None:
        called["n"] += 1

    monkeypatch.setattr(pipeline, "reset_session", fake_reset)
    assert client.post("/api/pipeline/reset", json={"confirm": "nope"}).status_code == 422
    assert called["n"] == 0
    assert client.post("/api/pipeline/reset", json={"confirm": "RESET"}).json() == {"reset": True}
    assert called["n"] == 1


def test_logs_after_beyond_end_returns_empty(client: TestClient) -> None:
    client.post("/api/pipeline/run/ingest")
    wait_for_state(client, "ingest", "done")
    payload = client.get("/api/pipeline/logs?after=999999").json()
    assert payload["lines"] == []


def test_spawn_failure_returns_500_with_detail(client: TestClient, monkeypatch) -> None:
    from app.api.routes import pipeline as pl

    async def boom(stage: str, cli_args: tuple[str, ...]) -> None:
        raise RuntimeError("failed to spawn stage 'ingest': boom")

    monkeypatch.setattr(pl.get_runner(), "start", boom)
    resp = client.post("/api/pipeline/run/ingest")
    assert resp.status_code == 500
    assert "failed to spawn" in resp.json()["detail"]


def test_reset_in_progress_blocks_run(client: TestClient, monkeypatch) -> None:
    import asyncio as _a

    from app.api.routes import pipeline as pl

    release = _a.Event()

    async def slow_reset() -> None:
        await release.wait()

    monkeypatch.setattr(pl, "reset_session", slow_reset)
    import threading

    codes: dict[str, int] = {}

    def do_reset() -> None:
        codes["reset"] = client.post(
            "/api/pipeline/reset", json={"confirm": "RESET"}
        ).status_code

    t = threading.Thread(target=do_reset)
    t.start()
    deadline = time.time() + 5
    while not getattr(pl, "_resetting", False) and time.time() < deadline:
        time.sleep(0.02)
    assert pl._resetting is True
    assert client.post("/api/pipeline/run/ingest").status_code == 409
    assert client.post("/api/pipeline/reset", json={"confirm": "RESET"}).status_code == 409
    release.set()
    t.join(timeout=10)
    assert codes["reset"] == 200


def test_manual_run_blocked_while_autorun_active(client: TestClient, monkeypatch) -> None:
    # point autorun at the slow stage so the leg stays active
    monkeypatch.setattr(
        "app.orchestrator.autorun.leg_stages",
        lambda leg: (StageDef("slow", "Slow", ("import time; time.sleep(30)",), 1),),
    )
    client.post("/api/pipeline/auto/1")
    wait_for_state(client, "slow", "running")
    assert client.post("/api/pipeline/run/ingest").status_code == 409
    assert client.post("/api/pipeline/reset", json={"confirm": "RESET"}).status_code == 409
    client.post("/api/pipeline/cancel")
    wait_for_state(client, "slow", "cancelled")


def test_autorun_blocked_while_autorun_active(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.orchestrator.autorun.leg_stages",
        lambda leg: (StageDef("slow", "Slow", ("import time; time.sleep(30)",), 1),),
    )
    client.post("/api/pipeline/auto/1")
    wait_for_state(client, "slow", "running")
    assert client.post("/api/pipeline/auto/1").status_code == 409
    client.post("/api/pipeline/cancel")
    wait_for_state(client, "slow", "cancelled")


def test_system_stats_shape(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    from app.monitor import stats as stats_mod
    from app.monitor.stats import SystemMonitor

    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    monkeypatch.setattr(system, "get_monitor", lambda: SystemMonitor(mounts={"photos": tmp_path}))
    payload = client.get("/api/system/stats").json()
    assert payload["current"]["gpu"] is None
    assert "warnings" in payload
