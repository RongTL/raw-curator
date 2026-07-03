"""Auto-run leg chaining: run stages in order, halt on first failure."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.orchestrator.autorun import AutoRun
from app.orchestrator.runner import JobBusyError, JobRunner, JobStatus
from app.orchestrator.stages import StageDef

PY = (sys.executable, "-c")


def fake_stages(*snippets: tuple[str, str]) -> tuple[StageDef, ...]:
    return tuple(StageDef(name, name.title(), (code,), 1) for name, code in snippets)


async def wait_idle(auto: AutoRun) -> None:
    async with asyncio.timeout(15):
        while auto.active_leg is not None:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_leg_runs_all_stages_in_order(tmp_path: Path, monkeypatch) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    stages = fake_stages(("a", "print('a')"), ("b", "print('b')"))
    monkeypatch.setattr("app.orchestrator.autorun.leg_stages", lambda leg: stages)
    auto = AutoRun(runner)
    auto.start(1)
    await wait_idle(auto)
    assert runner.record_for("a").status is JobStatus.DONE
    assert runner.record_for("b").status is JobStatus.DONE


@pytest.mark.asyncio
async def test_leg_halts_on_failure(tmp_path: Path, monkeypatch) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    stages = fake_stages(
        ("a", "print('a')"),
        ("b", "import sys; sys.exit(1)"),
        ("c", "print('c')"),
    )
    monkeypatch.setattr("app.orchestrator.autorun.leg_stages", lambda leg: stages)
    auto = AutoRun(runner)
    auto.start(1)
    await wait_idle(auto)
    assert runner.record_for("a").status is JobStatus.DONE
    assert runner.record_for("b").status is JobStatus.FAILED
    assert runner.record_for("c") is None


@pytest.mark.asyncio
async def test_start_unknown_leg_raises(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    auto = AutoRun(runner)
    with pytest.raises(ValueError):
        auto.start(9)


@pytest.mark.asyncio
async def test_start_while_running_raises_busy(tmp_path: Path, monkeypatch) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    stages = fake_stages(("slow", "import time; time.sleep(30)"))
    monkeypatch.setattr("app.orchestrator.autorun.leg_stages", lambda leg: stages)
    auto = AutoRun(runner)
    auto.start(1)
    with pytest.raises(JobBusyError):
        auto.start(1)
    await auto.cancel()
    assert auto.active_leg is None


@pytest.mark.asyncio
async def test_cancel_mid_chain_stops_current_and_skips_next(
    tmp_path: Path, monkeypatch
) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    stages = fake_stages(
        ("slow", "import time; time.sleep(30)"),
        ("after", "print('never')"),
    )
    monkeypatch.setattr("app.orchestrator.autorun.leg_stages", lambda leg: stages)
    auto = AutoRun(runner)
    auto.start(1)
    async with asyncio.timeout(15):
        while runner.running_stage != "slow":
            await asyncio.sleep(0.02)
        await auto.cancel()
    assert auto.active_leg is None
    assert runner.record_for("slow").status is JobStatus.CANCELLED
    assert runner.record_for("after") is None
    # the slot must be reusable afterwards
    await runner.start("again", ("print('ok')",))
    assert (await runner.wait()).status is JobStatus.DONE
