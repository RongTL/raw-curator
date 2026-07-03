"""JobRunner state machine tests (no raw-curator binary needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.orchestrator.runner import JobBusyError, JobRunner, JobStatus

PY = (sys.executable, "-c")


@pytest.mark.asyncio
async def test_successful_job_is_done_with_logs(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("ingest", ("print('hello'); print('world')",))
    record = await runner.wait()
    assert record.status is JobStatus.DONE
    assert record.exit_code == 0
    next_offset, lines = runner.log_lines(0)
    assert lines == ["hello", "world"]
    assert next_offset == 2
    assert record.log_path is not None
    assert Path(record.log_path).read_text() == "hello\nworld\n"


@pytest.mark.asyncio
async def test_failing_job_is_failed(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("score", ("import sys; sys.exit(3)",))
    record = await runner.wait()
    assert record.status is JobStatus.FAILED
    assert record.exit_code == 3


@pytest.mark.asyncio
async def test_second_start_while_running_raises_busy(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("ingest", ("import time; time.sleep(30)",))
    with pytest.raises(JobBusyError):
        await runner.start("filter", ("print('nope')",))
    await runner.cancel()
    record = await runner.wait()
    assert record.status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_when_idle_returns_false(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    assert await runner.cancel() is False


@pytest.mark.asyncio
async def test_log_offsets_are_incremental(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("ingest", ("print('a'); print('b'); print('c')",))
    await runner.wait()
    offset, first = runner.log_lines(0)
    assert first == ["a", "b", "c"]
    offset2, rest = runner.log_lines(offset)
    assert rest == []
    assert offset2 == offset


@pytest.mark.asyncio
async def test_record_for_and_history_survive_next_run(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("ingest", ("print('one')",))
    await runner.wait()
    await runner.start("filter", ("print('two')",))
    await runner.wait()
    assert runner.record_for("ingest") is not None
    assert runner.record_for("ingest").status is JobStatus.DONE
    assert runner.record_for("filter").status is JobStatus.DONE
    assert runner.run_seq == 2


@pytest.mark.asyncio
async def test_carriage_return_progress_lines_reach_tail(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    code = "import sys\nfor i in range(5): sys.stdout.write(f'prog {i}\\r')\nsys.stdout.write('end\\n')"
    await runner.start("ingest", (code,))
    record = await runner.wait()
    assert record.status is JobStatus.DONE
    _, lines = runner.log_lines(0)
    assert "prog 0" in lines and "end" in lines


@pytest.mark.asyncio
async def test_giant_unterminated_line_does_not_brick_runner(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    code = "import sys; sys.stdout.write('x' * 300000)"
    await runner.start("ingest", (code,))
    record = await runner.wait()
    assert record.status is JobStatus.DONE
    assert runner.running_stage is None


@pytest.mark.asyncio
async def test_sigterm_ignoring_child_is_killed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.orchestrator.runner.SIGKILL_GRACE_SECONDS", 0.3)
    runner = JobRunner(log_dir=tmp_path, command=PY)
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)"
    await runner.start("enhance", (code,))
    import asyncio as _a
    await _a.sleep(0.3)  # let the child install its handler
    assert await runner.cancel() is True
    record = await runner.wait()
    assert record.status is JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_tail_eviction_keeps_global_offsets(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    code = "print('\\n'.join(str(i) for i in range(2100)))"
    await runner.start("ingest", (code,))
    await runner.wait()
    next_offset, lines = runner.log_lines(0)
    assert next_offset == 2100
    assert len(lines) == 2000
    assert lines[0] == "100"


@pytest.mark.asyncio
async def test_drain_timeout_forces_cancelled_and_next_run_is_clean(
    tmp_path: Path, monkeypatch
) -> None:
    import asyncio as _a

    monkeypatch.setattr("app.orchestrator.runner.SIGKILL_GRACE_SECONDS", 0.2)
    monkeypatch.setattr("app.orchestrator.runner.READER_DRAIN_TIMEOUT_SECONDS", 0.2)
    runner = JobRunner(log_dir=tmp_path, command=PY)
    # Child spawns a grandchild in a NEW session that inherits stdout and outlives the kill.
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'],\n"
        "                 start_new_session=True)\n"
        "time.sleep(60)\n"
    )
    await runner.start("ingest", (code,))
    await _a.sleep(0.3)
    assert await runner.cancel() is True
    record = await runner.wait()
    assert record.status is JobStatus.CANCELLED
    assert runner.running_stage is None
    # Next run must be clean despite the orphaned reader still draining.
    await runner.start("filter", ("print('fresh')",))
    record2 = await runner.wait()
    assert record2.status is JobStatus.DONE
    _, lines = runner.log_lines(0)
    assert lines == ["fresh"]


@pytest.mark.asyncio
async def test_clear_history_while_running_raises(tmp_path: Path) -> None:
    runner = JobRunner(log_dir=tmp_path, command=PY)
    await runner.start("ingest", ("import time; time.sleep(30)",))
    with pytest.raises(RuntimeError):
        runner.clear_history()
    await runner.cancel()
    await runner.wait()
