# Control Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One web interface (served by `make serve`) that runs every pipeline stage, shows live progress/logs, monitors CPU/RAM/GPU/VRAM/disk in real time, and replaces all per-stage `make` commands.

**Architecture:** A new `app/orchestrator/` package runs each stage as a `raw-curator <stage>` subprocess inside the long-running `ui` container (single job slot, two-leg auto-run around human review). A new `app/monitor/` package samples psutil + NVML on a 1 s loop. New FastAPI routes expose status/run/cancel/logs/reset/stats. The frontend is rewritten as small ES modules (CDN React + htm, no build step) implementing the approved "Pipeline Timeline + Context Panel" layout; the existing review components are ported, not rewritten.

**Tech Stack:** Python 3.12, FastAPI, asyncio subprocesses, psutil, nvidia-ml-py, SQLAlchemy 2.0, pytest + pytest-asyncio; frontend: React 18 via esm.sh, htm, Tailwind CDN.

**Spec:** `docs/superpowers/specs/2026-07-03-control-center-design.md`

**Deviations from spec (approved rationale inline):**
- Log endpoint is `GET /api/pipeline/logs?after=N` (not `/logs/{stage}`): the in-memory tail belongs to the latest run; the payload carries `stage` + `run_seq` so the client resets correctly.
- The review "Submit & continue" button starts auto-run leg 2 (whose first stage is the `submit` CLI job) instead of calling the legacy `POST /api/submit` route; the legacy route stays untouched for compatibility.

**Environment notes for the executor:**
- This machine (macOS) has **no podman and no Python tooling installed**. Task 1 creates `.venv-dev` for local lint/type/test verification. Anything GPU/container-bound (`make test`, `make image`, `make serve`) is final verification on the GPU host by the user.
- Local venv is Python 3.14 while the project targets 3.12 — fine for these pure-python checks; authoritative runs happen in-container.
- Run all commands from the repo root: `/Users/ted_rong/Private/desktop/raw-curator`.
- Branch `feat/control-center` already exists and is checked out; spec + onnxruntime fix are already committed.

---

### Task 1: Dev tooling + new runtime dependencies

**Files:**
- Modify: `pyproject.toml` (dependencies block, after line 47 `codeformer-pip = "^0.0.4"`)
- Create: `.venv-dev/` (do NOT commit)
- Modify: `.gitignore`

- [ ] **Step 1: Create the local verification venv**

```bash
python3 -m venv .venv-dev
.venv-dev/bin/pip install -q ruff mypy pytest pytest-asyncio fastapi httpx \
  pydantic pydantic-settings sqlalchemy sqlite-vec psutil
```

Expected: exits 0.

- [ ] **Step 2: Add runtime deps to pyproject.toml**

In `[tool.poetry.dependencies]`, after the `codeformer-pip` line, add:

```toml
psutil = "^6.0"
nvidia-ml-py = "^12.560"
```

- [ ] **Step 3: Add `.venv-dev/` to .gitignore**

Change the `.venv/` line to:

```
.venv/
.venv-dev/
```

- [ ] **Step 4: Sanity-check existing light tests still pass locally**

```bash
.venv-dev/bin/pytest tests/test_decision_rules.py tests/test_paths.py -q
```

Expected: all pass (these import only light modules).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: add psutil + nvidia-ml-py deps for control center"
```

---

### Task 2: Resource tuning (torch CPU threads + .env.example)

**Files:**
- Modify: `app/workers/gpu_worker.py:26-28` (the `warmup()` function)
- Create: `.env.example`

Audit result (already verified): ingest (`app/ingest/ingest_job.py:92-95`), filter (`app/filters/filter_job.py:65-69`), and export-jpeg (`app/export/jpeg_job.py:100-112`) all use `ProcessPoolExecutor(max_workers=settings.cpu_workers)` and `cpu_workers` defaults to `os.cpu_count()` (= 8 on the R3 3100). The only gap: GPU stages never pin torch's CPU thread pool, so intra-op threading can oversubscribe against DataLoader/PIL work.

- [ ] **Step 1: Pin torch CPU threads in warmup()**

Replace `warmup()` in `app/workers/gpu_worker.py`:

```python
def warmup() -> None:
    # Pin torch's intra-op pool to the SMT thread count so CPU-side
    # preprocessing in GPU stages uses the whole R3 3100 without
    # oversubscribing (torch defaults to physical cores only).
    from app.config import settings

    torch.set_num_threads(max(1, settings.cpu_workers))
    if cuda_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 2: Create `.env.example`**

```bash
# raw-curator — tuned for AMD Ryzen 3 3100 (4C/8T), 24 GB RAM, RTX 2060 6 GB.
# Copy to `.env`; compose injects these into both containers via env_file.

# CPU: process pools default to all logical cores (8). Override only to throttle.
#RAWCURATOR_CPU_WORKERS=8

# GPU batch sizes for 6 GB VRAM. Lower CLIP_BATCH to 4 on CUDA OOM during scoring.
RAWCURATOR_CLIP_BATCH=8
RAWCURATOR_IQA_BATCH=1

# Enhancement VRAM/disk knobs. AI_SCALE 1.0 = native pixels; drop to 0.85/0.7 on OOM.
RAWCURATOR_ENHANCE_AI_SCALE=1.0
# `native` downsamples Real-ESRGAN's x2 back to sensor size (~4x smaller TIFFs).
# Use `200%` only if you have the disk for ~12k x 8k 16-bit TIFFs.
RAWCURATOR_ENHANCE_TARGET_RES=native

# JPEG export
RAWCURATOR_JPEG_QUALITY=92
#RAWCURATOR_JPEG_LONG_EDGE=4000   # cap long edge for sharing; 0 = native
```

- [ ] **Step 3: Verify lint passes on the touched file**

```bash
.venv-dev/bin/ruff check app/workers/gpu_worker.py
```

Expected: no findings.

- [ ] **Step 4: Commit**

```bash
git add app/workers/gpu_worker.py .env.example
git commit -m "perf: pin torch CPU threads to SMT count; add tuned .env.example"
```

---

### Task 3: Stage registry (`app/orchestrator/stages.py`)

**Files:**
- Create: `app/orchestrator/__init__.py` (empty)
- Create: `app/orchestrator/stages.py`
- Test: `tests/test_orchestrator_stages.py`

- [ ] **Step 1: Write the failing test**

```python
"""Stage registry invariants."""

from __future__ import annotations

from app.orchestrator.stages import STAGE_BY_NAME, STAGES, leg_stages


def test_stage_order_matches_pipeline() -> None:
    assert [s.name for s in STAGES] == [
        "ingest", "filter", "score", "cluster", "submit", "enhance", "export-jpeg",
    ]


def test_legs_split_around_review() -> None:
    assert [s.name for s in leg_stages(1)] == ["ingest", "filter", "score", "cluster"]
    assert [s.name for s in leg_stages(2)] == ["submit", "enhance", "export-jpeg"]
    assert leg_stages(3) == ()


def test_lookup_and_cli_args() -> None:
    assert STAGE_BY_NAME["export-jpeg"].cli_args == ("export-jpeg",)
    assert STAGE_BY_NAME["score"].cli_args == ("score",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_stages.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.orchestrator'`

- [ ] **Step 3: Implement**

`app/orchestrator/__init__.py`: empty file.

`app/orchestrator/stages.py`:

```python
"""Stage registry — single source of truth for runnable pipeline stages.

Review is not a stage here: it is a human step handled entirely by the UI
between auto-run legs 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDef:
    name: str  # API identifier and log-file prefix
    title: str  # human label for the UI
    cli_args: tuple[str, ...]  # argv appended to the `raw-curator` command
    leg: int  # auto-run leg this stage belongs to (1 or 2)


STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", "Ingest", ("ingest",), 1),
    StageDef("filter", "Filter", ("filter",), 1),
    StageDef("score", "Score", ("score",), 1),
    StageDef("cluster", "Cluster", ("cluster",), 1),
    StageDef("submit", "Submit", ("submit",), 2),
    StageDef("enhance", "Enhance", ("enhance",), 2),
    StageDef("export-jpeg", "Export JPEG", ("export-jpeg",), 2),
)

STAGE_BY_NAME: dict[str, StageDef] = {s.name: s for s in STAGES}


def leg_stages(leg: int) -> tuple[StageDef, ...]:
    return tuple(s for s in STAGES if s.leg == leg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_stages.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator/ tests/test_orchestrator_stages.py
git commit -m "feat(orchestrator): stage registry with auto-run legs"
```

---

### Task 4: Job runner (`app/orchestrator/runner.py`)

**Files:**
- Create: `app/orchestrator/runner.py`
- Test: `tests/test_orchestrator_runner.py`

The runner is testable without the `raw-curator` binary: `command` is injectable, tests use `(sys.executable, "-c")` and pass python snippets as `cli_args`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_runner.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` on `app.orchestrator.runner`

- [ ] **Step 3: Implement `app/orchestrator/runner.py`**

```python
"""Single-slot async job runner: one pipeline-stage subprocess at a time.

The web server spawns `raw-curator <stage>` as a child process; a crash or
CUDA OOM kills only the child. The stdout+stderr stream is written to a log
file and kept in a bounded in-memory tail for the UI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COMMAND: tuple[str, ...] = ("raw-curator",)
LOG_TAIL_MAX_LINES = 2000
SIGKILL_GRACE_SECONDS = 10.0


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    stage: str
    status: JobStatus
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None
    log_path: str | None = None


class JobBusyError(RuntimeError):
    """Raised when a job is started while another is still running."""

    def __init__(self, running: str) -> None:
        super().__init__(f"a job is already running: {running}")
        self.running = running


class JobRunner:
    def __init__(self, log_dir: Path, command: tuple[str, ...] = DEFAULT_COMMAND) -> None:
        self._log_dir = log_dir
        self._command = command
        self._proc: asyncio.subprocess.Process | None = None
        self._current: JobRecord | None = None
        self._history: dict[str, JobRecord] = {}
        self._tail: deque[str] = deque(maxlen=LOG_TAIL_MAX_LINES)
        self._dropped = 0  # lines evicted from the bounded tail
        self._run_seq = 0
        self._reader: asyncio.Task[None] | None = None
        self._cancelling = False

    # -- state ---------------------------------------------------------------

    @property
    def running_stage(self) -> str | None:
        if self._current is not None and self._current.status is JobStatus.RUNNING:
            return self._current.stage
        return None

    @property
    def run_seq(self) -> int:
        return self._run_seq

    def latest_record(self) -> JobRecord | None:
        return self._current

    def record_for(self, stage: str) -> JobRecord | None:
        if self._current is not None and self._current.stage == stage:
            return self._current
        return self._history.get(stage)

    def log_lines(self, after: int) -> tuple[int, list[str]]:
        """Return (next_offset, lines) with per-run global line numbering."""
        start = max(after - self._dropped, 0)
        lines = list(self._tail)[start:]
        return self._dropped + len(self._tail), lines

    def clear_history(self) -> None:
        """Forget all runs (used by session reset). Must not be called mid-job."""
        self._history.clear()
        self._current = None
        self._tail.clear()
        self._dropped = 0

    # -- lifecycle -------------------------------------------------------------

    async def start(self, stage: str, cli_args: tuple[str, ...]) -> None:
        running = self.running_stage
        if running is not None:
            raise JobBusyError(running)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_seq += 1
        self._tail.clear()
        self._dropped = 0
        self._cancelling = False
        log_path = self._log_dir / f"{stage}-{self._run_seq:03d}.log"
        self._current = JobRecord(
            stage=stage,
            status=JobStatus.RUNNING,
            started_at=time.time(),
            log_path=str(log_path),
        )
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,
                *cli_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self._finish(JobStatus.FAILED, exit_code=None)
            raise RuntimeError(f"failed to spawn stage {stage!r}: {exc}") from exc
        self._reader = asyncio.create_task(self._pump(log_path))

    async def wait(self) -> JobRecord:
        if self._reader is not None:
            await self._reader
        record = self._current
        if record is None:
            raise RuntimeError("wait() called before any job was started")
        return record

    async def cancel(self) -> bool:
        proc = self._proc
        if proc is None or self.running_stage is None:
            return False
        self._cancelling = True
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=SIGKILL_GRACE_SECONDS)
        except TimeoutError:
            proc.kill()
        if self._reader is not None:
            await self._reader
        return True

    # -- internals ---------------------------------------------------------------

    async def _pump(self, log_path: Path) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:  # pragma: no cover - defensive
            return
        with log_path.open("ab") as fh:
            async for raw in proc.stdout:
                fh.write(raw)
                if len(self._tail) == self._tail.maxlen:
                    self._dropped += 1
                self._tail.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
        exit_code = await proc.wait()
        if self._cancelling:
            self._finish(JobStatus.CANCELLED, exit_code)
        elif exit_code == 0:
            self._finish(JobStatus.DONE, exit_code)
        else:
            self._finish(JobStatus.FAILED, exit_code)

    def _finish(self, status: JobStatus, exit_code: int | None) -> None:
        record = self._current
        if record is None:  # pragma: no cover - defensive
            return
        record.status = status
        record.finished_at = time.time()
        record.exit_code = exit_code
        self._history[record.stage] = record
        self._proc = None
        logger.info("stage %s finished: %s (rc=%s)", record.stage, status.value, exit_code)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_runner.py -q`
Expected: 6 passed

- [ ] **Step 5: Lint + typecheck the new module, then commit**

```bash
.venv-dev/bin/ruff check app/orchestrator/ tests/test_orchestrator_runner.py
.venv-dev/bin/mypy app/orchestrator/runner.py
git add app/orchestrator/runner.py tests/test_orchestrator_runner.py
git commit -m "feat(orchestrator): single-slot subprocess job runner with log tail"
```

---

### Task 5: Auto-run legs (`app/orchestrator/autorun.py`)

**Files:**
- Create: `app/orchestrator/autorun.py`
- Test: `tests/test_orchestrator_autorun.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    await asyncio.sleep(0.2)  # let the subprocess spawn
    with pytest.raises(JobBusyError):
        auto.start(1)
    await auto.cancel()
    assert auto.active_leg is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_autorun.py -q`
Expected: FAIL with `ModuleNotFoundError` on `app.orchestrator.autorun`

- [ ] **Step 3: Implement `app/orchestrator/autorun.py`**

```python
"""Chains the stages of an auto-run leg; halts on the first failure.

Leg 1 = ingest -> filter -> score -> cluster (then the human reviews).
Leg 2 = submit -> enhance -> export-jpeg (after "Submit & continue").
"""

from __future__ import annotations

import asyncio
import logging

from app.orchestrator.runner import JobBusyError, JobRunner, JobStatus
from app.orchestrator.stages import StageDef, leg_stages

logger = logging.getLogger(__name__)


class AutoRun:
    def __init__(self, runner: JobRunner) -> None:
        self._runner = runner
        self._task: asyncio.Task[None] | None = None
        self._leg: int | None = None

    @property
    def active_leg(self) -> int | None:
        if self._task is not None and not self._task.done():
            return self._leg
        return None

    def start(self, leg: int) -> None:
        stages = leg_stages(leg)
        if not stages:
            raise ValueError(f"unknown auto-run leg: {leg}")
        if self.active_leg is not None:
            raise JobBusyError(f"auto-run leg {self._leg}")
        running = self._runner.running_stage
        if running is not None:
            raise JobBusyError(running)
        self._leg = leg
        self._task = asyncio.create_task(self._run(stages))

    async def cancel(self) -> None:
        """Stop the chain first (so no next stage starts), then the current job."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._runner.cancel()

    async def _run(self, stages: tuple[StageDef, ...]) -> None:
        for stage in stages:
            await self._runner.start(stage.name, stage.cli_args)
            record = await self._runner.wait()
            if record.status is not JobStatus.DONE:
                logger.warning(
                    "auto-run halted: stage %s ended %s", stage.name, record.status.value
                )
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_autorun.py -q`
Expected: 4 passed

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv-dev/bin/ruff check app/orchestrator/autorun.py tests/test_orchestrator_autorun.py
.venv-dev/bin/mypy app/orchestrator/autorun.py
git add app/orchestrator/autorun.py tests/test_orchestrator_autorun.py
git commit -m "feat(orchestrator): two-leg auto-run chaining with halt-on-failure"
```

---

### Task 6: Progress derivation (`app/orchestrator/progress.py`)

**Files:**
- Create: `app/orchestrator/progress.py`
- Test: `tests/test_orchestrator_progress.py`

Progress comes from read-only DB counts + filesystem counts so no job code changes. Denominators are best-effort display values; `total == 0` means indeterminate.

- [ ] **Step 1: Write the failing tests**

```python
"""Stage progress derived from DB + filesystem state."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models import Decision, Photo
from app.orchestrator import progress


@pytest.fixture
def env(tmp_db, tmp_path: Path, monkeypatch):
    @contextmanager
    def fake_scope():
        yield tmp_db

    fake_settings = SimpleNamespace(photos=tmp_path / "photos", jpeg_subdir="jpeg")
    (tmp_path / "photos" / "incoming").mkdir(parents=True)
    monkeypatch.setattr(progress, "session_scope", fake_scope)
    monkeypatch.setattr(progress, "settings", fake_settings)
    return tmp_db, fake_settings


def test_filter_progress_counts_blur_var(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3", blur_var=10.0))
    sess.add(Photo(hash="b" * 32, source_path="/x/b.cr3"))
    sess.flush()
    assert progress.stage_progress("filter") == (1, 2)


def test_score_progress_counts_technical_score(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3", technical_score=0.7))
    sess.add(Photo(hash="b" * 32, source_path="/x/b.cr3"))
    sess.flush()
    assert progress.stage_progress("score") == (1, 2)


def test_ingest_progress_counts_incoming_files(env) -> None:
    sess, settings = env
    (settings.photos / "incoming" / "a.cr3").write_bytes(b"x")
    (settings.photos / "incoming" / "b.cr3").write_bytes(b"x")
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3"))
    sess.flush()
    assert progress.stage_progress("ingest") == (1, 2)


def test_enhance_progress_counts_tiffs_vs_eligible(env) -> None:
    sess, settings = env
    for i, action in enumerate(["keep_and_enhance", "enhance_only", "none"]):
        h = str(i) * 32
        sess.add(Photo(hash=h, source_path=f"/x/{i}.cr3"))
        sess.add(Decision(photo_hash=h, selected="yes", action=action))
    sess.flush()
    exported = settings.photos / "exported"
    exported.mkdir(parents=True)
    (exported / "0.tif").write_bytes(b"x")
    assert progress.stage_progress("enhance") == (1, 2)


def test_batch_summary_shape(env) -> None:
    sess, _ = env
    sess.add(Photo(hash="a" * 32, source_path="/x/a.cr3"))
    sess.add(Decision(photo_hash="a" * 32, selected="yes"))
    sess.flush()
    summary = progress.batch_summary()
    assert summary == {"photos": 1, "decided": 1, "incoming_files": 0}


def test_unknown_stage_is_indeterminate(env) -> None:
    assert progress.stage_progress("nope") == (0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_progress.py -q`
Expected: FAIL with `ImportError` on `app.orchestrator.progress`

- [ ] **Step 3: Implement `app/orchestrator/progress.py`**

```python
"""Derive per-stage progress and batch summary from DB + filesystem state.

Read-only: keeps pipeline job code untouched. Values are display-only,
best-effort; a `total` of 0 means indeterminate. Directory scans are cheap
relative to the 1-3 s UI poll (local SSD, one batch of photos).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from app.config import settings
from app.db import session_scope
from app.models import Cluster, Decision, Photo

ENHANCE_ACTIONS = ("keep_and_enhance", "enhance_only")
TIFF_SUFFIXES = (".tif", ".tiff")
JPEG_SUFFIXES = (".jpg", ".jpeg")


def _count_files(root: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if suffixes is not None and p.suffix.lower() not in suffixes:
            continue
        total += 1
    return total


def batch_summary() -> dict[str, int]:
    with session_scope() as sess:
        photos = sess.scalar(select(func.count()).select_from(Photo)) or 0
        decided = (
            sess.scalar(
                select(func.count())
                .select_from(Decision)
                .where(Decision.selected != "undecided")
            )
            or 0
        )
    return {
        "photos": photos,
        "decided": decided,
        "incoming_files": _count_files(settings.photos / "incoming"),
    }


def _db_progress(stage: str) -> tuple[int, int] | None:
    with session_scope() as sess:
        photos = sess.scalar(select(func.count()).select_from(Photo)) or 0
        if stage == "ingest":
            return photos, _count_files(settings.photos / "incoming")
        if stage == "filter":
            done = (
                sess.scalar(
                    select(func.count()).select_from(Photo).where(Photo.blur_var.is_not(None))
                )
                or 0
            )
            return done, photos
        if stage == "score":
            done = (
                sess.scalar(
                    select(func.count())
                    .select_from(Photo)
                    .where(Photo.technical_score.is_not(None))
                )
                or 0
            )
            return done, photos
        if stage == "cluster":
            clusters = sess.scalar(select(func.count()).select_from(Cluster)) or 0
            return (1 if clusters else 0), 1
        if stage == "submit":
            applied = (
                sess.scalar(
                    select(func.count()).select_from(Decision).where(Decision.applied == 1)
                )
                or 0
            )
            decided = (
                sess.scalar(
                    select(func.count())
                    .select_from(Decision)
                    .where(Decision.selected != "undecided")
                )
                or 0
            )
            return applied, decided
        if stage == "enhance":
            eligible = (
                sess.scalar(
                    select(func.count())
                    .select_from(Decision)
                    .where(Decision.action.in_(ENHANCE_ACTIONS))
                )
                or 0
            )
            tiffs = _count_files(settings.photos / "exported", TIFF_SUFFIXES)
            return min(tiffs, eligible), eligible
    return None


def stage_progress(stage: str) -> tuple[int, int]:
    """(current, total) for display. total == 0 means indeterminate."""
    if stage == "export-jpeg":
        jpegs = _count_files(settings.photos / settings.jpeg_subdir, JPEG_SUFFIXES)
        sources = _count_files(settings.photos / "library") + _count_files(
            settings.photos / "exported", TIFF_SUFFIXES
        )
        return min(jpegs, sources), sources
    result = _db_progress(stage)
    if result is None:
        return 0, 0
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/pytest tests/test_orchestrator_progress.py -q`
Expected: 6 passed

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv-dev/bin/ruff check app/orchestrator/progress.py tests/test_orchestrator_progress.py
.venv-dev/bin/mypy app/orchestrator/progress.py
git add app/orchestrator/progress.py tests/test_orchestrator_progress.py
git commit -m "feat(orchestrator): derive stage progress from DB and filesystem"
```

---

### Task 7: Reset wrapper (`app/orchestrator/reset.py`)

**Files:**
- Create: `app/orchestrator/reset.py`

Tiny module (tested through the route tests in Task 9 — the underlying `end_session` is existing, already-exercised code).

- [ ] **Step 1: Implement `app/orchestrator/reset.py`**

```python
"""In-container session reset for the "New batch" UI action."""

from __future__ import annotations

import asyncio

from scripts.end_session import end_session

RESET_CONFIRM_TOKEN = "RESET"


async def reset_session() -> None:
    """Wipe DB + cache + working dirs and re-run migrations.

    `end_session` is blocking (file IO + an alembic subprocess), so it runs
    in a worker thread to keep the event loop responsive.
    """
    await asyncio.to_thread(end_session, True)
```

- [ ] **Step 2: Lint, typecheck, commit**

```bash
.venv-dev/bin/ruff check app/orchestrator/reset.py
.venv-dev/bin/mypy app/orchestrator/reset.py
git add app/orchestrator/reset.py
git commit -m "feat(orchestrator): async session-reset wrapper for New batch"
```

---

### Task 8: System monitor (`app/monitor/stats.py`)

**Files:**
- Create: `app/monitor/__init__.py` (empty)
- Create: `app/monitor/stats.py`
- Test: `tests/test_monitor_stats.py`

- [ ] **Step 1: Write the failing tests**

```python
"""System monitor: psutil sampling, NVML graceful degradation, disk warnings."""

from __future__ import annotations

from pathlib import Path

from app.monitor import stats as stats_mod
from app.monitor.stats import SystemMonitor


def test_sample_shape_without_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    sample = mon.sample()
    assert sample["gpu"] is None
    assert 0 <= sample["cpu_pct"] <= 100 * 64
    assert sample["ram_total_mb"] > 0
    assert sample["disks"]["photos"]["total_gb"] > 0


def test_missing_mount_degrades_to_none(monkeypatch) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": Path("/definitely-not-a-mount")})
    sample = mon.sample()
    assert sample["disks"]["photos"] is None


def test_snapshot_includes_window_and_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    mon.sample()
    mon.sample()
    snap = mon.snapshot()
    assert len(snap["window"]) == 2
    assert snap["current"] == snap["window"][-1]
    assert isinstance(snap["warnings"], list)


def test_low_disk_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    sample = mon.sample()
    sample["disks"]["photos"]["free_gb"] = 3.0
    warnings = mon._warnings(sample)
    assert any("free space" in w.lower() for w in warnings)


def test_gpu_failure_disables_further_reads(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return None

    monkeypatch.setattr(stats_mod, "_read_gpu", flaky)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    mon.sample()
    mon.sample()
    assert calls["n"] == 1  # second sample skipped the broken NVML path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/pytest tests/test_monitor_stats.py -q`
Expected: FAIL with `ModuleNotFoundError` on `app.monitor`

- [ ] **Step 3: Implement**

`app/monitor/__init__.py`: empty file.

`app/monitor/stats.py`:

```python
"""System resource sampling: CPU/RAM/disk via psutil, GPU via NVML (optional).

NVML absence or failure must never break the stats endpoint: the GPU field
degrades to None and further NVML reads are skipped for the process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 1.0
WINDOW_SAMPLES = 60
DISK_WARN_FREE_GB = 20.0
_MB = 1024 * 1024
_GB = 1024 * 1024 * 1024


def _read_gpu() -> dict[str, Any] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "util_pct": float(util.gpu),
            "vram_used_mb": int(mem.used) // _MB,
            "vram_total_mb": int(mem.total) // _MB,
            "temp_c": int(temp),
        }
    except Exception:  # noqa: BLE001 — any NVML breakage means "no GPU stats"
        return None


class SystemMonitor:
    def __init__(self, mounts: dict[str, Path]) -> None:
        self._mounts = mounts
        self._window: deque[dict[str, Any]] = deque(maxlen=WINDOW_SAMPLES)
        self._gpu_available = True

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        sample = {
            "ts": time.time(),
            "cpu_pct": psutil.cpu_percent(),
            "cpu_per_core": psutil.cpu_percent(percpu=True),
            "ram_used_mb": (vm.total - vm.available) // _MB,
            "ram_total_mb": vm.total // _MB,
            "disks": {name: self._disk(path) for name, path in self._mounts.items()},
            "gpu": self._gpu(),
        }
        self._window.append(sample)
        return sample

    def snapshot(self) -> dict[str, Any]:
        current = self._window[-1] if self._window else self.sample()
        return {
            "current": current,
            "window": list(self._window),
            "warnings": self._warnings(current),
        }

    async def run(self) -> None:
        while True:
            try:
                self.sample()
            except Exception:  # noqa: BLE001 — sampling must never die
                logger.exception("system sampling failed")
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)

    def _disk(self, path: Path) -> dict[str, float] | None:
        try:
            usage = psutil.disk_usage(str(path))
        except OSError:
            return None
        return {
            "used_gb": usage.used / _GB,
            "total_gb": usage.total / _GB,
            "free_gb": usage.free / _GB,
        }

    def _gpu(self) -> dict[str, Any] | None:
        if not self._gpu_available:
            return None
        gpu = _read_gpu()
        if gpu is None:
            self._gpu_available = False
            logger.info("GPU stats unavailable (NVML init/read failed); disabling")
        return gpu

    def _warnings(self, sample: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        photos = (sample.get("disks") or {}).get("photos")
        if photos is not None and photos["free_gb"] < DISK_WARN_FREE_GB:
            warnings.append(
                f"Low free space on photos volume: {photos['free_gb']:.0f} GB left "
                "— enhance/export may fill the disk"
            )
        return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-dev/bin/pytest tests/test_monitor_stats.py -q`
Expected: 5 passed

- [ ] **Step 5: Lint, typecheck, commit**

```bash
.venv-dev/bin/ruff check app/monitor/ tests/test_monitor_stats.py
.venv-dev/bin/mypy app/monitor/
git add app/monitor/ tests/test_monitor_stats.py
git commit -m "feat(monitor): psutil+NVML system sampler with rolling window"
```

---

### Task 9: API routes (`deps.py`, `routes/pipeline.py`, `routes/system.py`)

**Files:**
- Create: `app/api/deps.py`
- Create: `app/api/routes/pipeline.py`
- Create: `app/api/routes/system.py`
- Test: `tests/test_pipeline_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Pipeline + system routes, with the real runner (python -c) and mocked DB."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import pipeline, system
from app.orchestrator.runner import JobRunner
from app.orchestrator.stages import StageDef

PY = (sys.executable, "-c")


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
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
    return TestClient(app)


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


def test_system_stats_shape(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    from app.monitor import stats as stats_mod
    from app.monitor.stats import SystemMonitor

    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    monkeypatch.setattr(system, "get_monitor", lambda: SystemMonitor(mounts={"photos": tmp_path}))
    payload = client.get("/api/system/stats").json()
    assert payload["current"]["gpu"] is None
    assert "warnings" in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-dev/bin/pytest tests/test_pipeline_routes.py -q`
Expected: FAIL with `ImportError` on `app.api.deps` / `app.api.routes.pipeline`

- [ ] **Step 3: Implement `app/api/deps.py`**

```python
"""Lazily-constructed singletons shared by API routes (single-user app)."""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.monitor.stats import SystemMonitor
from app.orchestrator.autorun import AutoRun
from app.orchestrator.runner import JobRunner


@lru_cache(maxsize=1)
def get_runner() -> JobRunner:
    return JobRunner(log_dir=settings.cache / "logs")


@lru_cache(maxsize=1)
def get_autorun() -> AutoRun:
    return AutoRun(get_runner())


@lru_cache(maxsize=1)
def get_monitor() -> SystemMonitor:
    return SystemMonitor(
        mounts={
            "photos": settings.photos,
            "cache": settings.cache,
            "models": settings.models,
        }
    )
```

- [ ] **Step 4: Implement `app/api/routes/pipeline.py`**

```python
"""Pipeline orchestration endpoints: status, run, auto-run, cancel, logs, reset."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import get_autorun, get_runner
from app.orchestrator import progress
from app.orchestrator.reset import RESET_CONFIRM_TOKEN, reset_session
from app.orchestrator.runner import JobBusyError
from app.orchestrator.stages import STAGE_BY_NAME, STAGES

router = APIRouter()

AUTO_RUN_LEGS = (1, 2)


class ResetRequest(BaseModel):
    confirm: str


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _stage_payload(name: str) -> dict[str, Any]:
    stage = STAGE_BY_NAME[name]
    record = get_runner().record_for(name)
    current, total = progress.stage_progress(name)
    if record is not None:
        state = record.status.value
    elif total > 0 and current >= total:
        state = "done"  # re-derived after a server restart
    else:
        state = "pending"
    return {
        "name": stage.name,
        "title": stage.title,
        "leg": stage.leg,
        "state": state,
        "progress": {"current": current, "total": total},
        "started_at": _iso(record.started_at) if record else None,
        "finished_at": _iso(record.finished_at) if record else None,
        "exit_code": record.exit_code if record else None,
    }


@router.get("/status")
def status() -> dict[str, Any]:
    runner = get_runner()
    return {
        "stages": [_stage_payload(s.name) for s in STAGES],
        "running": runner.running_stage,
        "run_seq": runner.run_seq,
        "autorun_leg": get_autorun().active_leg,
        "batch": progress.batch_summary(),
    }


@router.post("/run/{stage_name}")
async def run_stage(stage_name: str) -> dict[str, str]:
    stage = STAGE_BY_NAME.get(stage_name)
    if stage is None:
        raise HTTPException(status_code=422, detail=f"unknown stage: {stage_name}")
    try:
        await get_runner().start(stage.name, stage.cli_args)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"started": stage.name}


@router.post("/auto/{leg}")
async def run_auto(leg: int) -> dict[str, int]:
    if leg not in AUTO_RUN_LEGS:
        raise HTTPException(status_code=422, detail=f"leg must be one of {AUTO_RUN_LEGS}")
    try:
        get_autorun().start(leg)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"autorun_leg": leg}


@router.post("/cancel")
async def cancel() -> dict[str, bool]:
    await get_autorun().cancel()
    return {"cancelled": True}


@router.get("/logs")
def logs(after: int = 0) -> dict[str, Any]:
    runner = get_runner()
    next_offset, lines = runner.log_lines(max(after, 0))
    record = runner.latest_record()
    return {
        "stage": record.stage if record else None,
        "run_seq": runner.run_seq,
        "log_path": record.log_path if record else None,
        "next": next_offset,
        "lines": lines,
    }


@router.post("/reset")
async def reset(req: ResetRequest) -> dict[str, bool]:
    if req.confirm != RESET_CONFIRM_TOKEN:
        raise HTTPException(
            status_code=422, detail=f'confirm must be "{RESET_CONFIRM_TOKEN}"'
        )
    runner = get_runner()
    if runner.running_stage is not None:
        raise HTTPException(status_code=409, detail="cannot reset while a job is running")
    await reset_session()
    runner.clear_history()
    return {"reset": True}
```

- [ ] **Step 5: Implement `app/api/routes/system.py`**

```python
"""System resource stats endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import get_monitor

router = APIRouter()


@router.get("/stats")
def stats() -> dict[str, Any]:
    return get_monitor().snapshot()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv-dev/bin/pytest tests/test_pipeline_routes.py -q`
Expected: 8 passed

- [ ] **Step 7: Lint, typecheck, commit**

```bash
.venv-dev/bin/ruff check app/api/deps.py app/api/routes/pipeline.py app/api/routes/system.py tests/test_pipeline_routes.py
.venv-dev/bin/mypy app/api/deps.py app/api/routes/pipeline.py app/api/routes/system.py
git add app/api/deps.py app/api/routes/pipeline.py app/api/routes/system.py tests/test_pipeline_routes.py
git commit -m "feat(api): pipeline orchestration + system stats routes"
```

---

### Task 10: Wire routes + monitor into the app (`app/api/main.py`)

**Files:**
- Modify: `app/api/main.py` (whole file — it is 39 lines)

- [ ] **Step 1: Replace `app/api/main.py`**

```python
"""FastAPI app serving the control-center UI + review API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import get_monitor
from app.api.routes import cluster, decide, photo, pipeline, queue, submit, system
from app.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    sampler = asyncio.create_task(get_monitor().run())
    yield
    sampler.cancel()


app = FastAPI(title="raw-curator", lifespan=lifespan)

app.include_router(queue.router, prefix="/api/queue", tags=["queue"])
app.include_router(photo.router, prefix="/api/photo", tags=["photo"])
app.include_router(cluster.router, prefix="/api/cluster", tags=["cluster"])
app.include_router(decide.router, prefix="/api/decide", tags=["decide"])
app.include_router(submit.router, prefix="/api/submit", tags=["submit"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Raw preview/thumb files live in the bind-mounted cache dir. check_dir=False:
# the dir exists in-container but not necessarily at import time elsewhere.
app.mount(
    "/cache", StaticFiles(directory=str(settings.cache), check_dir=False), name="cache"
)

# Static SPA assets bundled in the image.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR)), name="ui")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
```

- [ ] **Step 2: Verify the app imports and all routes are registered**

```bash
.venv-dev/bin/python -c "
from app.api.main import app
paths = {r.path for r in app.routes}
assert '/api/pipeline/status' in paths and '/api/system/stats' in paths, paths
print('routes ok')
"
```

Expected: `routes ok`

- [ ] **Step 3: Run the whole new-backend test set, lint, typecheck, commit**

```bash
.venv-dev/bin/pytest tests/test_orchestrator_stages.py tests/test_orchestrator_runner.py \
  tests/test_orchestrator_autorun.py tests/test_orchestrator_progress.py \
  tests/test_monitor_stats.py tests/test_pipeline_routes.py -q
.venv-dev/bin/ruff check app/api/main.py
.venv-dev/bin/mypy app/api/main.py
git add app/api/main.py
git commit -m "feat(api): wire pipeline/system routers and stats sampler lifespan"
```

---

### Task 11: Frontend foundation (`ui.js`, `api.js`, new `index.html`)

**Files:**
- Create: `app/api/static/js/ui.js`
- Create: `app/api/static/js/api.js`
- Modify: `app/api/static/index.html` (replaced by a thin shell; old inline app is ported in Tasks 12-13 — the porting source stays readable via `git show HEAD:app/api/static/index.html`)

**Porting note for all frontend tasks:** the current `app/api/static/index.html` contains the working review app. Tasks 12-13 move its components into modules **verbatim except where a diff is shown**. Until Task 13 completes, the UI is broken — that is fine mid-branch; Tasks 11-13 land as one commit at the end of Task 13.

- [ ] **Step 1: Create `app/api/static/js/ui.js`**

```js
// Shared React binding + hooks (no build step; esm.sh CDN).
import React from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import htm from "https://esm.sh/htm@3.1.1";

export const html = htm.bind(React.createElement);
export { React, createRoot };
export const { useState, useEffect, useCallback, useMemo, useRef, Fragment } = React;

export function useQuery(fetcher, deps) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const refetch = useCallback(async () => {
    setLoading(true); setError(null);
    try { setData(await fetcher()); }
    catch (e) { setError(e); }
    finally { setLoading(false); }
  }, deps);
  useEffect(() => { refetch(); }, [refetch]);
  return { data, error, loading, refetch };
}

export function useKeyboardShortcuts(handlers) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const h = handlers[e.key];
      if (h) { e.preventDefault(); h(e); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handlers]);
}

// Poll `fn` every `ms` (fires immediately). Errors are swallowed so a
// transient fetch failure doesn't kill the loop.
export function usePoll(fn, ms) {
  useEffect(() => {
    let alive = true;
    const tick = () => { if (alive) fn().catch(() => {}); };
    tick();
    const id = setInterval(tick, ms);
    return () => { alive = false; clearInterval(id); };
  }, [fn, ms]);
}
```

- [ ] **Step 2: Create `app/api/static/js/api.js`**

```js
// Single API client for all backend endpoints.
async function j(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(`${path}: ${detail}`);
  }
  return r.json();
}

const post = (path, body) => j(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  // review
  queue: (sort = "score") => j(`/api/queue/?sort=${sort}`),
  clusters: () => j(`/api/cluster/`),
  photo: (hash) => j(`/api/photo/${hash}`),
  decide: (body) => post(`/api/decide/`, body),
  decideAll: (selected) => post(`/api/decide/all`, { selected }),
  pending: () => j(`/api/decide/pending`),
  submit: () => post(`/api/submit/`),
  // pipeline
  pipelineStatus: () => j(`/api/pipeline/status`),
  runStage: (name) => post(`/api/pipeline/run/${name}`),
  autoRun: (leg) => post(`/api/pipeline/auto/${leg}`),
  cancelJob: () => post(`/api/pipeline/cancel`),
  logs: (after = 0) => j(`/api/pipeline/logs?after=${after}`),
  resetSession: () => post(`/api/pipeline/reset`, { confirm: "RESET" }),
  // system
  systemStats: () => j(`/api/system/stats`),
};
```

- [ ] **Step 3: Replace `app/api/static/index.html` with the shell**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>raw-curator — control center</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    html, body, #root { height: 100%; background: #0a0a0a; color: #e5e5e5; }
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size: 0.7rem; padding: 0.1rem 0.35rem;
           border: 1px solid #404040; border-radius: 4px; background: #1a1a1a; }
    .photo-tile { transition: transform 0.12s; }
    .photo-tile:hover { transform: scale(1.02); }
    .spark { font-family: ui-monospace, Menlo, monospace; letter-spacing: -1px; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/ui/js/app.js"></script>
</body>
</html>
```

(No commit yet — the app entry point doesn't exist until Task 13.)

---

### Task 12: Control-center components (`timeline.js`, `stage-panel.js`, `resources.js`)

**Files:**
- Create: `app/api/static/js/timeline.js`
- Create: `app/api/static/js/stage-panel.js`
- Create: `app/api/static/js/resources.js`

- [ ] **Step 1: Create `app/api/static/js/timeline.js`**

```js
// TimelineBar — one chip per pipeline stage plus the Review pseudo-stage.
import { html } from "./ui.js";

const CHIP_STYLES = {
  pending: "bg-zinc-900 text-zinc-500",
  running: "bg-sky-950 text-sky-200 ring-2 ring-sky-500",
  done: "bg-emerald-950 text-emerald-200",
  failed: "bg-rose-950 text-rose-200 ring-2 ring-rose-600",
  cancelled: "bg-amber-950 text-amber-200",
  review: "bg-amber-950 text-amber-200",
};

function pct(progress) {
  if (!progress || !progress.total) return null;
  return Math.min(100, Math.round((progress.current / progress.total) * 100));
}

function elapsed(startIso, endIso) {
  if (!startIso) return null;
  const ms = (endIso ? new Date(endIso) : new Date()) - new Date(startIso);
  const s = Math.max(0, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

function chipSub(s) {
  const p = pct(s.progress);
  if (s.state === "running") return p == null ? elapsed(s.started_at) : `${p}% · ${elapsed(s.started_at)}`;
  if (s.state === "done") return elapsed(s.started_at, s.finished_at) ?? "done";
  if (s.state === "failed") return `exit ${s.exit_code ?? "?"}`;
  if (s.state === "cancelled") return "cancelled";
  return null;
}

function Chip({ label, sub, state, active, onClick, disabled }) {
  const style = CHIP_STYLES[state] ?? CHIP_STYLES.pending;
  const ring = active ? "outline outline-2 outline-zinc-400" : "";
  return html`
    <button
      class="flex-1 min-w-0 rounded-lg px-2 py-1.5 text-center text-xs ${style} ${ring}
             disabled:opacity-40 disabled:cursor-not-allowed"
      onClick=${onClick} disabled=${disabled} title=${label}>
      <div class="font-semibold truncate">${label}</div>
      <div class="truncate text-[10px] opacity-80">${sub ?? " "}</div>
    </button>`;
}

export function TimelineBar({ status, panel, onRunStage, onShowStage, onShowReview }) {
  const stages = status?.stages ?? [];
  const busy = Boolean(status?.running) || status?.autorun_leg != null;
  const batch = status?.batch ?? {};
  const chips = [];
  for (const s of stages) {
    chips.push(html`
      <${Chip} key=${s.name} label=${s.title} sub=${chipSub(s)} state=${s.state}
        active=${panel === "pipeline" && (status?.running === s.name)}
        disabled=${busy && s.state !== "running"}
        onClick=${() => (s.state === "running" ? onShowStage(s.name) : onRunStage(s.name))} />`);
    if (s.name === "cluster") {
      chips.push(html`
        <${Chip} key="review" label="Review" state="review"
          sub="${batch.decided ?? 0}/${batch.photos ?? 0} decided"
          active=${panel === "review"} disabled=${false}
          onClick=${onShowReview} />`);
    }
  }
  return html`<div class="flex gap-1.5 px-3 py-2">${chips}</div>`;
}
```

- [ ] **Step 2: Create `app/api/static/js/stage-panel.js`**

```js
// StagePanel — progress + live log for the running (or last-run) stage.
import { html, useEffect, useRef, useState } from "./ui.js";

export function StagePanel({ status, log }) {
  const [follow, setFollow] = useState(true);
  const boxRef = useRef(null);
  const lines = log?.lines ?? [];
  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [lines.length, follow]);

  const stageName = log?.stage ?? status?.running;
  const stage = status?.stages?.find((s) => s.name === stageName);
  if (!stage) {
    return html`<div class="p-10 text-zinc-500 text-sm">
      No stage has run yet. Click a stage above, or press Auto-run to go
      ingest → filter → score → cluster and stop for your review.
    </div>`;
  }
  const p = stage.progress ?? {};
  const pctv = p.total ? Math.min(100, Math.round((p.current / p.total) * 100)) : null;
  return html`
    <div class="flex flex-col h-full p-3 gap-2">
      <div class="flex items-baseline gap-3">
        <span class="font-semibold">${stage.title}</span>
        <span class="text-sm text-zinc-400">
          ${stage.state}${stage.exit_code != null && stage.state !== "done" ? ` (exit ${stage.exit_code})` : ""}
        </span>
        ${p.total ? html`<span class="text-sm text-zinc-400">${p.current} / ${p.total}</span>` : null}
      </div>
      ${pctv != null && html`
        <div class="h-2 bg-zinc-800 rounded">
          <div class="h-2 rounded ${stage.state === "failed" ? "bg-rose-600" : "bg-sky-500"}"
               style=${{ width: `${pctv}%` }}></div>
        </div>`}
      ${stage.state === "failed" && html`
        <div class="text-xs text-rose-300 bg-rose-950/60 rounded px-2 py-1">
          Stage failed — last log lines below; full log:
          <span class="font-mono">${log?.log_path ?? "cache/logs/"}</span>
        </div>`}
      <div class="flex items-center gap-2 text-xs text-zinc-500">
        <span>live log</span>
        <button class="px-1.5 rounded ${follow ? "bg-zinc-700" : "bg-zinc-900"}"
                onClick=${() => setFollow(!follow)}>auto-scroll ${follow ? "on" : "off"}</button>
      </div>
      <div ref=${boxRef}
           class="flex-1 overflow-auto bg-black/60 rounded p-2 font-mono text-[11px] leading-4 text-zinc-300">
        ${lines.map((l, i) => html`<div key=${i}>${l}</div>`)}
      </div>
    </div>`;
}
```

- [ ] **Step 3: Create `app/api/static/js/resources.js`**

```js
// ResourceBar — docked system stats with unicode sparklines; expandable charts.
import { html, useState } from "./ui.js";

const BARS = "▁▂▃▄▅▆▇█";

export function spark(values, max) {
  if (!values.length) return "";
  const m = max || Math.max(...values, 1);
  return values
    .map((v) => BARS[Math.max(0, Math.min(7, Math.floor((v / m) * 7.99)))])
    .join("");
}

const gb = (mb) => (mb / 1024).toFixed(1);
const seriesFrom = (win, pick) =>
  (win ?? []).map(pick).filter((v) => v != null).slice(-30);

export function ResourceBar({ stats }) {
  const [expanded, setExpanded] = useState(false);
  const cur = stats?.current;
  if (!cur) {
    return html`<div class="px-3 py-1.5 text-xs text-zinc-600 border-t border-zinc-800">system stats…</div>`;
  }
  const win = stats.window ?? [];
  const cpuS = seriesFrom(win, (s) => s.cpu_pct);
  const ramS = seriesFrom(win, (s) => s.ram_used_mb);
  const gpuS = seriesFrom(win, (s) => s.gpu?.util_pct);
  const g = cur.gpu;
  const photosDisk = cur.disks?.photos;
  return html`
    <div class="border-t border-zinc-800 text-xs">
      ${(stats.warnings ?? []).map((w) => html`
        <div key=${w} class="px-3 py-1 bg-amber-950/70 text-amber-200">⚠ ${w}</div>`)}
      <div class="flex items-center gap-4 px-3 py-1.5 text-zinc-300 flex-wrap">
        <span>CPU ${Math.round(cur.cpu_pct)}% <span class="spark text-sky-400">${spark(cpuS, 100)}</span></span>
        <span>RAM ${gb(cur.ram_used_mb)}/${gb(cur.ram_total_mb)} GB
          <span class="spark text-emerald-400">${spark(ramS, cur.ram_total_mb)}</span></span>
        ${g ? html`
          <span>GPU ${Math.round(g.util_pct)}% <span class="spark text-amber-400">${spark(gpuS, 100)}</span></span>
          <span>VRAM ${gb(g.vram_used_mb)}/${gb(g.vram_total_mb)} GB</span>`
          : html`<span class="text-zinc-600">GPU stats unavailable</span>`}
        ${photosDisk && html`<span>Disk ${photosDisk.used_gb.toFixed(0)}/${photosDisk.total_gb.toFixed(0)} GB
          (${photosDisk.free_gb.toFixed(0)} free)</span>`}
        <button class="ml-auto text-zinc-500 hover:text-zinc-300" onClick=${() => setExpanded(!expanded)}>
          ${expanded ? "▾ collapse" : "▴ expand"}
        </button>
      </div>
      ${expanded && html`
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 px-3 pb-2 text-zinc-400">
          <div>
            <div class="text-zinc-500">CPU per core</div>
            <div class="font-mono">${(cur.cpu_per_core ?? []).map((c) => Math.round(c)).join(" ")}</div>
            <div class="spark text-sky-400 text-base">${spark(cpuS, 100)}</div>
          </div>
          <div>
            <div class="text-zinc-500">RAM</div>
            <div>${gb(cur.ram_used_mb)} / ${gb(cur.ram_total_mb)} GB</div>
            <div class="spark text-emerald-400 text-base">${spark(ramS, cur.ram_total_mb)}</div>
          </div>
          <div>
            <div class="text-zinc-500">GPU</div>
            ${g ? html`
              <div>${Math.round(g.util_pct)}% · ${g.temp_c}°C · VRAM ${gb(g.vram_used_mb)}/${gb(g.vram_total_mb)} GB</div>
              <div class="spark text-amber-400 text-base">${spark(gpuS, 100)}</div>`
              : html`<div>unavailable</div>`}
          </div>
          <div>
            <div class="text-zinc-500">Disks</div>
            ${Object.entries(cur.disks ?? {}).map(([name, d]) => html`
              <div key=${name}>${name}: ${d
                ? `${d.used_gb.toFixed(0)}/${d.total_gb.toFixed(0)} GB (${d.free_gb.toFixed(0)} free)`
                : "n/a"}</div>`)}
          </div>
        </div>`}
    </div>`;
}
```

(No commit yet.)

---

### Task 13: Review module port + app root (`review.js`, `app.js`)

**Files:**
- Create: `app/api/static/js/review.js`
- Create: `app/api/static/js/app.js`

- [ ] **Step 1: Create `app/api/static/js/review.js` by porting from the OLD index.html**

The old inline app is readable via `git show HEAD:app/api/static/index.html` (old line numbers below refer to that file).

Header of the new file:

```js
// Review UI — ported from the pre-control-center index.html inline app.
// Components are unchanged except: imports, and ReviewPanel replacing App/Header.
import { api } from "./api.js";
import {
  Fragment, html, useCallback, useEffect, useMemo, useRef, useState,
  useKeyboardShortcuts, useQuery,
} from "./ui.js";

const PAGE_SIZE = 200;
```

Then copy **verbatim** from the old index.html these components (old line numbers):
- `Stars` (108-111)
- `DecisionBadge` (113-118)
- `PhotoTile` (120-151)
- `PhotoGrid` (153-158)
- `PagedGrid` (160-187)
- `ClusterSection` (189-216)
- `ClusterView` (218-226) — change the empty-state text `Run <span class="kbd">make cluster</span>.` to `Run the Cluster stage above.`
- `EngineQualityPanel` (228-270)
- `DetailModal` (272-382) — change the hint `Run <span class="font-mono">make enhance</span> to compute the quality breakdown.` to `Run the Enhance stage to compute the quality breakdown.`

Do **not** copy the old `Header` or `App`. Instead, append these (the old `App` with pipeline chrome removed and a toolbar added):

```js
function Toolbar({ count, shown, pendingCount, sort, setSort, view, setView,
                   onMarkAllNo, onSubmitAndContinue, busy }) {
  const truncated = shown != null && shown < count;
  const tabBtn = (id, label) => html`
    <button onClick=${() => setView(id)}
      class="px-3 py-1 rounded text-sm ${view === id ? "bg-zinc-700 text-zinc-100" : "bg-zinc-900 text-zinc-400"}">
      ${label}
    </button>`;
  return html`
    <div class="flex items-center justify-between border-b border-zinc-800 px-4 py-2 sticky top-0 bg-zinc-950/95 backdrop-blur z-20">
      <div class="flex items-baseline gap-4">
        <span class="text-sm text-zinc-400">
          ${count} photo${count === 1 ? "" : "s"}${truncated ? ` (showing ${shown})` : ""}
        </span>
        <span class="text-sm text-amber-400">${pendingCount} pending decision${pendingCount === 1 ? "" : "s"}</span>
      </div>
      <div class="flex items-center gap-2">
        <div class="flex gap-1 mr-2 p-0.5 bg-zinc-900 border border-zinc-800 rounded">
          ${tabBtn("all", "All")}
          ${tabBtn("clusters", "Clusters")}
        </div>
        ${view === "all" && html`
          <label class="text-xs text-zinc-500">sort</label>
          <select value=${sort} onChange=${(e) => setSort(e.target.value)}
                  class="bg-zinc-900 border border-zinc-800 text-sm rounded px-2 py-1">
            <option value="score">score (technical)</option>
            <option value="captured">captured</option>
          </select>`}
        <button onClick=${onMarkAllNo} disabled=${busy || count === 0}
                class="px-3 py-1.5 rounded bg-rose-800 hover:bg-rose-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm">
          don't keep any RAW
        </button>
        <button onClick=${onSubmitAndContinue} disabled=${busy || pendingCount === 0}
                class="px-3 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm">
          ${busy ? "working…" : `✔ Submit & continue (${pendingCount})`}
        </button>
      </div>
    </div>`;
}

export function ReviewPanel({ pipelineBusy, onSubmitAndContinue }) {
  const [sort, setSort] = useState("score");
  const [view, setView] = useState("all");
  const { data: queueItems, refetch: refetchQueue } = useQuery(() => api.queue(sort), [sort]);
  const { data: clusters, refetch: refetchClusters } = useQuery(() => api.clusters(), []);
  const { data: pending, refetch: refetchPending } = useQuery(() => api.pending(), []);
  const [openHash, setOpenHash] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  useEffect(() => { setVisibleCount(PAGE_SIZE); }, [view, sort]);
  const loadMore = useCallback(() => setVisibleCount((c) => c + PAGE_SIZE), []);

  const items = useMemo(() => {
    if (view === "clusters") return (clusters ?? []).flatMap((c) => c.photos);
    return queueItems ?? null;
  }, [view, queueItems, clusters]);

  const refreshAll = useCallback(() => {
    refetchQueue(); refetchClusters(); refetchPending();
  }, [refetchQueue, refetchClusters, refetchPending]);

  const openIndex = useMemo(
    () => items?.findIndex((p) => p.hash === openHash) ?? -1,
    [items, openHash]
  );
  const next = useCallback(() => {
    if (!items || items.length === 0) return;
    const i = openIndex < 0 ? 0 : (openIndex + 1) % items.length;
    setOpenHash(items[i].hash);
  }, [items, openIndex]);
  const prev = useCallback(() => {
    if (!items || items.length === 0) return;
    const i = openIndex <= 0 ? items.length - 1 : openIndex - 1;
    setOpenHash(items[i].hash);
  }, [items, openIndex]);

  const submitAndContinue = useCallback(() => {
    const n = pending?.length ?? 0;
    if (!confirm(
      `Submit ${n} decision(s) and continue?\n\n` +
      `This runs: Submit (moves files on disk) → Enhance → Export JPEG.\n` +
      `Photos marked "no" get their source RAW deleted after enhancement.`
    )) return;
    onSubmitAndContinue();
  }, [pending, onSubmitAndContinue]);

  const onMarkAllNo = useCallback(async () => {
    if (!confirm(
      `Mark EVERY photo in the batch as NO? Originals will NOT be kept in ` +
      `library — every RAW is enhanced, then the source RAW is deleted after ` +
      `processing. You can still review before Submit.`
    )) return;
    setBusy(true);
    try {
      const res = await api.decideAll("no");
      setStatus(`Marked ${res.staged} photo(s) as no.`);
      refreshAll();
    } catch (e) {
      setStatus(`Mark all failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [refreshAll]);

  const body = (() => {
    if (view === "clusters") {
      if (!clusters) return html`<div class="p-8 text-zinc-500">loading…</div>`;
      return html`<${ClusterView} clusters=${clusters} onSelect=${setOpenHash} />`;
    }
    if (!items) return html`<div class="p-8 text-zinc-500">loading…</div>`;
    if (items.length === 0) {
      return html`<div class="p-8 text-zinc-500">No photos yet. Run the Ingest stage first.</div>`;
    }
    return html`<${PagedGrid} items=${items} visible=${visibleCount}
                               onMore=${loadMore} onSelect=${setOpenHash} />`;
  })();

  return html`
    <${Fragment}>
      <${Toolbar} count=${items?.length ?? 0}
                  shown=${(view === "all" && items) ? Math.min(visibleCount, items.length) : null}
                  pendingCount=${pending?.length ?? 0}
                  sort=${sort} setSort=${setSort} view=${view} setView=${setView}
                  onMarkAllNo=${onMarkAllNo}
                  onSubmitAndContinue=${submitAndContinue}
                  busy=${busy || pipelineBusy} />
      ${status && html`<div class="px-4 py-1 text-xs bg-zinc-900 border-b border-zinc-800">${status}</div>`}
      ${body}
      ${openHash && html`<${DetailModal} hash=${openHash}
                          onClose=${() => setOpenHash(null)}
                          onPrev=${prev} onNext=${next}
                          onMutated=${refreshAll} />`}
    </${Fragment}>`;
}
```

- [ ] **Step 2: Create `app/api/static/js/app.js`**

```js
// Control-center root: header, timeline, context panel, resource bar.
import { api } from "./api.js";
import {
  createRoot, Fragment, html, useCallback, useRef, useState,
  usePoll,
} from "./ui.js";
import { TimelineBar } from "./timeline.js";
import { StagePanel } from "./stage-panel.js";
import { ResourceBar } from "./resources.js";
import { ReviewPanel } from "./review.js";

const POLL_BUSY_MS = 1000;
const POLL_IDLE_MS = 3000;
const LOG_KEEP_LINES = 2000;

function ResetModal({ onClose, onDone }) {
  const [text, setText] = useState("");
  const [err, setErr] = useState(null);
  const go = async () => {
    try { await api.resetSession(); onDone(); }
    catch (e) { setErr(e.message); }
  };
  return html`
    <div class="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div class="bg-zinc-900 rounded-lg p-5 w-96 space-y-3 text-sm">
        <div class="font-semibold text-rose-300">Start a new batch?</div>
        <p class="text-zinc-400">This wipes the session DB, previews, and the
          library/exported working folders (incoming photos stay). Type
          <span class="kbd">RESET</span> to confirm.</p>
        <input class="w-full bg-zinc-800 rounded px-2 py-1" value=${text}
               onInput=${(e) => setText(e.target.value)} placeholder="RESET" />
        ${err && html`<div class="text-rose-400 text-xs">${err}</div>`}
        <div class="flex justify-end gap-2">
          <button class="px-3 py-1 rounded bg-zinc-800" onClick=${onClose}>cancel</button>
          <button class="px-3 py-1 rounded bg-rose-800 disabled:opacity-40"
                  disabled=${text !== "RESET"} onClick=${go}>wipe session</button>
        </div>
      </div>
    </div>`;
}

function App() {
  const [status, setStatus] = useState(null);
  const [stats, setStats] = useState(null);
  const [log, setLog] = useState(null);
  const [panel, setPanel] = useState("pipeline"); // "pipeline" | "review"
  const [showReset, setShowReset] = useState(false);
  const [error, setError] = useState(null);
  const logRef = useRef({ seq: 0, next: 0, lines: [] });

  const busy = Boolean(status?.running) || status?.autorun_leg != null;

  const refresh = useCallback(async () => {
    const st = await api.pipelineStatus();
    setStatus(st);
    if (st.run_seq > 0) {
      const sameRun = st.run_seq === logRef.current.seq;
      const chunk = await api.logs(sameRun ? logRef.current.next : 0);
      const prev = chunk.run_seq === logRef.current.seq ? logRef.current.lines : [];
      logRef.current = {
        seq: chunk.run_seq,
        next: chunk.next,
        lines: [...prev, ...chunk.lines].slice(-LOG_KEEP_LINES),
      };
      setLog({ ...chunk, lines: logRef.current.lines });
    }
  }, []);
  usePoll(refresh, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
  usePoll(useCallback(async () => setStats(await api.systemStats()), []),
          busy ? POLL_BUSY_MS : POLL_IDLE_MS);

  const act = useCallback(async (fn) => {
    setError(null);
    try { await fn(); await refresh(); }
    catch (e) { setError(e.message); }
  }, [refresh]);

  const onRunStage = useCallback((name) => {
    const stage = status?.stages?.find((s) => s.name === name);
    if (!stage) return;
    if (stage.state === "done" && !confirm(`Re-run ${stage.title}?`)) return;
    setPanel("pipeline");
    act(() => api.runStage(name));
  }, [status, act]);

  const autoLabel = status?.autorun_leg != null
    ? `auto-run leg ${status.autorun_leg}…`
    : "▶ Auto-run (ingest → cluster)";

  return html`
    <div class="h-full flex flex-col">
      <header class="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <div class="flex items-baseline gap-3">
          <h1 class="text-lg font-semibold">raw-curator</h1>
          <span class="text-sm text-zinc-400">
            ${status?.batch?.photos ?? 0} photos · ${status?.batch?.decided ?? 0} decided
            · ${status?.batch?.incoming_files ?? 0} files incoming
          </span>
        </div>
        <div class="flex items-center gap-2 text-sm">
          <button class="px-3 py-1.5 rounded bg-sky-800 hover:bg-sky-700 disabled:opacity-40"
                  disabled=${busy} onClick=${() => act(() => api.autoRun(1))}>${autoLabel}</button>
          <button class="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40"
                  disabled=${!busy} onClick=${() => act(() => api.cancelJob())}>⏹ Stop</button>
          <button class="px-3 py-1.5 rounded bg-zinc-900 text-rose-300 hover:bg-zinc-800 disabled:opacity-40"
                  disabled=${busy} onClick=${() => setShowReset(true)}>🗑 New batch</button>
        </div>
      </header>
      ${error && html`<div class="px-4 py-1 text-xs bg-rose-950 text-rose-200">${error}</div>`}
      <${TimelineBar} status=${status} panel=${panel}
        onRunStage=${onRunStage}
        onShowStage=${() => setPanel("pipeline")}
        onShowReview=${() => setPanel("review")} />
      <main class="flex-1 min-h-0 overflow-auto">
        ${panel === "review"
          ? html`<${ReviewPanel} pipelineBusy=${busy}
                   onSubmitAndContinue=${() => { setPanel("pipeline"); act(() => api.autoRun(2)); }} />`
          : html`<${StagePanel} status=${status} log=${log} />`}
      </main>
      <${ResourceBar} stats=${stats} />
      ${showReset && html`<${ResetModal} onClose=${() => setShowReset(false)}
          onDone=${() => {
            setShowReset(false);
            logRef.current = { seq: 0, next: 0, lines: [] };
            setLog(null);
            refresh();
          }} />`}
    </div>`;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
```

- [ ] **Step 3: Local smoke test (no container needed)**

```bash
.venv-dev/bin/pip install -q "uvicorn[standard]"
mkdir -p /tmp/rc-smoke/{cache,photos/incoming,models,xmp}
RAWCURATOR_CACHE=/tmp/rc-smoke/cache \
RAWCURATOR_PHOTOS=/tmp/rc-smoke/photos \
RAWCURATOR_MODELS=/tmp/rc-smoke/models \
RAWCURATOR_XMP=/tmp/rc-smoke/xmp \
.venv-dev/bin/python -c "from app.db import engine; from app.models import Base; Base.metadata.create_all(engine)"
RAWCURATOR_CACHE=/tmp/rc-smoke/cache \
RAWCURATOR_PHOTOS=/tmp/rc-smoke/photos \
RAWCURATOR_MODELS=/tmp/rc-smoke/models \
RAWCURATOR_XMP=/tmp/rc-smoke/xmp \
.venv-dev/bin/python -m uvicorn app.api.main:app --port 8099 &
sleep 2
curl -s http://127.0.0.1:8099/api/pipeline/status | head -c 400; echo
curl -s http://127.0.0.1:8099/api/system/stats | head -c 200; echo
curl -s http://127.0.0.1:8099/ | grep -o "control center"
kill %1
```

Expected: status JSON with 7 stages (all `pending`), stats JSON with `"gpu": null` on the Mac, `control center` from the HTML title.

Then open `http://127.0.0.1:8099` in a browser (optional but recommended): timeline renders, resource bar shows CPU/RAM, GPU shows "unavailable", Review panel opens with "No photos yet".

- [ ] **Step 4: Commit the whole frontend**

```bash
git add app/api/static/
git commit -m "feat(ui): control-center frontend — timeline, stage panel, resources, ported review"
```

---

### Task 14: Docs + Makefile help

**Files:**
- Modify: `README.md` (usage section)
- Modify: `USER_GUIDE.md` (walkthrough)
- Modify: `CLAUDE.md` (serve phase + architecture notes)
- Modify: `Makefile` (help text only, line 20)

- [ ] **Step 1: Update Makefile help**

Replace the `serve` help line (line 20) with:

```makefile
	@echo "  serve           Control Center UI on http://localhost:8080 (runs all stages)"
```

Keep every target working as-is (power users / CI still use them).

- [ ] **Step 2: Update CLAUDE.md**

In the "Pipeline phases" section, extend point 5 (**serve**) — after the existing text about the bulk control, append:

```markdown
   Since the control-center rework, `serve` is the primary entry point: the UI
   (Pipeline Timeline + Context Panel) can run every stage itself. New pieces:
   `app/orchestrator/` (subprocess-per-stage `JobRunner` with a single global
   job slot, two-leg `AutoRun`: ingest→cluster, then submit→enhance→export-jpeg
   after "Submit & continue", DB-derived progress in `progress.py`),
   `app/monitor/stats.py` (psutil + NVML sampler, 60 s rolling window, disk
   warnings), and routes `/api/pipeline/*` + `/api/system/stats`. The frontend
   is ES modules under `app/api/static/js/` (CDN React, no build step).
```

In "Common commands", after the `make serve` line's block, add:

```markdown
The web UI can drive the whole pipeline (ingest → export-jpeg, plus New-batch
reset), so per-stage make targets are optional; `make image` and
`make download-models` remain host-side prerequisites.
```

- [ ] **Step 3: Update README.md and USER_GUIDE.md**

Read each file first. In README, update the workflow/quickstart to: `make image` → `make download-models` → drop photos into `photos/incoming/` → `make serve` → open http://localhost:8080 → Auto-run → review → Submit & continue. In USER_GUIDE, update the step-by-step walkthrough the same way, keeping the per-stage make commands documented in an "Advanced / CLI" subsection. Mention the resource bar and the disk-space warning. Mention `.env.example` as the tuned starting point.

- [ ] **Step 4: Commit**

```bash
git add README.md USER_GUIDE.md CLAUDE.md Makefile
git commit -m "docs: control-center flow — one UI for the whole pipeline"
```

---

### Task 15: Quality pass + final verification

**Files:**
- Possibly modify: anything ruff/mypy flags

- [ ] **Step 1: Ruff over the whole repo**

```bash
.venv-dev/bin/ruff check app/ tests/
```

Expected: clean. Fix anything reported (line length 100, ignore E501 per pyproject).

- [ ] **Step 2: mypy over the new/changed modules**

```bash
.venv-dev/bin/mypy app/orchestrator/ app/monitor/ app/api/deps.py \
  app/api/routes/pipeline.py app/api/routes/system.py app/api/main.py \
  app/workers/gpu_worker.py
```

Expected: clean under `--strict` (config in pyproject). `torch`/`pynvml` resolve as Any via `ignore_missing_imports`.

- [ ] **Step 3: Full local-capable test sweep**

```bash
.venv-dev/bin/pytest tests/test_orchestrator_stages.py tests/test_orchestrator_runner.py \
  tests/test_orchestrator_autorun.py tests/test_orchestrator_progress.py \
  tests/test_monitor_stats.py tests/test_pipeline_routes.py \
  tests/test_decision_rules.py tests/test_paths.py tests/test_bulk_decisions.py -q
```

Expected: all pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: quality pass — ruff/mypy fixes"   # only if there were changes
```

- [ ] **Step 5: Hand off container verification to the user (GPU host)**

The user runs on the GPU host:

```bash
make image          # rebuild with new deps + frontend
make test           # full suite in-container
make lint typecheck
make serve          # open http://localhost:8080, run a small batch end-to-end
```

Acceptance (from the spec): Auto-run leg 1 stops at Review; Submit & continue runs leg 2 unattended; resource bar shows live CPU/RAM/GPU/VRAM/disk; a failed stage turns red with its log visible and the server stays alive.

---

## Self-review (done)

- **Spec coverage:** orchestrator (Tasks 3-7), monitor (8), API (9-10), frontend layout C + review port + resource bar (11-13), resource tuning + .env.example (2), docs (14), quality gates (15), commit sequence honored. The spec's `logs/{stage}` endpoint and review-submit behavior deviations are documented in the header.
- **Placeholders:** none — every code step has full code; the only "copy" steps reference exact components + line numbers in git history with explicit diffs.
- **Type consistency:** `JobRunner.start(stage: str, cli_args)` used identically in autorun/routes/tests; `StageDef` fields consistent; `log_lines(after) -> (next, lines)` shape matches route + JS client (`chunk.next`, `chunk.lines`); `run_seq` exposed in status + logs payloads and used by `app.js`.
