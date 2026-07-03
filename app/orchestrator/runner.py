"""Single-slot async job runner: one pipeline-stage subprocess at a time.

The web server spawns `raw-curator <stage>` as a child process; a crash or
CUDA OOM kills only the child. The stdout+stderr stream is written to a log
file and kept in a bounded in-memory tail for the UI.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COMMAND: tuple[str, ...] = ("raw-curator",)
LOG_TAIL_MAX_LINES = 2000
SIGKILL_GRACE_SECONDS = 10.0
READER_DRAIN_TIMEOUT_SECONDS = 5.0
READ_CHUNK_BYTES = 65536
MAX_LINE_BYTES = 65536

_LINE_SEPARATORS = re.compile(rb"[\r\n]")


class JobStatus(StrEnum):
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


def _signal(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the child's whole process group; fall back to the child alone."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except OSError:  # includes PermissionError: pgid gone or not ours
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


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
        if self.running_stage is not None:
            raise RuntimeError("cannot clear history while a job is running")
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
                start_new_session=True,
            )
        except asyncio.CancelledError:
            # asyncio's transport kills the half-spawned child on cancellation;
            # the record must not stay RUNNING or the single slot wedges.
            self._finish(self._current, JobStatus.CANCELLED, exit_code=None)
            raise
        except OSError as exc:
            self._finish(self._current, JobStatus.FAILED, exit_code=None)
            raise RuntimeError(f"failed to spawn stage {stage!r}: {exc}") from exc
        self._reader = asyncio.create_task(self._pump(log_path))

    async def wait(self) -> JobRecord:
        record = self._current
        if record is not None and record.status is not JobStatus.RUNNING:
            # Already terminal (e.g. forced CANCELLED after a drain timeout):
            # don't block on an orphaned reader task.
            return record
        if self._reader is not None:
            # Shield: a cancelled waiter (e.g. an auto-run chain being torn
            # down) must not cancel the runner's internal reader task.
            await asyncio.shield(self._reader)
        record = self._current
        if record is None:
            raise RuntimeError("wait() called before any job was started")
        return record

    async def cancel(self) -> bool:
        proc = self._proc
        if proc is None or self.running_stage is None:
            return False
        self._cancelling = True
        _signal(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=SIGKILL_GRACE_SECONDS)
        except TimeoutError:
            _signal(proc, signal.SIGKILL)
        if self._reader is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._reader), READER_DRAIN_TIMEOUT_SECONDS)
            except TimeoutError:
                # Orphaned grandchildren may hold the pipe open past SIGKILL;
                # don't hang the caller. The reader task finishes on its own
                # later and its _finish call is an idempotent no-op.
                self._finish(self._current, JobStatus.CANCELLED, None)
        return True

    # -- internals ---------------------------------------------------------------

    async def _pump(self, log_path: Path) -> None:
        proc = self._proc
        my_record = self._current
        if proc is None or proc.stdout is None:  # pragma: no cover - defensive
            return

        def append(piece: bytes) -> None:
            # A drain timeout in cancel() can orphan this reader while a new
            # job starts; never touch the tail once this run is superseded.
            if self._current is my_record:
                self._append_tail(piece)

        try:
            pending = b""
            with log_path.open("ab") as fh:
                while True:
                    chunk = await proc.stdout.read(READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    pending += chunk
                    *complete, pending = _LINE_SEPARATORS.split(pending)
                    for piece in complete:
                        if piece:
                            append(piece)
                    if len(pending) > MAX_LINE_BYTES:
                        append(pending)
                        pending = b""
            if pending:
                append(pending)
            exit_code: int | None = await proc.wait()
        except Exception:
            stage = my_record.stage if my_record is not None else "?"
            logger.exception("log pump for stage %s crashed", stage)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            exit_code = await proc.wait()
            self._finish(my_record, JobStatus.FAILED, exit_code)
            return
        if exit_code == 0:
            # A clean exit that raced a cancel is still a success.
            self._finish(my_record, JobStatus.DONE, exit_code)
        elif self._cancelling:
            self._finish(my_record, JobStatus.CANCELLED, exit_code)
        else:
            self._finish(my_record, JobStatus.FAILED, exit_code)

    def _append_tail(self, piece: bytes) -> None:
        if len(self._tail) == self._tail.maxlen:
            self._dropped += 1
        self._tail.append(piece.decode("utf-8", errors="replace"))

    def _finish(
        self, record: JobRecord | None, status: JobStatus, exit_code: int | None
    ) -> None:
        if record is None or record is not self._current or record.status is not JobStatus.RUNNING:
            return
        record.status = status
        record.finished_at = time.time()
        record.exit_code = exit_code
        self._history[record.stage] = record
        self._proc = None
        logger.info("stage %s finished: %s (rc=%s)", record.stage, status.value, exit_code)
