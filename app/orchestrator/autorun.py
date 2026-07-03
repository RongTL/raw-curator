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
        """Stop the chain first (so no next stage starts), then the current job.

        Deliberately also cancels a manually-started single-stage job when no
        chain is active: the UI "Stop" means stop whatever is running.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                if not self._task.cancelled():
                    raise  # cancel() itself was cancelled, not the chain
        await self._runner.cancel()

    async def _run(self, stages: tuple[StageDef, ...]) -> None:
        try:
            for stage in stages:
                await self._runner.start(stage.name, stage.cli_args)
                record = await self._runner.wait()
                if record.status is not JobStatus.DONE:
                    logger.warning(
                        "auto-run halted: stage %s ended %s", stage.name, record.status.value
                    )
                    return
        except Exception:
            logger.exception("auto-run leg %s crashed", self._leg)
