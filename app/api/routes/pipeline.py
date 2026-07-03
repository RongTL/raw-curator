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

AUTO_RUN_LEGS = tuple(sorted({s.leg for s in STAGES}))

# True while /reset awaits the (seconds-long) wipe; /run and /auto refuse to
# start jobs mid-wipe. All three routes run on one event loop — no lock needed.
_resetting = False


class ResetRequest(BaseModel):
    confirm: str


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _stage_payload(name: str) -> dict[str, Any]:
    # Display-only reads: may be slightly torn vs the event loop's in-place
    # record mutation; self-corrects on the next poll — do not add locking.
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
    if _resetting:
        raise HTTPException(status_code=409, detail="reset in progress")
    stage = STAGE_BY_NAME.get(stage_name)
    if stage is None:
        raise HTTPException(status_code=422, detail=f"unknown stage: {stage_name}")
    active_leg = get_autorun().active_leg
    if active_leg is not None:
        raise HTTPException(status_code=409, detail=f"auto-run leg {active_leg} is active")
    try:
        await get_runner().start(stage.name, stage.cli_args)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"started": stage.name}


@router.post("/auto/{leg}")
async def run_auto(leg: int) -> dict[str, int]:
    if _resetting:
        raise HTTPException(status_code=409, detail="reset in progress")
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
    global _resetting
    if req.confirm != RESET_CONFIRM_TOKEN:
        raise HTTPException(
            status_code=422, detail=f'confirm must be "{RESET_CONFIRM_TOKEN}"'
        )
    if _resetting:
        raise HTTPException(status_code=409, detail="reset already in progress")
    active_leg = get_autorun().active_leg
    if active_leg is not None:
        raise HTTPException(
            status_code=409, detail=f"cannot reset while auto-run leg {active_leg} is active"
        )
    runner = get_runner()
    if runner.running_stage is not None:
        raise HTTPException(status_code=409, detail="cannot reset while a job is running")
    _resetting = True
    try:
        await reset_session()
        runner.clear_history()
    finally:
        _resetting = False
    return {"reset": True}
