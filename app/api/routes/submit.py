"""Apply staged decisions atomically (file moves + DB update)."""

from __future__ import annotations

from fastapi import APIRouter

from app.decision.decide_job import apply_decisions

router = APIRouter()


@router.post("/")
def submit_batch() -> dict:
    apply_decisions()
    return {"ok": True}
