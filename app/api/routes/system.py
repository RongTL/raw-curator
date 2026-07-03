"""System resource stats endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import get_monitor

router = APIRouter()


@router.get("/stats")
def stats() -> dict[str, Any]:
    return get_monitor().snapshot()
