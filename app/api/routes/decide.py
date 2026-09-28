"""Stage a decision in the decisions table. Nothing moves on disk until the Submit stage."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.db import session_scope
from app.decision.bulk import stage_all, stage_cluster
from app.models import Decision, Photo

router = APIRouter()


class DecisionIn(BaseModel):
    photo_hash: str
    selected: str | None = None
    stars: int | None = None
    favorite: bool | None = None
    note: str | None = None


class BulkDecisionIn(BaseModel):
    selected: str


class ClusterDecisionIn(BaseModel):
    cluster_id: int
    mode: str


@router.post("/")
def stage_decision(d: DecisionIn) -> dict[str, bool]:
    with session_scope() as sess:
        if not sess.get(Photo, d.photo_hash):
            raise HTTPException(status_code=404, detail="photo not found")
        existing = sess.get(Decision, d.photo_hash)
        if existing is None:
            existing = Decision(photo_hash=d.photo_hash)
            sess.add(existing)
        if d.selected is not None:
            existing.selected = d.selected
        if d.stars is not None:
            existing.stars = d.stars
        if d.favorite is not None:
            existing.favorite = d.favorite
        if d.note is not None:
            existing.note = d.note
        return {"ok": True}


@router.post("/all")
def stage_all_decisions(d: BulkDecisionIn) -> dict[str, Any]:
    try:
        with session_scope() as sess:
            staged = stage_all(sess, d.selected)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "staged": staged}


@router.post("/cluster")
def stage_cluster_decisions(d: ClusterDecisionIn) -> dict[str, Any]:
    try:
        with session_scope() as sess:
            staged = stage_cluster(sess, d.cluster_id, d.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "staged": staged}


@router.get("/pending")
def list_pending() -> list[dict[str, Any]]:
    """Decisions awaiting Submit: staged as ``yes``/``no`` and not yet applied.

    A row set back to ``undecided`` (or never decided) is not pending — it has
    nothing to submit — so the toolbar count and the Submit dialog match the
    header's "decided" tally, which also ignores undecided.
    """
    with session_scope() as sess:
        rows = (
            sess.execute(
                select(Decision).where(Decision.applied == 0, Decision.selected.in_(("yes", "no")))
            )
            .scalars()
            .all()
        )
        return [
            {
                "photo_hash": d.photo_hash,
                "selected": d.selected,
                "stars": d.stars,
                "favorite": bool(d.favorite),
                "note": d.note,
            }
            for d in rows
        ]
