"""Stage a decision in the decisions table. Nothing moves on disk until the Submit stage."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.decision.bulk import stage_all, stage_cluster
from app.decision.export_rules import KEEP_CHOICES, normalize_choice
from app.models import Decision, Photo

router = APIRouter()


class DecisionIn(BaseModel):
    photo_hash: str
    export_choice: str | None = None
    keep_raw: bool | None = None
    stars: int | None = None
    favorite: bool | None = None
    note: str | None = None


class BulkDecisionIn(BaseModel):
    export_choice: str | None = None
    keep_raw: bool | None = None


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
            existing = Decision(photo_hash=d.photo_hash, keep_raw=int(settings.keep_raw_default))
            sess.add(existing)
        # An already-exported row is locked for export_choice/keep_raw (mirrors the
        # applied-skip in bulk._decision); stars/favorite/note may still change.
        applied = bool(existing.applied)
        if d.export_choice is not None:
            choice = normalize_choice(d.export_choice)
            if choice is None:
                raise HTTPException(
                    status_code=400, detail=f"unknown export_choice {d.export_choice!r}"
                )
            if not applied:
                existing.export_choice = choice
        if d.keep_raw is not None and not applied:
            existing.keep_raw = bool(d.keep_raw)  # Integer-backed column; stores 0/1
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
            staged = stage_all(sess, export_choice=d.export_choice, keep_raw=d.keep_raw)
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
    """Decisions awaiting export: a keeper choice (``original``/``enhanced``) not yet applied.

    A row set to ``undecided`` or ``discard`` (or never decided) is not pending —
    it produces no share JPEG — so the toolbar count and the export dialog match
    the header's "decided" tally, which also ignores those.
    """
    with session_scope() as sess:
        rows = (
            sess.execute(
                select(Decision).where(
                    Decision.applied == 0, Decision.export_choice.in_(KEEP_CHOICES)
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "photo_hash": d.photo_hash,
                "export_choice": d.export_choice,
                "keep_raw": bool(d.keep_raw),
                "stars": d.stars,
                "favorite": bool(d.favorite),
                "note": d.note,
            }
            for d in rows
        ]
