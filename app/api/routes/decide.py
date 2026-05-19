"""Stage a decision (in-memory). Doesn't move files — that's /api/submit."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.db import session_scope
from app.models import Decision, Photo

router = APIRouter()


class DecisionIn(BaseModel):
    photo_hash: str
    selected: str | None = None
    score_tier: str | None = None
    stars: int | None = None
    favorite: bool | None = None
    enhance_requested: bool | None = None
    note: str | None = None


@router.post("/")
def stage_decision(d: DecisionIn) -> dict:
    with session_scope() as sess:
        if not sess.get(Photo, d.photo_hash):
            raise HTTPException(status_code=404, detail="photo not found")
        existing = sess.get(Decision, d.photo_hash)
        if existing is None:
            existing = Decision(photo_hash=d.photo_hash)
            sess.add(existing)
        if d.selected is not None:
            existing.selected = d.selected
        if d.score_tier is not None:
            existing.score_tier = d.score_tier
        if d.stars is not None:
            existing.stars = d.stars
        if d.favorite is not None:
            existing.favorite = 1 if d.favorite else 0
        if d.enhance_requested is not None:
            existing.enhance_requested = 1 if d.enhance_requested else 0
        if d.note is not None:
            existing.note = d.note
        return {"ok": True}


@router.get("/pending")
def list_pending() -> list[dict]:
    with session_scope() as sess:
        rows = sess.execute(select(Decision).where(Decision.applied == 0)).scalars().all()
        return [
            {
                "photo_hash": d.photo_hash,
                "selected": d.selected,
                "score_tier": d.score_tier,
                "stars": d.stars,
                "favorite": bool(d.favorite),
                "enhance_requested": bool(d.enhance_requested),
                "note": d.note,
            }
            for d in rows
        ]
