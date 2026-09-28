from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import select

from app.api.routes._review_data import enhanced_by_hash, face_counts_by_hash
from app.api.serializers import photo_summary
from app.db import session_scope
from app.models import Decision, Photo

router = APIRouter()


@router.get("/")
def list_queue(sort: str = "score", limit: int | None = None) -> list[dict[str, Any]]:
    """Return the full batch by default. `limit` is an optional safety cap;
    the UI paginates client-side (infinite scroll) and filters client-side, so
    no server cap is needed."""
    with session_scope() as sess:
        face_counts = face_counts_by_hash(sess)
        enhanced = enhanced_by_hash(sess)
        stmt = select(Photo, Decision).outerjoin(Decision, Photo.hash == Decision.photo_hash)
        if limit is not None:
            stmt = stmt.limit(limit)
        items = []
        for p, d in sess.execute(stmt).all():
            q_after, degraded = enhanced.get(p.hash, (None, False))
            items.append(
                photo_summary(
                    p, d, q_after=q_after, degraded=degraded, n_faces=face_counts.get(p.hash, 0)
                )
            )
    if sort == "score":
        items.sort(key=lambda r: (r["technical_score"] or 0.0), reverse=True)
    elif sort == "captured":
        items.sort(key=lambda r: r["captured_at"] or "")
    return items
