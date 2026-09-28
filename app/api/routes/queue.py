from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.serializers import photo_summary
from app.db import session_scope
from app.models import Decision, Face, Photo, PhotoQualityReport

router = APIRouter()


@router.get("/")
def list_queue(sort: str = "score", limit: int | None = None) -> list[dict[str, Any]]:
    """Return the full batch by default. `limit` is an optional safety cap;
    the UI paginates client-side (infinite scroll) and filters client-side, so
    no server cap is needed."""
    with session_scope() as sess:
        face_counts: dict[str, int] = {
            h: n
            for h, n in sess.execute(
                select(Face.photo_hash, func.count()).group_by(Face.photo_hash)
            ).all()
        }
        stmt = (
            select(Photo, Decision, PhotoQualityReport)
            .outerjoin(Decision, Photo.hash == Decision.photo_hash)
            .outerjoin(PhotoQualityReport, Photo.hash == PhotoQualityReport.photo_hash)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        items = [
            photo_summary(p, d, qr=qr, n_faces=face_counts.get(p.hash, 0))
            for p, d, qr in sess.execute(stmt).all()
        ]
    if sort == "score":
        items.sort(key=lambda r: (r["technical_score"] or 0.0), reverse=True)
    elif sort == "captured":
        items.sort(key=lambda r: r["captured_at"] or "")
    return items
