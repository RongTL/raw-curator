from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.routes._urls import cache_url
from app.db import session_scope
from app.models import Decision, Photo

router = APIRouter()


@router.get("/")
def list_queue(sort: str = "score", limit: int = 500) -> list[dict]:
    items: list[dict] = []
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo, Decision)
            .outerjoin(Decision, Photo.hash == Decision.photo_hash)
            .limit(limit)
        ).all()
        for p, d in rows:
            items.append(
                {
                    "hash": p.hash,
                    "thumb_url": cache_url(p.thumb_path),
                    "captured_at": p.captured_at.isoformat() if p.captured_at else None,
                    "camera_body": p.camera_body,
                    "blur_var": p.blur_var,
                    "aesthetic_score": p.aesthetic_score,
                    "technical_score": p.technical_score,
                    "cluster_id": p.cluster_id,
                    "is_recommended": bool(p.is_recommended),
                    "decision": (
                        {
                            "selected": d.selected,
                            "score_tier": d.score_tier,
                            "stars": d.stars,
                            "favorite": bool(d.favorite),
                            "enhance_requested": bool(d.enhance_requested),
                            "applied": bool(d.applied),
                        }
                        if d
                        else None
                    ),
                }
            )
    if sort == "score":
        items.sort(key=lambda r: (r["technical_score"] or 0.0), reverse=True)
    elif sort == "captured":
        items.sort(key=lambda r: r["captured_at"] or "")
    return items
