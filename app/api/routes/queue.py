from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.sql import ColumnElement

from app.api.routes._review_data import enhanced_by_hash, face_counts_by_hash
from app.api.serializers import photo_summary
from app.db import session_scope
from app.models import Decision, Photo

router = APIRouter()

# Review orderings. ``captured`` is the timeline view (oldest first, bursts kept in
# shooting order by path, undated frames last); ``filename`` is the full source
# path, which is also the order ingest walked the files in; ``score`` is the
# technical-quality ranking. Every ordering ends on ``source_path`` so ties are
# deterministic across reloads.
SORT_ORDERS: dict[str, tuple[ColumnElement[Any], ...]] = {
    "captured": (Photo.captured_at.asc().nulls_last(), Photo.source_path.asc()),
    "filename": (Photo.source_path.asc(),),
    "score": (Photo.technical_score.desc().nulls_last(), Photo.source_path.asc()),
}
DEFAULT_SORT = "captured"


@router.get("/")
def list_queue(sort: str = DEFAULT_SORT, limit: int | None = None) -> list[dict[str, Any]]:
    """Return the full batch by default. `limit` is an optional safety cap;
    the UI paginates client-side (infinite scroll) and filters client-side, so
    no server cap is needed."""
    order = SORT_ORDERS.get(sort)
    if order is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown sort {sort!r}; expected one of {sorted(SORT_ORDERS)}",
        )
    with session_scope() as sess:
        face_counts = face_counts_by_hash(sess)
        enhanced = enhanced_by_hash(sess)
        stmt = (
            select(Photo, Decision)
            .outerjoin(Decision, Photo.hash == Decision.photo_hash)
            .order_by(*order)
        )
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
    return items
