from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes._review_data import quality_report_or_none
from app.api.serializers import photo_detail
from app.db import session_scope
from app.models import Decision, Face, Photo

router = APIRouter()


@router.get("/{photo_hash}")
def get_photo(photo_hash: str) -> dict[str, Any]:
    with session_scope() as sess:
        photo = sess.get(Photo, photo_hash)
        if not photo:
            raise HTTPException(status_code=404, detail="photo not found")
        faces = sess.execute(select(Face).where(Face.photo_hash == photo_hash)).scalars().all()
        return photo_detail(
            photo,
            sess.get(Decision, photo_hash),
            faces,
            quality_report_or_none(sess, photo_hash),
        )
