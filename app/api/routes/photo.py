from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes._urls import cache_url
from app.db import session_scope
from app.models import Decision, Face, Photo

router = APIRouter()


@router.get("/{photo_hash}")
def get_photo(photo_hash: str) -> dict:
    with session_scope() as sess:
        photo = sess.get(Photo, photo_hash)
        if not photo:
            raise HTTPException(status_code=404, detail="photo not found")
        faces = list(sess.execute(select(Face).where(Face.photo_hash == photo_hash)).scalars())
        decision = sess.get(Decision, photo_hash)
        return {
            "hash": photo.hash,
            "filename": Path(photo.source_path).name if photo.source_path else None,
            "source_path": photo.source_path,
            "preview": photo.preview_path,
            "thumb": photo.thumb_path,
            "preview_url": cache_url(photo.preview_path),
            "thumb_url": cache_url(photo.thumb_path),
            "captured_at": photo.captured_at.isoformat() if photo.captured_at else None,
            "camera_body": photo.camera_body,
            "lens": photo.lens,
            "iso": photo.iso,
            "shutter": photo.shutter,
            "aperture": photo.aperture,
            "focal_length": photo.focal_length,
            "width": photo.width,
            "height": photo.height,
            "blur_var": photo.blur_var,
            "phash": photo.phash,
            "exposure_flag": photo.exposure_flag,
            "aesthetic_score": photo.aesthetic_score,
            "technical_score": photo.technical_score,
            "musiq_score": photo.musiq_score,
            "maniqa_score": photo.maniqa_score,
            "cluster_id": photo.cluster_id,
            "is_recommended": bool(photo.is_recommended),
            "faces": [
                {"x": f.bbox_x, "y": f.bbox_y, "w": f.bbox_w, "h": f.bbox_h, "score": f.det_score}
                for f in faces
            ],
            "decision": (
                {
                    "selected": decision.selected,
                    "score_tier": decision.score_tier,
                    "stars": decision.stars,
                    "favorite": bool(decision.favorite),
                    "enhance_requested": bool(decision.enhance_requested),
                    "action": decision.action,
                    "applied": bool(decision.applied),
                    "note": decision.note,
                }
                if decision
                else None
            ),
        }
