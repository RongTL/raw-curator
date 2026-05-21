from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes._urls import cache_url
from app.db import session_scope
from app.models import Decision, Face, Photo, PhotoQualityReport

router = APIRouter()


@router.get("/{photo_hash}")
def get_photo(photo_hash: str) -> dict:
    with session_scope() as sess:
        photo = sess.get(Photo, photo_hash)
        if not photo:
            raise HTTPException(status_code=404, detail="photo not found")
        faces = list(sess.execute(select(Face).where(Face.photo_hash == photo_hash)).scalars())
        decision = sess.get(Decision, photo_hash)
        qr = sess.get(PhotoQualityReport, photo_hash)
        return {
            "hash": photo.hash,
            "filename": Path(photo.source_path).name if photo.source_path else None,
            "source_path": photo.source_path,
            "file_kind": photo.file_kind,
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
                    "stars": decision.stars,
                    "favorite": bool(decision.favorite),
                    "action": decision.action,
                    "applied": bool(decision.applied),
                    "note": decision.note,
                }
                if decision
                else None
            ),
            "quality_report": (
                {
                    "score_q": qr.score_q,
                    "score_exposure": qr.score_exposure,
                    "score_dynamic_range": qr.score_dynamic_range,
                    "score_color": qr.score_color,
                    "score_sharpness": qr.score_sharpness,
                    "score_noise": qr.score_noise,
                    "mean_luma": qr.mean_luma,
                    "shadow_clip": qr.shadow_clip,
                    "highlight_clip": qr.highlight_clip,
                    "midtone_ratio": qr.midtone_ratio,
                    "midtone_deviation": qr.midtone_deviation,
                    "dr_p95_p5": qr.dr_p95_p5,
                    "local_dr_mean": qr.local_dr_mean,
                    "rg_ratio": qr.rg_ratio,
                    "bg_ratio": qr.bg_ratio,
                    "avg_saturation": qr.avg_saturation,
                    "oversat_ratio": qr.oversat_ratio,
                    "skin_hue_var": qr.skin_hue_var,
                    "lap_var": qr.lap_var,
                    "edge_density": qr.edge_density,
                    "hf_energy": qr.hf_energy,
                    "luma_noise": qr.luma_noise,
                    "chroma_noise": qr.chroma_noise,
                    "measured_at": qr.measured_at.isoformat() if qr.measured_at else None,
                }
                if qr
                else None
            ),
        }
