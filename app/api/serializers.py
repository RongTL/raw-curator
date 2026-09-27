"""ORM -> JSON shapes for the review API.

Three routes (queue, cluster, photo) used to hand-build overlapping dicts;
this module is the one place the photo/decision payload is defined.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

from app.api.routes._urls import cache_url
from app.enhancement.engine.plan import QualityReport
from app.models import Decision, Face, Photo, PhotoQualityReport

_REPORT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(QualityReport))


def decision_payload(d: Decision | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {
        "selected": d.selected,
        "stars": d.stars,
        "favorite": bool(d.favorite),
        "applied": bool(d.applied),
        "action": d.action,
        "note": d.note,
    }


def photo_summary(p: Photo, d: Decision | None, *, rank: int | None = None) -> dict[str, Any]:
    """The tile-level shape used by the queue grid and cluster sections."""
    return {
        "hash": p.hash,
        "filename": Path(p.source_path).name if p.source_path else None,
        "file_kind": p.file_kind,
        "thumb_url": cache_url(p.thumb_path),
        "captured_at": p.captured_at.isoformat() if p.captured_at else None,
        "camera_body": p.camera_body,
        "blur_var": p.blur_var,
        "aesthetic_score": p.aesthetic_score,
        "technical_score": p.technical_score,
        "cluster_id": p.cluster_id,
        "is_recommended": bool(p.is_recommended),
        "rank": rank,
        "decision": decision_payload(d),
    }


def quality_report_payload(qr: PhotoQualityReport | None) -> dict[str, Any] | None:
    if qr is None:
        return None
    out: dict[str, Any] = {name: getattr(qr, name) for name in _REPORT_FIELDS}
    out["measured_at"] = qr.measured_at.isoformat() if qr.measured_at else None
    out["plan"] = json.loads(qr.plan_json) if qr.plan_json else None
    out["verify"] = json.loads(qr.verify_json) if qr.verify_json else None
    out["score_q_after"] = qr.score_q_after
    out["degraded"] = bool(qr.degraded)
    return out


def photo_detail(
    p: Photo,
    d: Decision | None,
    faces: Iterable[Face],
    qr: PhotoQualityReport | None,
) -> dict[str, Any]:
    """Everything the detail modal shows: summary + EXIF + scores + faces + engine report."""
    out = photo_summary(p, d)
    out.update(
        {
            "source_path": p.source_path,
            "preview": p.preview_path,
            "thumb": p.thumb_path,
            "preview_url": cache_url(p.preview_path),
            "lens": p.lens,
            "iso": p.iso,
            "shutter": p.shutter,
            "aperture": p.aperture,
            "focal_length": p.focal_length,
            "width": p.width,
            "height": p.height,
            "phash": p.phash,
            "exposure_flag": p.exposure_flag,
            "musiq_score": p.musiq_score,
            "maniqa_score": p.maniqa_score,
            "faces": [
                {"x": f.bbox_x, "y": f.bbox_y, "w": f.bbox_w, "h": f.bbox_h, "score": f.det_score}
                for f in faces
            ],
            "quality_report": quality_report_payload(qr),
        }
    )
    return out
