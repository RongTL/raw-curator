"""One serializer for the photo/decision JSON shape shared by queue, cluster and photo routes."""

from __future__ import annotations

from datetime import datetime

from app.api.serializers import decision_payload, photo_detail, photo_summary
from app.config import settings
from app.models import Decision, Face, Photo


def _photo(**overrides: object) -> Photo:
    base: dict[str, object] = dict(
        hash="abc123",
        source_path="/data/photos/incoming/trip/IMG_0001.CR3",
        thumb_path=f"{settings.cache}/thumbs/abc123.jpg",
        preview_path=f"{settings.cache}/previews/abc123.jpg",
        file_kind="raw",
        captured_at=datetime(2026, 1, 1, 10, 0, 0),
        camera_body="EOS R6",
        technical_score=0.8,
        aesthetic_score=6.5,
        is_recommended=1,
    )
    base.update(overrides)
    return Photo(**base)


def test_decision_payload_none_stays_none() -> None:
    assert decision_payload(None) is None


def test_decision_payload_coerces_int_flags_to_bool() -> None:
    d = Decision(photo_hash="abc123", selected="yes", stars=4, favorite=1, applied=0, action="none")
    out = decision_payload(d)
    assert out == {
        "selected": "yes",
        "stars": 4,
        "favorite": True,
        "applied": False,
        "action": "none",
        "note": None,
    }


def test_photo_summary_shape() -> None:
    out = photo_summary(_photo(), None)
    assert out["hash"] == "abc123"
    assert out["filename"] == "IMG_0001.CR3"
    assert out["thumb_url"] == "/cache/thumbs/abc123.jpg"
    assert out["captured_at"] == "2026-01-01T10:00:00"
    assert out["is_recommended"] is True
    assert out["decision"] is None
    assert out["rank"] is None


def test_photo_summary_carries_rank_and_decision() -> None:
    d = Decision(photo_hash="abc123", selected="no")
    out = photo_summary(_photo(), d, rank=2)
    assert out["rank"] == 2
    assert out["decision"]["selected"] == "no"


def test_photo_detail_extends_summary_with_exif_faces_and_report() -> None:
    p = _photo(iso=400, aperture=2.8)
    faces = [Face(photo_hash="abc123", bbox_x=1, bbox_y=2, bbox_w=3, bbox_h=4, det_score=0.9)]
    out = photo_detail(p, None, faces, None)
    assert out["hash"] == "abc123"
    assert out["preview_url"] == "/cache/previews/abc123.jpg"
    assert out["iso"] == 400 and out["aperture"] == 2.8
    assert out["faces"] == [{"x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9}]
    assert out["quality_report"] is None
    assert out["decision"] is None
