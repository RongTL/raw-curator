"""One serializer for the photo/decision JSON shape shared by queue, cluster and photo routes."""

from __future__ import annotations

from datetime import datetime

from app.api.serializers import decision_payload, photo_detail, photo_summary
from app.config import settings
from app.models import Decision, Face, Photo, PhotoQualityReport


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


def test_decision_payload_shape() -> None:
    d = Decision(
        photo_hash="a" * 32, export_choice="enhanced", keep_raw=0, stars=0, favorite=0, applied=1
    )
    out = decision_payload(d)
    assert out == {
        "export_choice": "enhanced",
        "keep_raw": False,
        "stars": 0,
        "favorite": False,
        "applied": True,
        "note": None,
    }


def test_decision_payload_coerces_int_flags_to_bool() -> None:
    d = Decision(
        photo_hash="abc123", export_choice="enhanced", keep_raw=0, stars=4, favorite=1, applied=0
    )
    out = decision_payload(d)
    assert out == {
        "export_choice": "enhanced",
        "keep_raw": False,
        "stars": 4,
        "favorite": True,
        "applied": False,
        "note": None,
    }


def test_photo_summary_shape() -> None:
    out = photo_summary(_photo(), None)
    assert out["hash"] == "abc123"
    assert out["filename"] == "IMG_0001.CR3"
    assert out["thumb_url"] == "/cache/thumbs/abc123.jpg"
    assert out["preview_url"] == "/cache/previews/abc123.jpg"  # Compare uses the large preview
    assert out["captured_at"] == "2026-01-01T10:00:00"
    assert out["is_recommended"] is True
    assert out["decision"] is None
    assert out["rank"] is None


def test_photo_summary_carries_rank_and_decision() -> None:
    d = Decision(photo_hash="abc123", export_choice="original")
    out = photo_summary(_photo(), d, rank=2)
    assert out["rank"] == 2
    assert out["decision"]["export_choice"] == "original"


def test_photo_summary_has_before_after_urls() -> None:
    out = photo_summary(_photo(), None)
    assert out["before_url"] == "/cache/enhanced/abc123.before.jpg"
    assert out["after_url"] == "/cache/enhanced/abc123.after.jpg"


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


def test_quality_report_payload_exposes_recipe_and_verdict() -> None:
    from app.api.serializers import quality_report_payload

    qr = PhotoQualityReport(
        photo_hash="abc123",
        mean_luma=1,
        shadow_clip=0,
        highlight_clip=0,
        midtone_ratio=0,
        midtone_deviation=0,
        dr_p95_p5=0,
        local_dr_mean=0,
        rg_ratio=1,
        bg_ratio=1,
        avg_saturation=0,
        oversat_ratio=0,
        lap_var=0,
        edge_density=0,
        hf_energy=0,
        luma_noise=0,
        chroma_noise=0,
        score_exposure=0,
        score_dynamic_range=0,
        score_color=0,
        score_sharpness=0,
        score_noise=0,
        score_q=50.0,
        plan_json='{"steps": [{"name": "unsharp_mask", "params": {"amount": 0.4}, "reason": "soft"}]}',
        verify_json='{"degraded": false, "reasons": []}',
        score_q_after=55.0,
        degraded=0,
    )
    out = quality_report_payload(qr)
    assert out["plan"]["steps"][0]["name"] == "unsharp_mask"
    assert out["verify"] == {"degraded": False, "reasons": []}
    assert out["score_q_after"] == 55.0 and out["degraded"] is False


def test_photo_summary_enhanced_fields_present_after_enhance() -> None:
    out = photo_summary(_photo(), None, q_after=88.0, degraded=False, n_faces=2)
    assert out["enhanced"] is True
    assert out["q_after"] == 88.0
    assert out["degraded"] is False
    assert out["n_faces"] == 2


def test_photo_summary_not_enhanced_without_quality_report() -> None:
    out = photo_summary(_photo(), None)
    assert out["enhanced"] is False
    assert out["q_after"] is None
    assert out["degraded"] is False
    assert out["n_faces"] == 0
