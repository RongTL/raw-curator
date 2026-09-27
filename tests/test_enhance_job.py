"""enhance_job glue that can be exercised without darktable or a GPU."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.enhancement.engine.plan import QualityReport
from app.enhancement.enhance_job import persist_report
from app.models import Photo, PhotoQualityReport


def _report(**overrides: float) -> QualityReport:
    base = QualityReport(
        mean_luma=128.0,
        shadow_clip=0.01,
        highlight_clip=0.02,
        midtone_ratio=0.6,
        midtone_deviation=0.0,
        dr_p95_p5=110.0,
        local_dr_mean=90.0,
        rg_ratio=1.0,
        bg_ratio=1.0,
        avg_saturation=0.4,
        oversat_ratio=0.01,
        skin_hue_var=None,
        lap_var=180.0,
        edge_density=0.05,
        hf_energy=120.0,
        luma_noise=1.5,
        chroma_noise=1.0,
        score_exposure=90.0,
        score_dynamic_range=90.0,
        score_color=90.0,
        score_sharpness=88.0,
        score_noise=95.0,
        score_q=90.0,
    )
    return replace(base, **overrides)


def test_persist_report_upserts_one_row_per_photo(tmp_db: Session) -> None:
    tmp_db.add(Photo(hash="h1", source_path="/x/a.cr3"))
    tmp_db.flush()

    persist_report(tmp_db, "h1", _report(score_q=70.0))
    persist_report(tmp_db, "h1", _report(score_q=91.5, skin_hue_var=0.02))
    tmp_db.flush()

    rows = tmp_db.execute(select(PhotoQualityReport)).scalars().all()
    assert len(rows) == 1
    assert rows[0].score_q == 91.5
    assert rows[0].skin_hue_var == 0.02
    assert rows[0].mean_luma == 128.0


def test_persist_plan_stores_steps_as_json(tmp_db: Session) -> None:
    import json

    from app.enhancement.engine.decision import plan_from_report
    from app.enhancement.enhance_job import persist_plan

    tmp_db.add(Photo(hash="h2", source_path="/x/b.cr3"))
    tmp_db.flush()
    report = _report(luma_noise=6.0)
    persist_report(tmp_db, "h2", report)
    persist_plan(tmp_db, "h2", plan_from_report(report))
    tmp_db.flush()
    row = tmp_db.get(PhotoQualityReport, "h2")
    steps = json.loads(row.plan_json)["steps"]
    assert any(s["name"] == "scunet_denoise" for s in steps)
    assert all({"name", "params", "reason"} <= set(s) for s in steps)


def test_persist_report_plan_verdict_share_one_non_autoflush_session(tmp_db: Session) -> None:
    """Regression: report/plan/verdict persist in one autoflush=False session (prod semantics).

    ``session_scope`` uses ``autoflush=False``; a freshly added quality_reports row must be
    flushed by ``persist_report`` so the later ``persist_plan``/``persist_verdict`` gets see it.
    """
    from app.enhancement.engine.decision import plan_from_report
    from app.enhancement.enhance_job import persist_plan, persist_verdict
    from app.enhancement.verify import Verdict

    tmp_db.add(Photo(hash="h3", source_path="/x/c.cr3"))
    tmp_db.commit()

    sess = Session(bind=tmp_db.get_bind(), autoflush=False)
    try:
        report = _report()
        persist_report(sess, "h3", report)
        persist_plan(sess, "h3", plan_from_report(report))
        persist_verdict(sess, "h3", Verdict(False, (), 90.0, 91.0))
        sess.commit()
        row = sess.get(PhotoQualityReport, "h3")
        assert row is not None
        assert row.plan_json is not None
        assert row.verify_json is not None
    finally:
        sess.close()


def test_load_linear_float_reads_16bit_rgb_tiff(tmp_path: Path) -> None:
    import tifffile

    from app.enhancement.enhance_job import _load_linear_float

    tifffile.imwrite(
        tmp_path / "t.tif", np.full((3, 3, 3), 32768, dtype=np.uint16), photometric="rgb"
    )
    out = _load_linear_float(tmp_path / "t.tif")
    assert out.dtype == np.float32 and abs(float(out[0, 0, 0]) - 0.5) < 1e-3


def test_run_enhancement_delegates_to_run_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.enhancement import batch, enhance_job

    photos = [
        (enhance_job.PhotoCandidate("bad", "/x/bad.cr3", "raw", "enhance_only"), []),
        (enhance_job.PhotoCandidate("ok", "/x/ok.cr3", "raw", "enhance_only"), []),
    ]
    expected = enhance_job.EnhanceSummary(enhanced=1, skipped=0, failed=1)
    received: list[object] = []

    def fake_run_batch(items: list[object], **kw: object) -> enhance_job.EnhanceSummary:
        received.append(items)
        return expected

    monkeypatch.setattr(enhance_job, "warmup", lambda: None)
    monkeypatch.setattr(enhance_job, "_candidates", lambda: photos)
    monkeypatch.setattr(batch, "run_batch", fake_run_batch)

    summary = enhance_job.run_enhancement()

    assert summary is expected
    assert received == [photos]


def test_preview_size_missing_file_means_no_faces(tmp_path: Path) -> None:
    from app.enhancement.enhance_job import preview_size

    assert preview_size(tmp_path / "nope.jpg") is None


def test_degraded_result_never_deletes_the_source(tmp_path: Path) -> None:
    from app.enhancement import enhance_job
    from app.enhancement.verify import Verdict

    src = tmp_path / "IMG.CR3"
    src.write_bytes(b"raw")
    out = tmp_path / "IMG.tif"
    out.write_bytes(b"tif")
    photo = enhance_job.PhotoCandidate("h", str(src), "raw", "enhance_only")
    assert (
        enhance_job.may_delete_source(photo, out, Verdict(True, ("q_drop",), 80.0, 70.0)) is False
    )
    assert enhance_job.may_delete_source(photo, out, Verdict(False, (), 80.0, 82.0)) is True
    keep = enhance_job.PhotoCandidate("h", str(src), "raw", "keep_and_enhance")
    assert enhance_job.may_delete_source(keep, out, Verdict(False, (), 80.0, 82.0)) is False
