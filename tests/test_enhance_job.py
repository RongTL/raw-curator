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


def test_load_linear_float_reads_16bit_rgb_tiff(tmp_path: Path) -> None:
    import tifffile

    from app.enhancement.enhance_job import _load_linear_float

    tifffile.imwrite(
        tmp_path / "t.tif", np.full((3, 3, 3), 32768, dtype=np.uint16), photometric="rgb"
    )
    out = _load_linear_float(tmp_path / "t.tif")
    assert out.dtype == np.float32 and abs(float(out[0, 0, 0]) - 0.5) < 1e-3


def test_run_enhancement_keeps_going_after_one_photo_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.enhancement import enhance_job

    photos = [
        (enhance_job.PhotoCandidate("bad", "/x/bad.cr3", "raw", "enhance_only"), []),
        (enhance_job.PhotoCandidate("ok", "/x/ok.cr3", "raw", "enhance_only"), []),
    ]
    seen: list[str] = []

    def fake_enhance_one(photo: enhance_job.PhotoCandidate, faces: list[object]) -> Path | None:
        seen.append(photo.hash)
        if photo.hash == "bad":
            raise RuntimeError("darktable-cli exploded")
        return tmp_path / "ok.tif"

    monkeypatch.setattr(enhance_job, "warmup", lambda: None)
    monkeypatch.setattr(enhance_job, "_candidates", lambda: photos)
    monkeypatch.setattr(enhance_job, "_enhance_one", fake_enhance_one)

    summary = enhance_job.run_enhancement()

    assert seen == ["bad", "ok"]
    assert (summary.enhanced, summary.skipped, summary.failed) == (1, 0, 1)


def test_preview_size_missing_file_means_no_faces(tmp_path: Path) -> None:
    from app.enhancement.enhance_job import preview_size

    assert preview_size(tmp_path / "nope.jpg") is None
