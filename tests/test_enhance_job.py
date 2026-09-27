"""enhance_job glue that can be exercised without darktable or a GPU."""

from __future__ import annotations

from dataclasses import replace

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
