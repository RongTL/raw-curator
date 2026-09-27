"""Verification decides whether an enhanced frame is trustworthy enough to replace its RAW."""

from __future__ import annotations

import numpy as np

from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.plan import FaceInfo
from app.enhancement.verify import safe_plan, verify
from tests.test_enhance_job import _report

GOOD = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32) * 0.6 + 0.2


def test_small_quality_drop_is_not_degraded_even_from_a_low_base() -> None:
    v = verify(_report(score_q=20.0), _report(score_q=17.0), GOOD)
    assert v.degraded is False and v.reasons == ()


def test_quality_drop_over_five_is_degraded() -> None:
    v = verify(_report(score_q=80.0), _report(score_q=74.0), GOOD)
    assert v.degraded is True and "q_drop" in v.reasons


def test_new_clipping_is_degraded() -> None:
    v = verify(_report(highlight_clip=0.01), _report(highlight_clip=0.04), GOOD)
    assert "highlight_clip" in v.reasons


def test_collapsed_image_is_degraded_regardless_of_scores() -> None:
    flat = np.full((16, 16, 3), 0.02, dtype=np.float32)
    v = verify(_report(), _report(score_q=99.0), flat)
    assert v.degraded is True and {"mean_luma", "std"} & set(v.reasons)


def test_safe_plan_keeps_only_tone_steps() -> None:
    plan = plan_from_report(
        _report(luma_noise=6.0, lap_var=50.0, highlight_clip=0.05),
        faces=[FaceInfo((0, 0, 120, 120), 40.0)],
    )
    safe = safe_plan(plan)
    names = {s.name for s in safe.steps}
    assert names <= {
        "exposure_gamma",
        "shadow_lift",
        "highlight_recover",
        "backlit_recover",
        "highlight_rolloff",
    }
    assert "highlight_recover" in names and safe.note.startswith("safe retry")
