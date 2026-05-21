"""Unit tests for the Auto Enhancement Engine decision layer.

Each test constructs a QualityReport biased toward one deficiency and
asserts the planner adds (or omits) the corresponding step.
"""

from __future__ import annotations

from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.plan import QualityReport


def _baseline(**overrides) -> QualityReport:
    base = dict(
        mean_luma=128.0,
        shadow_clip=0.0,
        highlight_clip=0.0,
        midtone_ratio=0.60,
        midtone_deviation=0.0,
        dr_p95_p5=110.0,
        local_dr_mean=95.0,
        rg_ratio=1.0,
        bg_ratio=1.0,
        avg_saturation=0.40,
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
    base.update(overrides)
    return QualityReport(**base)


def _step_names(plan) -> list[str]:
    return [s.name for s in plan.steps]


def test_balanced_input_plan_is_minimal() -> None:
    plan = plan_from_report(_baseline(), has_faces=False)
    names = _step_names(plan)
    assert "realesrgan_upscale" in names
    assert names[-1] == "tone_map_final"
    assert "exposure_gamma" not in names
    assert "shadow_lift" not in names


def test_underexposed_triggers_gamma_lift() -> None:
    plan = plan_from_report(_baseline(mean_luma=60.0))
    assert "exposure_gamma" in _step_names(plan)


def test_blown_highlights_triggers_recovery() -> None:
    plan = plan_from_report(_baseline(highlight_clip=0.06))
    assert "highlight_recover" in _step_names(plan)


def test_backlit_signature_triggers_backlit_recover() -> None:
    plan = plan_from_report(_baseline(shadow_clip=0.22, highlight_clip=0.12))
    assert "backlit_recover" in _step_names(plan)


def test_high_dr_triggers_rolloff_not_clahe() -> None:
    plan = plan_from_report(_baseline(dr_p95_p5=200.0))
    names = _step_names(plan)
    assert "highlight_rolloff" in names
    assert "clahe_local_contrast" not in names


def test_color_cast_triggers_white_balance() -> None:
    plan = plan_from_report(_baseline(rg_ratio=1.20, bg_ratio=0.85))
    assert "white_balance" in _step_names(plan)


def test_oversaturated_triggers_desaturate() -> None:
    plan = plan_from_report(_baseline(oversat_ratio=0.15, avg_saturation=0.65))
    assert "saturation_adjust" in _step_names(plan)


def test_undersaturated_triggers_boost() -> None:
    plan = plan_from_report(_baseline(avg_saturation=0.15))
    assert "saturation_adjust" in _step_names(plan)


def test_noisy_input_triggers_scunet() -> None:
    plan = plan_from_report(_baseline(luma_noise=6.0))
    assert "scunet_denoise" in _step_names(plan)


def test_clean_input_skips_scunet() -> None:
    plan = plan_from_report(_baseline(luma_noise=1.0, chroma_noise=0.5))
    assert "scunet_denoise" not in _step_names(plan)


def test_blurry_input_triggers_unsharp() -> None:
    plan = plan_from_report(_baseline(lap_var=40.0))
    assert "unsharp_mask" in _step_names(plan)


def test_faces_trigger_codeformer() -> None:
    plan = plan_from_report(_baseline(), has_faces=True)
    assert "codeformer_restore" in _step_names(plan)


def test_no_faces_no_codeformer() -> None:
    plan = plan_from_report(_baseline(), has_faces=False)
    assert "codeformer_restore" not in _step_names(plan)


def test_plan_order_matches_spec_section_7() -> None:
    """All pre-AI steps precede AI steps, which precede post-AI."""
    pre = {
        "exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover",
        "highlight_rolloff", "white_balance", "saturation_adjust",
    }
    ai = {"scunet_denoise", "realesrgan_upscale", "codeformer_restore"}
    post = {"unsharp_mask", "clahe_local_contrast", "tone_map_final"}

    plan = plan_from_report(_baseline(
        mean_luma=70.0, highlight_clip=0.05, oversat_ratio=0.10,
        avg_saturation=0.60, luma_noise=5.0, lap_var=40.0,
    ), has_faces=True)
    names = _step_names(plan)

    seen_ai = False
    seen_post = False
    for name in names:
        if name in pre:
            assert not seen_ai, f"pre-AI {name} after AI"
            assert not seen_post, f"pre-AI {name} after post-AI"
        elif name in ai:
            assert not seen_post, f"AI {name} after post-AI"
            seen_ai = True
        elif name in post:
            seen_post = True
        else:
            raise AssertionError(f"unknown step kind: {name}")
