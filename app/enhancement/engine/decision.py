"""Translate a QualityReport into an ordered EnhancementPlan.

Implements the "if X then Y" tables in spec §1.5 (exposure), §2.4
(dynamic range), §3 (color), §4 (sharpness), §5 (noise). The output
list is ordered per spec §7:

    1. Exposure normalisation
    2. Dynamic-range recovery
    3. White balance correction
    4. Color correction
    5. Noise reduction         (AI: SCUNet)
    6. Sharpening              (AI: Real-ESRGAN + classical unsharp)
    7. Local contrast enhancement (CLAHE)
    8. Final tone mapping

Quality-first defaults (matched to a RTX 2060 6 GB / R3 3100 24 GB box):
- AI steps are considered when their indicator metric is above the spec
  threshold. Strength scales with the measured deficit so clean inputs
  are not over-processed.
- Classical step parameters are picked from the spec's recommended bands.
- Plan never includes both `global_compress` and `clahe_local_contrast`;
  the §2.4 high-DR branch picks one path.
"""

from __future__ import annotations

from app.enhancement.engine.plan import EnhancementPlan, QualityReport, StepSpec


def _strength_from_deficit(metric: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if metric >= high else 0.0
    return max(0.0, min(1.0, (metric - low) / (high - low)))


def plan_from_report(
    report: QualityReport,
    has_faces: bool = False,
    enhance_codeformer_w: float = 0.85,
    enhance_realesrgan_fidelity: float = 0.7,
    enhance_denoise_strength: float = 0.75,
    backlit_shadow_lift: float = 0.4,
    backlit_highlight_protect: float = 0.15,
) -> EnhancementPlan:
    steps: list[StepSpec] = []

    # §1 Exposure (also runs the existing backlit detector if bimodal)
    if report.shadow_clip >= 0.18 and report.highlight_clip >= 0.10:
        steps.append(StepSpec(
            name="backlit_recover",
            params={"shadow_lift": backlit_shadow_lift,
                    "highlight_protect": backlit_highlight_protect,
                    "force": False},
            reason=f"shadow_clip={report.shadow_clip:.3f}, "
                   f"highlight_clip={report.highlight_clip:.3f}",
        ))
    if report.mean_luma < 90.0:
        gain = min(0.6, (90.0 - report.mean_luma) / 90.0 * 0.8)
        steps.append(StepSpec(
            name="exposure_gamma",
            params={"gain": gain},
            reason=f"underexposed: mean_luma={report.mean_luma:.1f}",
        ))
    elif report.mean_luma > 200.0:
        gain = -min(0.4, (report.mean_luma - 200.0) / 55.0 * 0.5)
        steps.append(StepSpec(
            name="exposure_gamma",
            params={"gain": gain},
            reason=f"overexposed: mean_luma={report.mean_luma:.1f}",
        ))
    if report.shadow_clip > 0.02 and report.mean_luma >= 90.0:
        amt = _strength_from_deficit(report.shadow_clip, 0.02, 0.15) * 0.5
        steps.append(StepSpec(
            name="shadow_lift",
            params={"amount": amt},
            reason=f"shadow_clip={report.shadow_clip:.3f}",
        ))
    if report.highlight_clip > 0.01:
        amt = _strength_from_deficit(report.highlight_clip, 0.01, 0.10)
        steps.append(StepSpec(
            name="highlight_recover",
            params={"amount": amt, "knee": 0.78},
            reason=f"highlight_clip={report.highlight_clip:.3f}",
        ))

    # §2 Dynamic Range
    if report.dr_p95_p5 > 150.0:
        amt = _strength_from_deficit(report.dr_p95_p5, 150.0, 220.0)
        steps.append(StepSpec(
            name="highlight_rolloff",
            params={"strength": 0.3 + 0.4 * amt},
            reason=f"dr_p95_p5={report.dr_p95_p5:.1f} (excessive)",
        ))

    # §3 Color
    cast = max(abs(report.rg_ratio - 1.0), abs(report.bg_ratio - 1.0))
    if cast > 0.05:
        strength = min(0.85, (cast - 0.05) * 5.0)
        steps.append(StepSpec(
            name="white_balance",
            params={"target_rg": 1.0, "target_bg": 1.0, "strength": strength},
            reason=f"rg={report.rg_ratio:.3f}, bg={report.bg_ratio:.3f}",
        ))
    if report.oversat_ratio > 0.05 or report.avg_saturation > 0.55:
        excess = max(
            report.oversat_ratio - 0.05,
            (report.avg_saturation - 0.55) if report.avg_saturation > 0.55 else 0.0,
        )
        factor = max(0.7, 1.0 - excess * 1.5)
        steps.append(StepSpec(
            name="saturation_adjust",
            params={"factor": factor, "protect_skin": True},
            reason=f"avg_sat={report.avg_saturation:.2f}, oversat={report.oversat_ratio:.3f}",
        ))
    elif report.avg_saturation < 0.25:
        boost = min(1.25, 1.0 + (0.25 - report.avg_saturation) * 1.0)
        steps.append(StepSpec(
            name="saturation_adjust",
            params={"factor": boost, "protect_skin": True},
            reason=f"undersaturated: avg_sat={report.avg_saturation:.2f}",
        ))

    # §5 Noise (must come before §4 sharpening per §7)
    n = max(report.luma_noise, report.chroma_noise * 0.5)
    if n > 2.0:
        strength = max(enhance_denoise_strength * 0.6, min(0.95, 0.5 + (n - 2.0) * 0.05))
        steps.append(StepSpec(
            name="scunet_denoise",
            params={"strength": strength},
            reason=f"luma_noise={report.luma_noise:.2f}, chroma_noise={report.chroma_noise:.2f}",
        ))

    # §4 Sharpness — Real-ESRGAN always runs (recovers detail after VRAM downscale)
    steps.append(StepSpec(
        name="realesrgan_upscale",
        params={"fidelity": enhance_realesrgan_fidelity},
        reason="default detail-recovery x2",
    ))
    if has_faces:
        steps.append(StepSpec(
            name="codeformer_restore",
            params={"weight": enhance_codeformer_w},
            reason="faces detected",
        ))
    if report.lap_var < 150.0:
        amt = _strength_from_deficit(150.0 - report.lap_var, 0.0, 120.0) * 0.8 + 0.2
        steps.append(StepSpec(
            name="unsharp_mask",
            params={"amount": amt, "radius": 1.4, "threshold": 0.006},
            reason=f"lap_var={report.lap_var:.1f}",
        ))

    # §7.7 Local contrast
    if report.dr_p95_p5 < 150.0 and report.local_dr_mean < 80.0:
        clip = 1.6 + _strength_from_deficit(80.0 - report.local_dr_mean, 0.0, 50.0) * 1.4
        steps.append(StepSpec(
            name="clahe_local_contrast",
            params={"clip_limit": clip, "tile_grid": (8, 8)},
            reason=f"local_dr_mean={report.local_dr_mean:.1f} (flat)",
        ))
    elif report.dr_p95_p5 < 150.0:
        steps.append(StepSpec(
            name="clahe_local_contrast",
            params={"clip_limit": 1.5, "tile_grid": (8, 8)},
            reason="default local-contrast polish",
        ))

    # §7.8 Final tone map
    steps.append(StepSpec(
        name="tone_map_final",
        params={"shoulder": 0.88, "toe": 0.02},
        reason="final filmic polish",
    ))

    note = f"Q={report.score_q:.1f}; steps={len(steps)}"
    return EnhancementPlan(steps=tuple(steps), report=report, has_faces=has_faces, note=note)
