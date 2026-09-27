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
    8. (tone mapping is darktable's job; not planned here)

Quality-first defaults (matched to a RTX 2060 6 GB / R3 3100 24 GB box):
- AI steps are considered when their indicator metric is above the spec
  threshold. Strength scales with the measured deficit so clean inputs
  are not over-processed.
- Classical step parameters are picked from the spec's recommended bands.
- Plan never includes both `global_compress` and `clahe_local_contrast`;
  the §2.4 high-DR branch picks one path.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.enhancement.engine.plan import EnhancementPlan, FaceInfo, QualityReport, StepSpec


def _strength_from_deficit(metric: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if metric >= high else 0.0
    return max(0.0, min(1.0, (metric - low) / (high - low)))


def plan_from_report(
    report: QualityReport,
    *,
    faces: Sequence[FaceInfo] = (),
    denoise: bool = True,
    face_restore: bool = True,
    backlit_recovery: bool = True,
    iso: int | None = None,
    native_long_edge: int | None = None,
    target_scale: float = 1.0,
    sr_min_long_edge: int = 3000,
    enhance_codeformer_w: float = 0.85,
    enhance_realesrgan_fidelity: float = 0.7,
    enhance_denoise_strength: float = 0.75,
    backlit_shadow_lift: float = 0.4,
    backlit_highlight_protect: float = 0.15,
    face_restore_max_px: int = 300,
) -> EnhancementPlan:
    has_faces = bool(faces)
    steps: list[StepSpec] = []

    # §1 Exposure (also runs the existing backlit detector if bimodal)
    if backlit_recovery and report.shadow_clip >= 0.18 and report.highlight_clip >= 0.10:
        steps.append(
            StepSpec(
                name="backlit_recover",
                params={
                    "shadow_lift": backlit_shadow_lift,
                    "highlight_protect": backlit_highlight_protect,
                    "force": False,
                },
                reason=f"shadow_clip={report.shadow_clip:.3f}, "
                f"highlight_clip={report.highlight_clip:.3f}",
            )
        )
    if report.mean_luma < 90.0:
        gain = min(0.6, (90.0 - report.mean_luma) / 90.0 * 0.8)
        steps.append(
            StepSpec(
                name="exposure_gamma",
                params={"gain": gain},
                reason=f"underexposed: mean_luma={report.mean_luma:.1f}",
            )
        )
    elif report.mean_luma > 200.0:
        gain = -min(0.4, (report.mean_luma - 200.0) / 55.0 * 0.5)
        steps.append(
            StepSpec(
                name="exposure_gamma",
                params={"gain": gain},
                reason=f"overexposed: mean_luma={report.mean_luma:.1f}",
            )
        )
    if report.shadow_clip > 0.02 and report.mean_luma >= 90.0:
        amt = _strength_from_deficit(report.shadow_clip, 0.02, 0.15) * 0.5
        steps.append(
            StepSpec(
                name="shadow_lift",
                params={"amount": amt},
                reason=f"shadow_clip={report.shadow_clip:.3f}",
            )
        )
    if report.highlight_clip > 0.01:
        amt = _strength_from_deficit(report.highlight_clip, 0.01, 0.10)
        steps.append(
            StepSpec(
                name="highlight_recover",
                params={"amount": amt, "knee": 0.78},
                reason=f"highlight_clip={report.highlight_clip:.3f}",
            )
        )

    # §2 Dynamic Range
    if report.dr_p95_p5 > 150.0:
        amt = _strength_from_deficit(report.dr_p95_p5, 150.0, 220.0)
        steps.append(
            StepSpec(
                name="highlight_rolloff",
                params={"strength": 0.3 + 0.4 * amt},
                reason=f"dr_p95_p5={report.dr_p95_p5:.1f} (excessive)",
            )
        )

    # §3 Color — white balance is estimated from near-neutral pixels only and
    # capped at half strength, so warm sunsets / tungsten interiors keep their
    # mood instead of being neutralised on every frame (Task 22).
    if (
        report.neutral_fraction >= 0.02
        and report.rg_neutral is not None
        and report.bg_neutral is not None
    ):
        cast = max(abs(report.rg_neutral - 1.0), abs(report.bg_neutral - 1.0))
        if cast > 0.12:
            steps.append(
                StepSpec(
                    name="white_balance",
                    params={
                        "target_rg": 1.0,
                        "target_bg": 1.0,
                        "strength": min(0.5, (cast - 0.12) * 2.5),
                        "neutral_only": True,
                    },
                    reason=f"neutral cast rg={report.rg_neutral:.3f} "
                    f"bg={report.bg_neutral:.3f} over {report.neutral_fraction:.0%}",
                )
            )
    if report.oversat_ratio > 0.05 or report.avg_saturation > 0.55:
        excess = max(
            report.oversat_ratio - 0.05,
            (report.avg_saturation - 0.55) if report.avg_saturation > 0.55 else 0.0,
        )
        factor = max(0.7, 1.0 - excess * 1.5)
        steps.append(
            StepSpec(
                name="saturation_adjust",
                params={"factor": factor, "protect_skin": True},
                reason=f"avg_sat={report.avg_saturation:.2f}, oversat={report.oversat_ratio:.3f}",
            )
        )
    elif report.avg_saturation < 0.25 and (
        report.mean_chroma is None or report.mean_chroma >= 0.01
    ):
        # Skip the boost on a monochrome frame (mean OKLCh chroma < 0.01): a
        # black-and-white or near-neutral image is an intent, not a defect.
        boost = min(1.25, 1.0 + (0.25 - report.avg_saturation) * 1.0)
        steps.append(
            StepSpec(
                name="saturation_adjust",
                params={"factor": boost, "protect_skin": True},
                reason=f"undersaturated: avg_sat={report.avg_saturation:.2f}",
            )
        )

    # §5 Noise (must come before §4 sharpening per §7)
    n = max(report.luma_noise, report.chroma_noise * 0.5)
    noisy = n > 4.0 or (n > 2.0 and (iso is None or iso >= 800))
    if denoise and noisy:
        strength = max(enhance_denoise_strength * 0.6, min(0.95, 0.5 + (n - 2.0) * 0.05))
        steps.append(
            StepSpec(
                name="scunet_denoise",
                params={"strength": strength},
                reason=f"luma_noise={report.luma_noise:.2f}, chroma_noise={report.chroma_noise:.2f}",
            )
        )

    # §4 Sharpness — Real-ESRGAN runs only when enlarging (target > native, D3) or the
    # source is small enough that x2 recovers real detail rather than upsampling noise.
    needs_sr = target_scale > 1.0 or (
        native_long_edge is not None and native_long_edge < sr_min_long_edge
    )
    if needs_sr:
        reason = (
            f"enlarging x{target_scale:.2f}"
            if target_scale > 1.0
            else f"long edge {native_long_edge} < {sr_min_long_edge}"
        )
        steps.append(
            StepSpec(
                name="realesrgan_upscale",
                params={"fidelity": enhance_realesrgan_fidelity},
                reason=reason,
            )
        )
    # Restore only faces that are small, soft, or sitting in noise; a sharp,
    # large, clean face is left alone so CodeFormer cannot rebuild it worse.
    degraded_faces = [
        f
        for f in faces
        if max(f.box[2], f.box[3]) < face_restore_max_px or f.lap_var < 100.0 or noisy
    ]
    if face_restore and degraded_faces:
        steps.append(
            StepSpec(
                name="codeformer_restore",
                params={"weight": enhance_codeformer_w},
                reason=f"{len(degraded_faces)} of {len(faces)} faces small/soft",
            )
        )
    # Sharpen on the sharpest region's variance when available, so a crisp subject
    # against a creamy bokeh background isn't unsharped just because the whole-frame
    # variance is low.
    sharp_metric = "lap_var_top" if report.lap_var_top is not None else "lap_var"
    sharp_var = report.lap_var_top if report.lap_var_top is not None else report.lap_var
    if sharp_var < 150.0:
        amt = _strength_from_deficit(150.0 - sharp_var, 0.0, 120.0) * 0.8 + 0.2
        steps.append(
            StepSpec(
                name="unsharp_mask",
                params={"amount": amt, "radius": 1.4, "threshold": 0.006},
                reason=f"{sharp_metric}={sharp_var:.1f}",
            )
        )

    # §7.7 Local contrast — skipped entirely when a face is present (CLAHE
    # harshens skin) and the clip capped at 2.0 so it stays a gentle polish.
    if not faces and report.dr_p95_p5 < 150.0 and report.local_dr_mean < 80.0:
        clip = min(2.0, 1.6 + _strength_from_deficit(80.0 - report.local_dr_mean, 0.0, 50.0) * 1.4)
        steps.append(
            StepSpec(
                name="clahe_local_contrast",
                params={"clip_limit": clip, "tile_grid": (8, 8)},
                reason=f"local_dr_mean={report.local_dr_mean:.1f} (flat)",
            )
        )
    elif not faces and report.dr_p95_p5 < 150.0:
        steps.append(
            StepSpec(
                name="clahe_local_contrast",
                params={"clip_limit": 1.5, "tile_grid": (8, 8)},
                reason="default local-contrast polish",
            )
        )

    note = f"Q={report.score_q:.1f}; steps={len(steps)}"
    return EnhancementPlan(steps=tuple(steps), report=report, has_faces=has_faces, note=note)
