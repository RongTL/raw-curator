"""Data structures for the Auto Enhancement Engine.

Three immutable records:
- StepSpec       — one operation the runner will execute, with parameters.
- QualityReport  — the measured input (5 spec dimensions + composite Q).
- EnhancementPlan— the ordered list of steps the decision engine produced
                   from a report, plus the report itself for traceability.

Everything is a frozen dataclass per project coding-style.md (immutable
data, no hidden mutation). Step parameters are plain dicts because the
set of admissible keys varies per step; the runner validates at dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StepName = Literal[
    # Layer 1 — objective corrections (spec §1.5, §2.4, §3, §5)
    "exposure_gamma",         # gamma curve, params: gain (+ brightens, - darkens)
    "shadow_lift",            # luminance-masked shadow lift, params: amount
    "highlight_recover",      # roll off highlights above knee, params: amount, knee
    "backlit_recover",        # bimodal-histogram bilateral lift (existing module)
    "clahe_local_contrast",   # CLAHE in LAB-L, params: clip_limit, tile_grid
    "highlight_rolloff",      # global filmic-style compression on high DR
    "white_balance",          # gray-world correction, params: target_rg, target_bg
    "saturation_adjust",      # HSV saturation scale with skin protection
    # AI primitives (spec §5 noise / §4 sharpness)
    "scunet_denoise",         # params: strength
    "realesrgan_upscale",     # params: fidelity (blend with Lanczos)
    "codeformer_restore",     # params: weight; only when faces present
    # Layer 1 — sharpening (spec §4)
    "unsharp_mask",           # luminance unsharp, params: amount, radius, threshold
    # Layer 1 — final pass (spec §7 last step)
    "tone_map_final",         # filmic curve, params: shoulder, toe
]


@dataclass(frozen=True)
class StepSpec:
    name: StepName
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # human-readable why-we-added-it, surfaced in logs


@dataclass(frozen=True)
class QualityReport:
    """Measured input quality across the five spec dimensions.

    All metrics are computed at full resolution on float32 linear RGB
    in [0, 1]. Sub-scores are 0..100 (higher = better).
    """

    # §1 Exposure
    mean_luma: float           # 0..255 (after sRGB encode for histogram)
    shadow_clip: float         # fraction in [0, 5]
    highlight_clip: float      # fraction in [250, 255]
    midtone_ratio: float       # fraction in [64, 192]
    midtone_deviation: float   # 0..1, distance from center of 50%-75% band

    # §2 Dynamic Range
    dr_p95_p5: float           # 0..255
    local_dr_mean: float       # 0..255, mean over 16x16 blocks

    # §3 Color
    rg_ratio: float
    bg_ratio: float
    avg_saturation: float      # HSV S, mean
    oversat_ratio: float       # fraction with S > 0.85
    skin_hue_var: float | None # YCbCr hue variance over face crops; None if no faces

    # §4 Sharpness
    lap_var: float             # Laplacian variance (8-bit luma)
    edge_density: float        # Canny edge pixel ratio
    hf_energy: float           # mean of |FFT| outside the center quarter

    # §5 Noise
    luma_noise: float          # σ of L flat patches, 8-bit scale
    chroma_noise: float        # σ of (a, b) in LAB

    # Sub-scores (0..100, §1.4 / §6)
    score_exposure: float
    score_dynamic_range: float
    score_color: float
    score_sharpness: float
    score_noise: float
    score_q: float             # weighted composite


@dataclass(frozen=True)
class EnhancementPlan:
    steps: tuple[StepSpec, ...]
    report: QualityReport
    has_faces: bool
    note: str = ""
