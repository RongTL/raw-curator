"""Sub-scores + composite Q per spec §1.4 / §6.

All scores are in [0, 100]; higher = better. The weighting for Q is the
spec default:

    Q = 0.25*E + 0.20*D + 0.25*C + 0.15*S + 0.15*N

Each sub-score is shaped so that a typical "good" photo lands around 85
and a typical "bad" photo around 30, with the spec's recommended
thresholds anchoring the curve.
"""

from __future__ import annotations

from app.enhancement.engine.plan import QualityReport

# §1.4 exposure-score weights for clipping/midtone penalties.
_W_SHADOW = 1500.0    # 1% clip -> -15
_W_HIGHLIGHT = 1500.0
_W_MIDTONE = 30.0     # deviation in [0,1] -> up to -30


def _score_exposure(m: dict) -> float:
    sc = m["shadow_clip"]
    hc = m["highlight_clip"]
    md = m["midtone_deviation"]
    penalty = _W_SHADOW * sc + _W_HIGHLIGHT * hc + _W_MIDTONE * md
    mean = m["mean_luma"]
    if mean < 90.0:
        penalty += min(25.0, (90.0 - mean) * 0.3)
    elif mean > 170.0:
        penalty += min(25.0, (mean - 170.0) * 0.3)
    return max(0.0, 100.0 - penalty)


def _score_dynamic_range(m: dict) -> float:
    dr = m["dr_p95_p5"]
    if dr < 60.0:
        return 100.0 * (dr / 60.0) * 0.6
    if dr < 80.0:
        return 60.0 + 20.0 * (dr - 60.0) / 20.0
    if dr <= 150.0:
        return 90.0 + 10.0 * (1.0 - abs(dr - 115.0) / 35.0)
    return max(70.0, 100.0 - (dr - 150.0) * 0.3)


def _score_color(m: dict) -> float:
    rg = m["rg_ratio"]
    bg = m["bg_ratio"]
    cast = max(abs(rg - 1.0), abs(bg - 1.0))
    cast_penalty = max(0.0, (cast - 0.05) * 400.0)
    s = m["avg_saturation"]
    if s < 0.25:
        sat_penalty = (0.25 - s) * 100.0
    elif s > 0.55:
        sat_penalty = (s - 0.55) * 100.0
    else:
        sat_penalty = 0.0
    over = m["oversat_ratio"]
    over_penalty = max(0.0, (over - 0.05) * 200.0)
    return max(0.0, 100.0 - cast_penalty - sat_penalty - over_penalty)


def _score_sharpness(m: dict) -> float:
    v = m["lap_var"]
    if v >= 150.0:
        return max(75.0, 100.0 - max(0.0, (v - 400.0) * 0.05))
    if v >= 50.0:
        return 50.0 + 50.0 * (v - 50.0) / 100.0
    return max(0.0, 50.0 * v / 50.0)


def _score_noise(m: dict) -> float:
    lum = m["luma_noise"]
    chr_ = m["chroma_noise"]
    n = max(lum, chr_ * 0.5)
    if n <= 2.0:
        return 100.0 - n * 2.5
    if n <= 5.0:
        return 95.0 - (n - 2.0) * 10.0
    if n <= 10.0:
        return 65.0 - (n - 5.0) * 8.0
    return max(0.0, 25.0 - (n - 10.0) * 2.0)


def score_report(metrics: dict) -> QualityReport:
    e = _score_exposure(metrics)
    d = _score_dynamic_range(metrics)
    c = _score_color(metrics)
    s = _score_sharpness(metrics)
    n = _score_noise(metrics)
    q = 0.25 * e + 0.20 * d + 0.25 * c + 0.15 * s + 0.15 * n
    skin = metrics.get("skin_hue_var")
    return QualityReport(
        mean_luma=float(metrics["mean_luma"]),
        shadow_clip=float(metrics["shadow_clip"]),
        highlight_clip=float(metrics["highlight_clip"]),
        midtone_ratio=float(metrics["midtone_ratio"]),
        midtone_deviation=float(metrics["midtone_deviation"]),
        dr_p95_p5=float(metrics["dr_p95_p5"]),
        local_dr_mean=float(metrics["local_dr_mean"]),
        rg_ratio=float(metrics["rg_ratio"]),
        bg_ratio=float(metrics["bg_ratio"]),
        avg_saturation=float(metrics["avg_saturation"]),
        oversat_ratio=float(metrics["oversat_ratio"]),
        skin_hue_var=(float(skin) if skin is not None else None),
        lap_var=float(metrics["lap_var"]),
        edge_density=float(metrics["edge_density"]),
        hf_energy=float(metrics["hf_energy"]),
        luma_noise=float(metrics["luma_noise"]),
        chroma_noise=float(metrics["chroma_noise"]),
        score_exposure=float(e),
        score_dynamic_range=float(d),
        score_color=float(c),
        score_sharpness=float(s),
        score_noise=float(n),
        score_q=float(q),
    )
