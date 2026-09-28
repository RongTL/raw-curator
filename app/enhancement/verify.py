"""Post-enhancement verification: did the chain make the frame worse?"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from app.arrays import Array
from app.enhancement.colorspace import encode_srgb, luma
from app.enhancement.engine.plan import EnhancementPlan, QualityReport

Q_DROP_LIMIT = 5.0
SAFE_STEPS = frozenset(
    {"exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover", "highlight_rolloff"}
)


@dataclass(frozen=True)
class Verdict:
    degraded: bool
    reasons: tuple[str, ...]
    q_before: float
    q_after: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "degraded": self.degraded,
                "reasons": list(self.reasons),
                "q_before": self.q_before,
                "q_after": self.q_after,
            }
        )


def verify(before: QualityReport, after: QualityReport, result: Array) -> Verdict:
    reasons: list[str] = []
    if after.score_q < before.score_q - Q_DROP_LIMIT:
        reasons.append("q_drop")
    if after.highlight_clip > 2.0 * before.highlight_clip + 0.01:
        reasons.append("highlight_clip")
    # The spec's collapse sentinels are display-referred: evaluate them on the
    # sRGB-encoded luma so a legitimately dark (but visible) linear frame is not
    # mistaken for a black one (linear 0.03 ~= display 0.19).
    lum = encode_srgb(luma(result))
    if float(lum.mean()) < 0.03:
        reasons.append("mean_luma")
    if float(lum.std()) < 0.02:
        reasons.append("std")
    return Verdict(bool(reasons), tuple(reasons), before.score_q, after.score_q)


def safe_plan(plan: EnhancementPlan) -> EnhancementPlan:
    steps = tuple(s for s in plan.steps if s.name in SAFE_STEPS)
    return replace(plan, steps=steps, note=f"safe retry ({len(steps)} tone step(s))")
