"""(selected, score_tier) -> action mapping from plan.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    selected: str   # "yes" | "no"
    score_tier: str # "high" | "low"
    action: str     # "keep_raw" | "enhance_export"
    dest_subdir: str


# Workflow: only "yes + high" keeps the RAW untouched. Everything else
# (kept-but-low quality, or anything the curator said "no" to) goes
# through the AI enhancement chain so it gets a second chance.
RULES: dict[tuple[str, str], Rule] = {
    ("yes", "high"): Rule("yes", "high", "keep_raw", "library"),
    ("yes", "low"):  Rule("yes", "low",  "enhance_export", "exported"),
    ("no",  "high"): Rule("no",  "high", "enhance_export", "exported"),
    ("no",  "low"):  Rule("no",  "low",  "enhance_export", "exported"),
}


def resolve(selected: str, score_tier: str) -> Rule | None:
    return RULES.get((selected.lower(), score_tier.lower()))


def tier_from_scores(technical: float | None, aesthetic: float | None) -> str:
    """High if combined score >= 0.55, else low. Threshold tuned for v1."""
    tech = technical or 0.0
    aesthetic = aesthetic or 0.0
    aesthetic_n = max(0.0, min(1.0, (aesthetic - 1.0) / 9.0))
    combined = 0.6 * tech + 0.4 * aesthetic_n
    return "high" if combined >= 0.55 else "low"
