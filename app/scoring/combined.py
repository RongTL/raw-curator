"""The technical/aesthetic blend used for in-cluster ranking and the display tier.

    combined = 0.6 * technical + 0.4 * normalised_aesthetic

``technical`` is already in [0, 1] (mean of min-max normalised MUSIQ and
MANIQA). Aesthetic-predictor v2.5 scores land roughly in [1, 10] and are
mapped onto [0, 1] here.
"""

from __future__ import annotations

TECH_WEIGHT = 0.6
AESTHETIC_WEIGHT = 0.4
HIGH_TIER_THRESHOLD = 0.55


def normalise_aesthetic(aesthetic: float | None) -> float:
    return max(0.0, min(1.0, ((aesthetic or 0.0) - 1.0) / 9.0))


def combined_score(technical: float | None, aesthetic: float | None) -> float:
    return TECH_WEIGHT * (technical or 0.0) + AESTHETIC_WEIGHT * normalise_aesthetic(aesthetic)
