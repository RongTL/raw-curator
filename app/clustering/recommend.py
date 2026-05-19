"""Rank within a cluster: 0.6 * technical + 0.4 * aesthetic."""

from __future__ import annotations

from app.models import Photo


def score(p: Photo) -> float:
    tech = p.technical_score or 0.0
    aesthetic = p.aesthetic_score or 0.0
    # Aesthetic scores from v2.5 are roughly in [1,10] — normalize quickly.
    aesthetic_n = max(0.0, min(1.0, (aesthetic - 1.0) / 9.0))
    return 0.6 * tech + 0.4 * aesthetic_n


def rank(photos: list[Photo]) -> list[Photo]:
    return sorted(photos, key=score, reverse=True)
