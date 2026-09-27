"""Rank within a cluster by the shared technical/aesthetic blend (app.scoring.combined)."""

from __future__ import annotations

from app.models import Photo
from app.scoring.combined import combined_score


def score(p: Photo) -> float:
    return combined_score(p.technical_score, p.aesthetic_score)


def rank(photos: list[Photo]) -> list[Photo]:
    return sorted(photos, key=score, reverse=True)
