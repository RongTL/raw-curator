"""Within-burst near-duplicate dedupe via pHash Hamming distance."""

from __future__ import annotations

from typing import Iterable

from app.filters.phash import hamming
from app.models import Photo


def dedupe_pairs(photos: Iterable[Photo], threshold: int = 8) -> list[tuple[Photo, Photo]]:
    photos = [p for p in photos if p.phash]
    out: list[tuple[Photo, Photo]] = []
    for i, a in enumerate(photos):
        for b in photos[i + 1 :]:
            if hamming(a.phash, b.phash) <= threshold:
                out.append((a, b))
    return out
