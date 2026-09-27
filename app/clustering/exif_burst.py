"""Group photos into bursts by (camera_body, captured_at +/- N seconds)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from app.models import Photo


def burst_groups(photos: Iterable[Photo], window_seconds: int = 2) -> list[list[Photo]]:
    stamped: list[tuple[str, datetime, Photo]] = [
        (p.camera_body, p.captured_at, p)
        for p in photos
        if p.captured_at is not None and p.camera_body
    ]
    if not stamped:
        return []
    stamped.sort(key=lambda t: (t[0], t[1]))
    window = timedelta(seconds=window_seconds)

    out: list[list[Photo]] = []
    current: list[Photo] = [stamped[0][2]]
    last_body, last_ts = stamped[0][0], stamped[0][1]
    for body, ts, p in stamped[1:]:
        if body == last_body and (ts - last_ts) <= window:
            current.append(p)
        else:
            if len(current) > 1:
                out.append(current)
            current = [p]
        last_body, last_ts = body, ts
    if len(current) > 1:
        out.append(current)
    return out
