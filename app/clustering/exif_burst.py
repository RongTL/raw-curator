"""Group photos into bursts by (camera_body, captured_at +/- N seconds)."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from app.models import Photo


def burst_groups(photos: Iterable[Photo], window_seconds: int = 2) -> list[list[Photo]]:
    with_ts = [p for p in photos if p.captured_at is not None and p.camera_body]
    with_ts.sort(key=lambda p: (p.camera_body, p.captured_at))
    if not with_ts:
        return []
    out: list[list[Photo]] = []
    current: list[Photo] = [with_ts[0]]
    window = timedelta(seconds=window_seconds)
    for p in with_ts[1:]:
        last = current[-1]
        if (
            p.camera_body == last.camera_body
            and (p.captured_at - last.captured_at) <= window
        ):
            current.append(p)
        else:
            if len(current) > 1:
                out.append(current)
            current = [p]
    if len(current) > 1:
        out.append(current)
    return out
