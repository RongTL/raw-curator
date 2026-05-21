"""Phase 5 unit tests: burst grouping + recommendation ranking."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.clustering.exif_burst import burst_groups
from app.clustering.recommend import rank
from app.models import Photo


def _photo(idx: int, body: str, ts: datetime, tech: float = 0.5, aesthetic: float = 5.0) -> Photo:
    return Photo(
        hash=f"hash{idx:04d}", source_path=f"/dev/null/{idx}.cr3", camera_body=body,
        captured_at=ts, technical_score=tech, aesthetic_score=aesthetic,
    )


def test_burst_groups_within_window() -> None:
    base = datetime(2026, 1, 1, 12, 0, 0)
    photos = [
        _photo(1, "EOS R6", base + timedelta(seconds=0)),
        _photo(2, "EOS R6", base + timedelta(seconds=1)),
        _photo(3, "EOS R6", base + timedelta(seconds=2)),
        _photo(4, "EOS R6", base + timedelta(seconds=60)),
    ]
    groups = burst_groups(photos, window_seconds=2)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_rank_prefers_higher_technical() -> None:
    a = _photo(1, "X", datetime(2026, 1, 1), tech=0.9, aesthetic=5.0)
    b = _photo(2, "X", datetime(2026, 1, 1), tech=0.2, aesthetic=5.0)
    ordered = rank([b, a])
    assert ordered[0].hash == a.hash
