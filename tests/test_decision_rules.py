"""Phase 7 unit tests."""

from __future__ import annotations

from app.decision.rules import resolve, tier_from_scores


def test_rules_cover_all_combinations() -> None:
    assert resolve("yes", "high").action == "keep_raw"
    assert resolve("yes", "low").action == "enhance_export"
    assert resolve("no", "high").action == "enhance_export"
    assert resolve("no", "low").action == "enhance_export"
    assert resolve("undecided", "high") is None


def test_only_yes_high_keeps_raw() -> None:
    """yes+high is the only path that bypasses enhancement."""
    assert resolve("yes", "high").action == "keep_raw"
    assert resolve("yes", "high").dest_subdir == "library"
    for sel, tier in [("yes", "low"), ("no", "high"), ("no", "low")]:
        rule = resolve(sel, tier)
        assert rule.action == "enhance_export"
        assert rule.dest_subdir == "exported"


def test_tier_threshold() -> None:
    assert tier_from_scores(0.8, 8.0) == "high"
    assert tier_from_scores(0.1, 2.0) == "low"
