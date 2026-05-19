"""Phase 7 unit tests."""

from __future__ import annotations

from app.decision.rules import resolve, tier_from_scores


def test_rules_cover_all_combinations() -> None:
    assert resolve("yes", "high").action == "keep_raw"
    assert resolve("yes", "low").action == "enhance_export"
    assert resolve("no", "high").action == "archive"
    assert resolve("no", "low").action == "quarantine"
    assert resolve("undecided", "high") is None


def test_tier_threshold() -> None:
    assert tier_from_scores(0.8, 8.0) == "high"
    assert tier_from_scores(0.1, 2.0) == "low"
