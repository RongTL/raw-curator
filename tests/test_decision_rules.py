"""Binary-routing unit tests."""

from __future__ import annotations

from app.decision.rules import resolve, tier_from_scores


def test_yes_keeps_raw_and_enhances() -> None:
    rule = resolve("yes")
    assert rule is not None
    assert rule.action == "keep_and_enhance"
    assert rule.library_subdir == "library"
    assert rule.delete_source_after_enhance is False


def test_no_enhances_and_discards() -> None:
    rule = resolve("no")
    assert rule is not None
    assert rule.action == "enhance_only"
    assert rule.library_subdir is None
    assert rule.delete_source_after_enhance is True


def test_undecided_returns_none() -> None:
    assert resolve("undecided") is None
    assert resolve("") is None


def test_resolve_is_case_insensitive() -> None:
    assert resolve("YES").action == "keep_and_enhance"
    assert resolve("No").action == "enhance_only"


def test_tier_from_scores_threshold() -> None:
    """tier_from_scores no longer drives routing but stays as a display hint."""
    assert tier_from_scores(0.8, 8.0) == "high"
    assert tier_from_scores(0.1, 2.0) == "low"
