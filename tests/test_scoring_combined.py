"""The 0.6*technical + 0.4*aesthetic blend used for ranking and the display tier."""

from __future__ import annotations

from app.scoring.combined import combined_score


def test_missing_scores_count_as_zero() -> None:
    assert combined_score(None, None) == 0.0


def test_perfect_scores_blend_to_one() -> None:
    assert combined_score(1.0, 10.0) == 1.0


def test_aesthetic_is_normalised_from_1_to_10() -> None:
    # tech 0.5 -> 0.3 ; aesthetic 5.5 -> (5.5-1)/9 = 0.5 -> 0.2
    assert abs(combined_score(0.5, 5.5) - 0.5) < 1e-9


def test_aesthetic_is_clamped() -> None:
    assert combined_score(0.0, 12.0) == 0.4
    assert combined_score(0.0, -3.0) == 0.0
