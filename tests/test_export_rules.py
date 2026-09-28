from __future__ import annotations

from app.decision.export_rules import (
    EXPORT_CHOICES,
    KEEP_CHOICES,
    is_keeper,
    normalize_choice,
)


def test_choices() -> None:
    assert EXPORT_CHOICES == ("undecided", "discard", "original", "enhanced")
    assert KEEP_CHOICES == ("original", "enhanced")


def test_is_keeper() -> None:
    assert is_keeper("original") and is_keeper("enhanced")
    assert not is_keeper("discard")
    assert not is_keeper("undecided")


def test_normalize_choice() -> None:
    assert normalize_choice("Enhanced") == "enhanced"
    assert normalize_choice("bogus") is None
