from __future__ import annotations

from app.config import settings


def test_enhanced_dir_under_cache() -> None:
    assert settings.enhanced_dir == settings.cache / "enhanced"


def test_review_and_keep_raw_defaults() -> None:
    assert settings.review_long_edge == 3000
    assert settings.keep_raw_default is True
