"""Selected -> action mapping.

Binary decision model: the curator marks each photo Yes or No.
- Yes: original RAW is kept in `photos/library/`, AND the photo is enhanced.
- No:  original RAW is deleted after enhancement succeeds.

Every decided photo (yes or no) flows through the enhancement chain so that
every kept-or-discarded source produces a developed TIFF in `photos/exported/`.
The score tier is no longer part of routing; it remains in the DB for
display and analytics only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    selected: str           # "yes" | "no"
    action: str             # "keep_and_enhance" | "enhance_only"
    library_subdir: str | None  # where the RAW lands at submit; None means stay-in-place
    delete_source_after_enhance: bool  # for "no", remove the RAW once the TIFF is written


RULES: dict[str, Rule] = {
    "yes": Rule(
        selected="yes",
        action="keep_and_enhance",
        library_subdir="library",
        delete_source_after_enhance=False,
    ),
    "no": Rule(
        selected="no",
        action="enhance_only",
        library_subdir=None,
        delete_source_after_enhance=True,
    ),
}


def resolve(selected: str) -> Rule | None:
    return RULES.get(selected.lower())


def tier_from_scores(technical: float | None, aesthetic: float | None) -> str:
    """High if combined score >= 0.55, else low. Display-only since the
    binary routing change — kept so the UI can still surface a quality hint."""
    tech = technical or 0.0
    aesthetic = aesthetic or 0.0
    aesthetic_n = max(0.0, min(1.0, (aesthetic - 1.0) / 9.0))
    combined = 0.6 * tech + 0.4 * aesthetic_n
    return "high" if combined >= 0.55 else "low"
