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

from app.scoring.combined import HIGH_TIER_THRESHOLD, combined_score


@dataclass(frozen=True)
class Rule:
    selected: str  # "yes" | "no"
    action: str  # "keep_and_enhance" | "enhance_only"
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


# Every decided photo goes through enhance; this is the set enhance_job and
# the progress poller query for.
ENHANCE_ACTIONS: tuple[str, ...] = tuple(r.action for r in RULES.values())


def resolve(selected: str) -> Rule | None:
    return RULES.get(selected.lower())


def tier_from_scores(technical: float | None, aesthetic: float | None) -> str:
    """High if the combined score clears HIGH_TIER_THRESHOLD, else low.
    Display-only since the binary routing change."""
    return "high" if combined_score(technical, aesthetic) >= HIGH_TIER_THRESHOLD else "low"
