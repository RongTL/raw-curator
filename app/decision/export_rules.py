"""Post-enhance export choice model.

The curator makes one choice per photo after seeing before/after:
- ``discard``  — no JPEG, RAW left untouched in incoming/.
- ``original`` — JPEG from the developed (un-enhanced) render.
- ``enhanced`` — JPEG from the AI-enhanced render.
``keep_raw`` (a separate flag) decides whether a kept photo's RAW is archived
to library/ or deleted after the JPEG is written. This replaces the old binary
yes/no keep-RAW routing.
"""

from __future__ import annotations

EXPORT_CHOICES: tuple[str, ...] = ("undecided", "discard", "original", "enhanced")
KEEP_CHOICES: tuple[str, ...] = ("original", "enhanced")  # produce a share JPEG


def is_keeper(choice: str) -> bool:
    """True if this choice produces an exported JPEG (original or enhanced)."""
    return choice in KEEP_CHOICES


def normalize_choice(choice: str) -> str | None:
    """Case-insensitive lookup; returns a member of EXPORT_CHOICES or None."""
    low = choice.lower()
    return low if low in EXPORT_CHOICES else None
