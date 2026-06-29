"""Bulk-stage the same decision across every photo in the batch.

Used by the `POST /api/decide/all` route ("don't keep any RAW" UI control).
Operates on a caller-provided session so the route manages the transaction
(via session_scope) and tests can drive it with a temp session.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Decision, Photo

_ALLOWED = {"yes", "no", "undecided"}


def stage_all(sess: Session, selected: str) -> int:
    """Set ``Decision.selected = selected`` for every photo, skipping rows
    already applied (submitted). Returns the number of decisions staged
    (created or updated). Raises ValueError for an unknown ``selected``.
    """
    if selected not in _ALLOWED:
        raise ValueError(f"selected must be one of {sorted(_ALLOWED)}, got {selected!r}")

    staged = 0
    hashes = sess.execute(select(Photo.hash)).scalars().all()
    for h in hashes:
        dec = sess.get(Decision, h)
        if dec is None:
            sess.add(Decision(photo_hash=h, selected=selected))
            staged += 1
        elif not dec.applied:
            dec.selected = selected
            staged += 1
    return staged
