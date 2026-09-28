"""Bulk-stage the same choice across a cluster or the whole batch.

Used by the review header controls ("use enhanced for all", "keep no RAW", …)
and the cluster keep-best/reject-all buttons. Operates on a caller-provided
session so the route owns the transaction and tests drive it with a temp session.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.decision.export_rules import EXPORT_CHOICES, normalize_choice
from app.models import Decision, Photo

_CLUSTER_MODES = {"keep_recommended", "reject_all", "keep_all"}


def _decision(sess: Session, h: str) -> Decision | None:
    """The row to mutate, or None if it exists and is already applied (skip)."""
    dec = sess.get(Decision, h)
    if dec is None:
        dec = Decision(photo_hash=h, keep_raw=int(settings.keep_raw_default))
        sess.add(dec)
        return dec
    return None if dec.applied else dec


def stage_all(
    sess: Session, *, export_choice: str | None = None, keep_raw: bool | None = None
) -> int:
    """Set the provided field(s) on every non-applied photo's decision.

    Returns the number of rows staged. Raises ValueError for an unknown choice.
    """
    if export_choice is None and keep_raw is None:
        return 0  # nothing to stage — don't materialize empty undecided rows
    if export_choice is not None:
        normalized = normalize_choice(export_choice)  # canonicalize mixed-case input
        if normalized is None:
            raise ValueError(
                f"export_choice must be one of {EXPORT_CHOICES}, got {export_choice!r}"
            )
        export_choice = normalized
    staged = 0
    for h in sess.execute(select(Photo.hash)).scalars().all():
        dec = _decision(sess, h)
        if dec is None:
            continue
        if export_choice is not None:
            dec.export_choice = export_choice
        if keep_raw is not None:
            dec.keep_raw = bool(keep_raw)  # Integer-backed column; SQLAlchemy stores 0/1
        staged += 1
    return staged


def stage_cluster(sess: Session, cluster_id: int, mode: str) -> int:
    """keep_recommended: recommended→enhanced, rest→discard; reject_all: all
    discard; keep_all: all enhanced. Skips applied rows. Raises ValueError for
    an unknown mode or empty cluster."""
    if mode not in _CLUSTER_MODES:
        raise ValueError(f"mode must be one of {sorted(_CLUSTER_MODES)}, got {mode!r}")
    rows = sess.execute(
        select(Photo.hash, Photo.is_recommended).where(Photo.cluster_id == cluster_id)
    ).all()
    if not rows:
        raise ValueError(f"no photos in cluster {cluster_id}")
    staged = 0
    for h, is_recommended in rows:
        if mode == "keep_all":
            choice = "enhanced"
        elif mode == "reject_all":
            choice = "discard"
        else:  # keep_recommended
            choice = "enhanced" if is_recommended else "discard"
        dec = _decision(sess, h)
        if dec is None:
            continue
        dec.export_choice = choice
        staged += 1
    return staged
