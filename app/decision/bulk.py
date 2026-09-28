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
_CLUSTER_MODES = {"keep_recommended", "reject_all", "keep_all"}


def stage_cluster(sess: Session, cluster_id: int, mode: str) -> int:
    """Stage decisions for every photo in one cluster, skipping applied rows.

    - ``keep_recommended``: the recommended photo → ``yes``, the rest → ``no``
      (the burst-dedup shortcut: keep the best frame, drop the near-duplicates).
    - ``keep_all`` → every frame ``yes``; ``reject_all`` → every frame ``no``.

    Returns the number of decisions staged. Raises ValueError for an unknown
    ``mode`` or an empty/unknown cluster.
    """
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
            selected = "yes"
        elif mode == "reject_all":
            selected = "no"
        else:  # keep_recommended
            selected = "yes" if is_recommended else "no"
        dec = sess.get(Decision, h)
        if dec is None:
            sess.add(Decision(photo_hash=h, selected=selected))
            staged += 1
        elif not dec.applied:
            dec.selected = selected
            staged += 1
    return staged


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
