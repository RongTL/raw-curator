"""Shared, schema-tolerant lookups for the review routes.

The queue/cluster grids and the detail modal badge enhanced frames from the
``quality_reports`` table, but a session DB created before migration 0004
(``plan_and_verify``) lacks those columns. These helpers degrade to "no enhanced
info" (with a one-time warning) instead of 500-ing the whole review UI, so an old
batch is still reviewable. Run ``alembic upgrade head`` (or ``make reset``) to
restore the columns.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from app.models import Face, PhotoQualityReport

log = logging.getLogger(__name__)

# Columns added by migration 0004; their absence means the DB predates it.
_ENHANCED_COLS = {"score_q_after", "degraded"}
_DETAIL_SENTINEL = "plan_json"


def face_counts_by_hash(sess: Session) -> dict[str, int]:
    """{photo_hash: number of detected faces} for the whole batch."""
    return {
        h: n
        for h, n in sess.execute(
            select(Face.photo_hash, func.count()).group_by(Face.photo_hash)
        ).all()
    }


def _quality_columns(sess: Session) -> set[str]:
    return {c["name"] for c in inspect(sess.get_bind()).get_columns("quality_reports")}


def enhanced_by_hash(sess: Session) -> dict[str, tuple[float | None, bool]]:
    """{photo_hash: (score_q_after, degraded)} for enhanced-result badges.

    Returns ``{}`` (and warns once) when the ``quality_reports`` table predates the
    plan/verify migration, so the grid still loads without enhanced badges.
    """
    if not _quality_columns(sess) >= _ENHANCED_COLS:
        log.warning(
            "quality_reports predates the plan/verify migration; enhanced badges "
            "disabled. Run `alembic upgrade head` (or `make reset`) on this session DB."
        )
        return {}
    rows = sess.execute(
        select(
            PhotoQualityReport.photo_hash,
            PhotoQualityReport.score_q_after,
            PhotoQualityReport.degraded,
        )
    ).all()
    return {h: (qa, bool(deg)) for h, qa, deg in rows}


def quality_report_or_none(sess: Session, photo_hash: str) -> PhotoQualityReport | None:
    """The full quality report row, or ``None`` when the table predates migration 0004.

    Guards the detail modal: loading the ORM row selects every mapped column, so a
    pre-0004 DB would raise ``OperationalError`` on the missing columns.
    """
    if _DETAIL_SENTINEL not in _quality_columns(sess):
        return None
    return sess.get(PhotoQualityReport, photo_hash)
