"""Stage registry — single source of truth for runnable pipeline stages.

Review is not a stage here: it is a human step handled entirely by the UI
between auto-run legs 1 and 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class StageDef:
    name: str  # API identifier and log-file prefix
    title: str  # human label for the UI
    cli_args: tuple[str, ...]  # argv appended to the `raw-curator` command
    leg: int  # auto-run leg this stage belongs to (1 or 2)


STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", "Ingest", ("ingest",), 1),
    StageDef("filter", "Filter", ("filter",), 1),
    StageDef("score", "Score", ("score",), 1),
    StageDef("cluster", "Cluster", ("cluster",), 1),
    StageDef("submit", "Submit", ("submit",), 2),
    StageDef("enhance", "Enhance", ("enhance",), 2),
    StageDef("export-jpeg", "Export JPEG", ("export-jpeg",), 2),
)

STAGE_BY_NAME: Mapping[str, StageDef] = {s.name: s for s in STAGES}


def leg_stages(leg: int) -> tuple[StageDef, ...]:
    """Stages of an auto-run leg (1 or 2). Unknown legs return () — callers fail fast on empty."""
    return tuple(s for s in STAGES if s.leg == leg)
