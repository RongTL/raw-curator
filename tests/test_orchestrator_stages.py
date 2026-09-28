"""Stage registry invariants."""

from __future__ import annotations

from app.orchestrator.stages import STAGE_BY_NAME, STAGES


def test_leg_membership() -> None:
    from app.orchestrator.stages import leg_stages

    assert [s.name for s in leg_stages(1)] == ["ingest", "filter", "score", "cluster", "enhance"]
    assert [s.name for s in leg_stages(2)] == ["export"]


def test_export_cli_args() -> None:
    from app.orchestrator.stages import STAGE_BY_NAME

    assert STAGE_BY_NAME["export"].cli_args == ("export",)
    assert "submit" not in STAGE_BY_NAME
    assert "export-jpeg" not in STAGE_BY_NAME


def test_no_duplicate_stage_names() -> None:
    assert len(STAGE_BY_NAME) == len(STAGES)
