"""Stage registry invariants."""

from __future__ import annotations

from app.orchestrator.stages import STAGE_BY_NAME, STAGES, leg_stages


def test_stage_order_matches_pipeline() -> None:
    assert [s.name for s in STAGES] == [
        "ingest", "filter", "score", "cluster", "submit", "enhance", "export-jpeg",
    ]


def test_legs_split_around_review() -> None:
    assert [s.name for s in leg_stages(1)] == ["ingest", "filter", "score", "cluster"]
    assert [s.name for s in leg_stages(2)] == ["submit", "enhance", "export-jpeg"]
    assert leg_stages(3) == ()


def test_lookup_and_cli_args() -> None:
    assert STAGE_BY_NAME["export-jpeg"].cli_args == ("export-jpeg",)
    assert STAGE_BY_NAME["score"].cli_args == ("score",)


def test_no_duplicate_stage_names() -> None:
    assert len(STAGE_BY_NAME) == len(STAGES)


def test_cli_args_are_real_typer_commands() -> None:
    import typer.main

    from app.cli import app

    commands = set(typer.main.get_command(app).commands)
    assert {s.cli_args[0] for s in STAGES} <= commands
