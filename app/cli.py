"""Typer CLI — one command per pipeline phase plus `run --auto`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="AI RAW photo curation pipeline (ephemeral).")
console = Console()


@app.callback()
def _setup() -> None:
    """Runs before every sub-command: route log.* output to stderr."""
    from app.logging_setup import configure_logging

    configure_logging()


@app.command()
def ingest() -> None:
    """Walk photos/incoming -> DB rows + previews + thumbs."""
    from app.ingest.ingest_job import run_ingest

    run_ingest()


@app.command()
def filter() -> None:  # noqa: A001 — typer command, shadows builtin only in this module
    """Cheap CPU filters (blur / pHash / exposure flags)."""
    from app.filters.filter_job import run_filters

    run_filters()


@app.command()
def score(stage: str = "all") -> None:
    """GPU scoring. stage in {clip, iqa, faces, all}."""
    from app.scoring.score_job import run_scoring

    run_scoring(stage=stage)


@app.command()
def cluster() -> None:
    """EXIF burst grouping + CLIP HDBSCAN; one recommended photo per cluster."""
    from app.clustering.cluster_job import run_clustering

    run_clustering()


@app.command()
def enhance() -> None:
    """RAW -> AI chain -> before/after render JPEGs for every RAW photo."""
    from app.enhancement.enhance_job import run_enhancement

    run_enhancement()


@app.command()
def export() -> None:
    """Apply export decisions: chosen JPEG into photos/jpeg/, then RAW retention."""
    from app.export.export_job import run_export

    run_export()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Control-center UI + review API (runs every stage from the browser)."""
    import uvicorn

    uvicorn.run("app.api.main:app", host=host, port=port)


@app.command()
def reset(force: bool = False) -> None:
    """Wipe session state: DB, cache tiers, library/exported/jpeg. incoming/ is kept."""
    from app.orchestrator.reset import end_session

    end_session(force=force)


@app.command()
def run(auto: bool = False) -> None:
    """Leg 1 in one shot: ingest -> filter -> score -> cluster. Pass --auto to run."""
    if not auto:
        console.print("[yellow]Use --auto to run the full pipeline.[/yellow]")
        raise typer.Exit(2)

    from app.clustering.cluster_job import run_clustering
    from app.filters.filter_job import run_filters
    from app.ingest.ingest_job import run_ingest
    from app.scoring.score_job import run_scoring

    run_ingest()
    run_filters()
    run_scoring(stage="all")
    run_clustering()
    console.print("[green]Pipeline complete.[/green]")


@app.command()
def info() -> None:
    """Print resolved settings + DB status."""
    from sqlalchemy import inspect

    from app.config import settings
    from app.db import engine

    console.print(f"photos:  {settings.photos}")
    console.print(f"cache:   {settings.cache}")
    console.print(f"models:  {settings.models}")
    console.print(f"db_url:  {settings.db_url}")
    db_exists = Path(settings.db_path).exists()
    console.print(f"db file present: {db_exists}")
    if db_exists:
        tables = inspect(engine).get_table_names()
        console.print(f"tables ({len(tables)}): {tables}")


if __name__ == "__main__":
    app()
