"""Typer CLI — one command per pipeline phase plus `run --auto`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="AI RAW photo curation pipeline (ephemeral).")
console = Console()


@app.command()
def ingest() -> None:
    """Phase 2: walk photos/incoming -> DB rows + previews + thumbs."""
    from app.ingest.ingest_job import run_ingest

    run_ingest()


@app.command()
def filter() -> None:  # noqa: A001 — typer command, shadows builtin only in this module
    """Phase 3: cheap CPU filters (blur/phash/exposure)."""
    from app.filters.filter_job import run_filters

    run_filters()


@app.command()
def score(stage: str = "all") -> None:
    """Phase 4: GPU scoring. stage in {clip, iqa, faces, all}."""
    from app.scoring.score_job import run_scoring

    run_scoring(stage=stage)


@app.command()
def cluster() -> None:
    """Phase 5: burst + phash dedupe + CLIP HDBSCAN."""
    from app.clustering.cluster_job import run_clustering

    run_clustering()


@app.command()
def submit() -> None:
    """Phase 7: apply staged decisions atomically."""
    from app.decision.decide_job import apply_decisions

    apply_decisions()


@app.command()
def enhance() -> None:
    """Phase 8: hybrid RAW -> AI -> 16-bit TIFF for the Yes+Low set."""
    from app.enhancement.enhance_job import run_enhancement

    run_enhancement()


@app.command("export-jpeg")
def export_jpeg(
    source: str = typer.Option(
        "all", help="Which folders to export from: library | exported | all."
    ),
    quality: int = typer.Option(
        None, help="JPEG quality (0-100). Default from RAWCURATOR_JPEG_QUALITY."
    ),
    long_edge: int = typer.Option(
        None,
        help="Resize so the long edge equals this many pixels. 0 = native resolution.",
    ),
    overwrite: bool = typer.Option(
        False, help="Re-encode files whose JPEG already exists."
    ),
) -> None:
    """Phase 9 (optional): convert library RAWs and exported TIFFs to share-ready JPEGs."""
    from app.export.jpeg_job import run_jpeg_export

    run_jpeg_export(source=source, quality=quality, long_edge=long_edge, overwrite=overwrite)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Phase 6: FastAPI + UI."""
    import uvicorn

    uvicorn.run("app.api.main:app", host=host, port=port)


@app.command()
def reset(force: bool = False) -> None:
    """Phase 9: wipe session state (cache + working dirs + DB)."""
    from scripts.end_session import end_session

    end_session(force=force)


@app.command()
def run(auto: bool = False) -> None:
    """Phase 10: orchestrated full pipeline. Pass --auto to actually run."""
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
