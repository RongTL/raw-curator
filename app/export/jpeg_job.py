"""Orchestrate JPEG export from photos/library (RAWs) and photos/exported (TIFFs)."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from app.config import settings
from app.export.jpeg_writer import convert_image_to_jpeg, is_convertible

log = logging.getLogger(__name__)
console = Console()


def _list_candidates(source: str) -> list[Path]:
    """Return source files to convert. `source` in {'library', 'exported', 'all'}.

    Library can contain any supported image kind (RAW, JPEG, TIFF, HEIC, PNG).
    Exported contains the enhanced TIFFs from `make enhance`.
    """
    photos = settings.photos
    items: list[Path] = []
    if source in {"library", "all"}:
        lib = photos / "library"
        if lib.is_dir():
            items.extend(
                p for p in sorted(lib.iterdir()) if p.is_file() and is_convertible(p)
            )
    if source in {"exported", "all"}:
        exp = photos / "exported"
        if exp.is_dir():
            items.extend(
                p for p in sorted(exp.iterdir()) if p.is_file() and is_convertible(p)
            )
    return items


def _dest_for(src: Path) -> Path:
    return settings.photos / settings.jpeg_subdir / (src.stem + ".jpg")


def run_jpeg_export(
    source: str = "all",
    quality: int | None = None,
    long_edge: int | None = None,
    overwrite: bool = False,
) -> None:
    if source not in {"library", "exported", "all"}:
        raise ValueError(f"source must be one of library|exported|all, got {source!r}")

    q = quality if quality is not None else settings.jpeg_quality
    le = long_edge if long_edge is not None else settings.jpeg_long_edge
    progressive = settings.jpeg_progressive

    items = _list_candidates(source)
    if not items:
        console.print(f"[yellow]No files to export from source={source}.[/yellow]")
        return

    out_dir = settings.photos / settings.jpeg_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0
    console.print(
        f"[cyan]Exporting {len(items)} file(s) -> {out_dir} "
        f"(quality={q}, long_edge={le or 'native'}).[/cyan]"
    )
    with Progress() as progress:
        task = progress.add_task("export-jpeg", total=len(items))
        for src in items:
            dest = _dest_for(src)
            if dest.exists() and not overwrite:
                skipped += 1
                progress.advance(task)
                continue
            try:
                convert_image_to_jpeg(
                    src, dest, quality=q, long_edge=le, progressive=progressive
                )
                converted += 1
                console.print(f"  -> {dest.name}")
            except Exception as exc:  # noqa: BLE001 — keep the batch going
                failed += 1
                log.exception("failed to convert %s", src)
                console.print(f"  [red]x {src.name}: {exc}[/red]")
            progress.advance(task)

    console.print(
        f"[green]JPEG export complete:[/green] "
        f"converted={converted} skipped={skipped} failed={failed}"
    )
