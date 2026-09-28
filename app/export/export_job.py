"""Apply export decisions: chosen JPEG into photos/jpeg/, then RAW retention.

Replaces the old submit + export-jpeg stages. Driven by the DB, not the
filesystem: each kept decision picks exactly one source render, so the old
library-vs-exported jpeg collision is gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.decision.executor import Move, apply_moves
from app.decision.export_rules import is_keeper
from app.enhancement.render_jpeg import render_paths
from app.export.jpeg_writer import convert_image_to_jpeg
from app.models import Decision, Photo
from app.paths import relative_subpath

log = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class ExportSummary:
    exported: int = 0
    skipped: int = 0
    failed: int = 0


def _source_jpeg(choice: str, photo: Photo) -> Path:
    """Which image to encode into the share JPEG for this photo.

    ``enhanced`` uses the AI render, but a non-RAW or enhance-failed photo has no
    ``after_full``; fall back to the same source ``original`` uses (developed
    ``before_full`` if present, else the original file) so a keeper is never
    silently skipped at export.
    """
    paths = render_paths(photo.hash)
    if choice == "enhanced" and paths["after_full"].exists():
        return paths["after_full"]
    before = paths["before_full"]
    if before.exists():  # a developed RAW render
        return before
    return Path(photo.source_path)  # non-RAW: the original file itself


def _dest_for(source_path: str) -> Path:
    rel = relative_subpath(Path(source_path), settings.photos).with_suffix(".jpg")
    return settings.photos / settings.jpeg_subdir / rel


def run_export() -> ExportSummary:
    exported = skipped = failed = 0
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo, Decision)
            .join(Decision, Photo.hash == Decision.photo_hash)
            .where(Decision.applied == 0)
        ).all()
        todo = [(p, d) for p, d in rows if is_keeper(d.export_choice)]
        if not todo:
            console.print("[yellow]No photos to export.[/yellow]")
            return ExportSummary()

        with Progress() as progress:
            task = progress.add_task("export", total=len(todo))
            for photo, decision in todo:
                try:
                    src = _source_jpeg(decision.export_choice, photo)
                    if not src.exists():
                        log.warning("skip %s: source render missing %s", photo.hash, src)
                        skipped += 1
                        continue
                    dest = _dest_for(photo.source_path)
                    convert_image_to_jpeg(
                        src,
                        dest,
                        quality=settings.jpeg_quality,
                        long_edge=settings.jpeg_long_edge,
                        progressive=settings.jpeg_progressive,
                    )
                    if not dest.exists():
                        raise RuntimeError(f"export JPEG not written: {dest}")
                    # Retention runs only after the JPEG exists on disk.
                    raw = Path(photo.source_path)
                    if decision.keep_raw and raw.exists():
                        lib = settings.photos / "library" / relative_subpath(raw, settings.photos)
                        apply_moves([Move(src=raw, dst=lib)])
                        photo.source_path = str(lib)
                    elif not decision.keep_raw and raw.exists():
                        raw.unlink()
                        log.info("deleted source RAW after export: %s", raw)
                    decision.applied = 1
                    exported += 1
                    console.print(f"  -> {dest}")
                except Exception as exc:  # noqa: BLE001 — keep the batch going
                    failed += 1
                    log.exception("export failed for %s", photo.source_path)
                    console.print(f"  [red]x {photo.hash}: {exc}[/red]")
                finally:
                    progress.advance(task)

    colour = "red" if failed else "green"
    console.print(
        f"[{colour}]Export complete:[/{colour}] "
        f"exported={exported} skipped={skipped} failed={failed}"
    )
    return ExportSummary(exported=exported, skipped=skipped, failed=failed)
