"""Orchestrate the ingest stage: walk -> hash -> develop -> previews -> EXIF -> DB upsert."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.ingest.decode import classify_kind, decode_preview, extract_thumb_bytes
from app.ingest.exif import ExifData, ExifReader
from app.ingest.hasher import xxh3_file
from app.ingest.walker import walk
from app.models import Photo
from app.preview.jpeg_writer import decode_jpeg_bytes, resize_long_edge, write_jpeg

log = logging.getLogger(__name__)
console = Console()


@dataclass
class IngestResult:
    hash: str
    source_path: str
    thumb_path: str
    preview_path: str
    width: int
    height: int
    file_kind: str


def _ingest_one(src_path_str: str) -> IngestResult | None:
    src_path = Path(src_path_str)
    try:
        kind = classify_kind(src_path)
        if kind is None:
            log.warning("skipping unsupported file: %s", src_path)
            return None
        digest = xxh3_file(src_path)
        preview_arr = decode_preview(src_path)
        h, w = preview_arr.shape[:2]
        preview = resize_long_edge(preview_arr, settings.preview_long_edge)
        thumb_src = extract_thumb_bytes(src_path)
        thumb_arr = decode_jpeg_bytes(thumb_src) if thumb_src is not None else preview_arr
        thumb = resize_long_edge(thumb_arr, settings.thumb_long_edge)

        preview_path = settings.previews_dir / f"{digest}.jpg"
        thumb_path = settings.thumbs_dir / f"{digest}.jpg"
        write_jpeg(preview, preview_path, quality=settings.jpeg_quality_preview)
        write_jpeg(thumb, thumb_path, quality=settings.jpeg_quality_thumb)

        return IngestResult(
            hash=digest,
            source_path=str(src_path),
            thumb_path=str(thumb_path),
            preview_path=str(preview_path),
            width=int(w),
            height=int(h),
            file_kind=kind.value,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed for %s: %s", src_path, exc)
        return None


def _known_hashes() -> set[str]:
    with session_scope() as sess:
        rows = sess.execute(select(Photo.hash)).scalars().all()
    return set(rows)


def run_ingest() -> None:
    incoming = settings.photos / "incoming"
    paths = list(walk(incoming))
    if not paths:
        console.print(f"[yellow]No supported image files found under {incoming}.[/yellow]")
        return

    known = _known_hashes()
    console.print(f"Found {len(paths)} candidate image(s); {len(known)} already ingested.")

    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbs_dir.mkdir(parents=True, exist_ok=True)

    ingested: list[IngestResult] = []
    workers = max(1, settings.cpu_workers)
    with Progress() as progress:
        task = progress.add_task("ingest", total=len(paths))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_ingest_one, str(p)): p for p in paths}
            for fut in as_completed(futures):
                ir = fut.result()
                progress.advance(task)
                if ir is None or ir.hash in known:
                    continue
                ingested.append(ir)

    exif_by_path: dict[str, ExifData] = {}
    if ingested:
        with ExifReader() as reader:
            for ir in ingested:
                exif_by_path[ir.source_path] = reader.read(Path(ir.source_path))

    with session_scope() as sess:
        for ir in ingested:
            data = exif_by_path.get(ir.source_path, ExifData())
            exif_fields = {k: v for k, v in asdict(data).items() if v is not None}
            exif_fields.setdefault("width", ir.width)
            exif_fields.setdefault("height", ir.height)
            photo = Photo(
                hash=ir.hash,
                source_path=ir.source_path,
                thumb_path=ir.thumb_path,
                preview_path=ir.preview_path,
                file_kind=ir.file_kind,
                **exif_fields,
            )
            sess.merge(photo)
    console.print(f"[green]Ingested {len(ingested)} new photo(s).[/green]")
