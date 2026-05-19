"""Hybrid RAW -> AI -> TIFF for the Yes+Low set."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.enhancement.denoise import scunet_denoise
from app.enhancement.develop_full import darktable_cli
from app.enhancement.downsample import scale
from app.enhancement.face_restore import codeformer_restore
from app.enhancement.pack_tiff import write_tiff16
from app.enhancement.upsample_final import upsample_final
from app.enhancement.upscale import realesrgan_x2
from app.models import Decision, Face, Photo

log = logging.getLogger(__name__)
console = Console()


def _candidates() -> list[tuple[dict, bool]]:
    """Detached snapshots safe to consume after the session closes."""
    snapshots: list[tuple[dict, bool]] = []
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.source_path, Photo.file_kind, Decision.action)
            .join(Decision, Photo.hash == Decision.photo_hash)
            .where(Decision.action == "enhance_export")
        ).all()
        faces_by_hash: dict[str, bool] = {}
        for (digest,) in sess.execute(select(Face.photo_hash).distinct()).all():
            faces_by_hash[digest] = True
        for digest, source_path, file_kind, _action in rows:
            snapshots.append(
                (
                    {"hash": digest, "source_path": source_path, "file_kind": file_kind},
                    faces_by_hash.get(digest, False),
                )
            )
    return snapshots


def _xmp_for(source_path: str) -> Path | None:
    src = Path(source_path)
    candidate = settings.xmp / (src.stem + ".xmp")
    return candidate if candidate.exists() else None


def _free_gpu() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _enhance_one(photo: dict, has_faces: bool) -> Path | None:
    src = Path(photo["source_path"])
    if not src.exists():
        log.warning("source missing: %s", src)
        return None
    file_kind = photo.get("file_kind")
    if file_kind is not None and file_kind != "raw":
        log.warning(
            "skipping enhance for non-RAW source (kind=%s): %s", file_kind, src.name
        )
        return None
    full_tiff = darktable_cli(src, xmp=_xmp_for(photo["source_path"]))
    arr = np.asarray(Image.open(full_tiff))
    native_h, native_w = arr.shape[:2]
    ai_in = scale(arr, settings.enhance_ai_scale)
    if settings.enhance_denoise:
        ai_in = scunet_denoise(ai_in)
        _free_gpu()
    ai_up = realesrgan_x2(ai_in)
    _free_gpu()
    if settings.enhance_face_restore and has_faces:
        ai_up = codeformer_restore(ai_up, faces=None, weight=settings.enhance_codeformer_w)
        _free_gpu()
    final = upsample_final(ai_up, (native_w, native_h))
    out = settings.photos / "exported" / (src.stem + ".tif")
    write_tiff16(final, out)
    try:
        full_tiff.unlink()
    except OSError:
        pass
    return out


def run_enhancement() -> None:
    items = _candidates()
    if not items:
        console.print("[yellow]No photos to enhance.[/yellow]")
        return
    console.print(f"[cyan]Enhancing {len(items)} photo(s).[/cyan]")
    enhanced = 0
    skipped = 0
    with Progress() as progress:
        task = progress.add_task("enhance", total=len(items))
        for photo, has_faces in items:
            out = _enhance_one(photo, has_faces)
            if out:
                enhanced += 1
                console.print(f"  -> {out}")
            else:
                skipped += 1
            progress.advance(task)
    console.print(
        f"[green]Enhancement complete:[/green] enhanced={enhanced} skipped={skipped}"
    )
