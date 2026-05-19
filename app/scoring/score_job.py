"""Orchestrate the three scoring stages and write results to DB.

Stage A: CLIP image embeddings + aesthetic head -> unload.
Stage B: MUSIQ -> unload. Then MANIQA -> unload. Ensemble = mean of normalized.
Stage C: InsightFace buffalo_l -> unload.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select, update

from app.config import settings
from app.db import session_scope
from app.embedding.clip import ClipEmbedder, vec_to_bytes
from app.embedding.faces import FaceDetector
from app.models import Face, Photo, PhotoEmbedding
from app.scoring.iqa import ManiqaScorer, MusiqScorer, normalize
from app.workers.gpu_worker import assert_gpu_or_skip, chunked, memory_summary, warmup

log = logging.getLogger(__name__)
console = Console()


def _photos_missing(column: str) -> list[tuple[str, str]]:
    col = getattr(Photo, column)
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.preview_path).where(col.is_(None))
        ).all()
    return [(h, p) for h, p in rows if p and Path(p).exists()]


def _photos_missing_embedding() -> list[tuple[str, str]]:
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.preview_path)
            .outerjoin(PhotoEmbedding, Photo.hash == PhotoEmbedding.photo_hash)
            .where(PhotoEmbedding.photo_hash.is_(None))
        ).all()
    return [(h, p) for h, p in rows if p and Path(p).exists()]


def _photos_missing_faces() -> list[tuple[str, str]]:
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.preview_path)
            .outerjoin(Face, Photo.hash == Face.photo_hash)
            .where(Face.id.is_(None))
        ).all()
    return [(h, p) for h, p in rows if p and Path(p).exists()]


def _stage_clip(items):
    if not items:
        console.print("[green]CLIP: nothing to embed.[/green]")
        return
    console.print(f"[cyan]Stage A — CLIP ({len(items)} photos)[/cyan]")
    with ClipEmbedder() as clip, Progress() as progress:
        task = progress.add_task("clip", total=len(items))
        for chunk in chunked(items, settings.clip_batch):
            results = list(clip.embed_batch([Path(p) for _, p in chunk]))
            with session_scope() as sess:
                for (digest, _), (_path, vec) in zip(chunk, results, strict=True):
                    sess.merge(
                        PhotoEmbedding(
                            photo_hash=digest, dim=int(vec.shape[-1]), vec=vec_to_bytes(vec)
                        )
                    )
            progress.advance(task, advance=len(chunk))
    console.print(f"  memory: {memory_summary()}")


def _stage_aesthetic(items):
    if not items:
        return
    console.print(f"[cyan]Stage A.2 — Aesthetic v2.5 ({len(items)} photos)[/cyan]")
    try:
        from app.scoring.aesthetic import AestheticPredictor
    except Exception as exc:
        console.print(f"[yellow]Aesthetic predictor unavailable: {exc}[/yellow]")
        return
    with AestheticPredictor() as predictor, Progress() as progress:
        task = progress.add_task("aesthetic", total=len(items))
        for chunk in chunked(items, settings.clip_batch):
            imgs = [Image.open(p).convert("RGB") for _, p in chunk]
            scores = predictor.score_batch(imgs)
            with session_scope() as sess:
                for (digest, _), score in zip(chunk, scores, strict=True):
                    sess.execute(
                        update(Photo)
                        .where(Photo.hash == digest)
                        .values(aesthetic_score=float(score))
                    )
            progress.advance(task, advance=len(chunk))
    console.print(f"  memory: {memory_summary()}")


def _stage_iqa(items):
    if not items:
        return
    console.print(f"[cyan]Stage B.1 — MUSIQ ({len(items)} photos)[/cyan]")
    musiq_scores = {}
    with MusiqScorer() as musiq, Progress() as progress:
        task = progress.add_task("musiq", total=len(items))
        for digest, p in items:
            musiq_scores[digest] = musiq.score(Image.open(p))
            progress.advance(task)
    console.print(f"  memory: {memory_summary()}")

    console.print(f"[cyan]Stage B.2 — MANIQA ({len(items)} photos)[/cyan]")
    maniqa_scores = {}
    with ManiqaScorer() as maniqa, Progress() as progress:
        task = progress.add_task("maniqa", total=len(items))
        for digest, p in items:
            maniqa_scores[digest] = maniqa.score(Image.open(p))
            progress.advance(task)
    console.print(f"  memory: {memory_summary()}")

    musiq_vals = list(musiq_scores.values())
    maniqa_vals = list(maniqa_scores.values())
    if not musiq_vals:
        return
    musiq_norm = dict(
        zip(musiq_scores, normalize(musiq_vals, min(musiq_vals), max(musiq_vals)), strict=True)
    )
    maniqa_norm = dict(
        zip(maniqa_scores, normalize(maniqa_vals, min(maniqa_vals), max(maniqa_vals)), strict=True)
    )
    with session_scope() as sess:
        for digest in musiq_scores:
            tech = 0.5 * musiq_norm[digest] + 0.5 * maniqa_norm.get(digest, 0.0)
            sess.execute(
                update(Photo)
                .where(Photo.hash == digest)
                .values(
                    musiq_score=musiq_scores[digest],
                    maniqa_score=maniqa_scores[digest],
                    technical_score=float(tech),
                )
            )


def _stage_faces(items):
    if not items:
        return
    console.print(f"[cyan]Stage C — InsightFace ({len(items)} photos)[/cyan]")
    with FaceDetector() as fd, Progress() as progress:
        task = progress.add_task("faces", total=len(items))
        for chunk in chunked(items, 8):
            face_rows = []
            for digest, p in chunk:
                for det in fd.detect(Path(p)):
                    x, y, w, h = det.bbox
                    face_rows.append(
                        Face(
                            photo_hash=digest,
                            bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                            det_score=det.det_score,
                            embedding=det.embedding.tobytes(),
                        )
                    )
            if face_rows:
                with session_scope() as sess:
                    sess.add_all(face_rows)
            progress.advance(task, advance=len(chunk))
    console.print(f"  memory: {memory_summary()}")


def run_scoring(stage: str = "all") -> None:
    if not assert_gpu_or_skip():
        return
    warmup()
    stage = stage.lower()
    if stage in {"clip", "all"}:
        _stage_clip(_photos_missing_embedding())
        _stage_aesthetic(_photos_missing("aesthetic_score"))
    if stage in {"iqa", "all"}:
        _stage_iqa(_photos_missing("technical_score"))
    if stage in {"faces", "all"}:
        _stage_faces(_photos_missing_faces())
    console.print("[green]Scoring complete.[/green]")
