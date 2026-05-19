"""Run cheap CPU filters over all photos missing them."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select, update

from app.config import settings
from app.db import session_scope
from app.filters.blur import laplacian_variance
from app.filters.exposure import exposure_flag
from app.filters.phash import dhash, phash
from app.models import Photo

log = logging.getLogger(__name__)
console = Console()


@dataclass
class FilterResult:
    hash: str
    blur_var: float
    phash: str
    dhash: str
    exposure_flag: str
    hist_mean: float


def _process(hash_and_thumb: tuple[str, str]) -> FilterResult | None:
    digest, thumb_path = hash_and_thumb
    try:
        rgb = np.asarray(Image.open(thumb_path).convert("RGB"))
        flag, mean = exposure_flag(rgb)
        return FilterResult(
            hash=digest,
            blur_var=laplacian_variance(rgb),
            phash=phash(rgb),
            dhash=dhash(rgb),
            exposure_flag=flag,
            hist_mean=mean,
        )
    except Exception:
        log.exception("filter failed for %s", thumb_path)
        return None


def run_filters() -> None:
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.thumb_path).where(Photo.phash.is_(None))
        ).all()
    candidates = [(h, t) for h, t in rows if t and Path(t).exists()]
    if not candidates:
        console.print("[yellow]Nothing to filter (all photos already have phash).[/yellow]")
        return

    workers = max(1, settings.cpu_workers)
    results: list[FilterResult] = []
    with Progress() as progress:
        task = progress.add_task("filter", total=len(candidates))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, item) for item in candidates]
            for fut in as_completed(futures):
                r = fut.result()
                progress.advance(task)
                if r is not None:
                    results.append(r)

    with session_scope() as sess:
        for r in results:
            sess.execute(
                update(Photo)
                .where(Photo.hash == r.hash)
                .values(
                    blur_var=r.blur_var,
                    phash=r.phash,
                    dhash=r.dhash,
                    exposure_flag=r.exposure_flag,
                    hist_mean=r.hist_mean,
                )
            )
    console.print(f"[green]Filtered {len(results)} photo(s).[/green]")
