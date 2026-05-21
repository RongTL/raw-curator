"""Enhancement orchestrator — drives the Auto Enhancement Engine per photo.

For each photo whose decision action is in {keep_and_enhance, enhance_only}:

    1. darktable develops the RAW to a 16-bit linear TIFF.
    2. Load that TIFF as float32 RGB in [0, 1]. (uint16 -> /65535)
    3. Measure quality across spec §1-§5 dimensions.
    4. Score the measurements into a QualityReport (§6 weights).
    5. Plan an ordered sequence of enhancement steps (§7 order).
    6. Execute the plan via engine.runner.run_plan, which handles:
       - classical steps at full native resolution in float32
       - AI steps on a VRAM-fitted downscale, then resampled back
       - VRAM cache eviction between GPU stages
    7. Write the result as a 16-bit linear TIFF in photos/exported/.
    8. If the decision was "no" (enhance_only), delete the source RAW
       only after the output TIFF exists on disk.

Hardware fit (Ryzen 3 3100 + 24 GB RAM + RTX 2060 6 GB):
- Full-res 24 MP float32 RGB = ~290 MB; well within 24 GB.
- AI step downscale (enhance_ai_scale=0.7) keeps SCUNet/Real-ESRGAN under 6 GB.
- cv2 kernels release the GIL — they saturate all 8 threads automatically.
"""

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
from app.enhancement.develop_full import darktable_cli
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.plan import QualityReport
from app.enhancement.engine.runner import run_plan
from app.enhancement.pack_tiff import write_tiff16
from app.models import Decision, Face, Photo, PhotoQualityReport

log = logging.getLogger(__name__)
console = Console()

_ENHANCE_ACTIONS = ("keep_and_enhance", "enhance_only")


def _candidates() -> list[tuple[dict, list[tuple[int, int, int, int]]]]:
    """Detached snapshots safe to consume after the session closes.

    Returns (photo_dict, face_boxes). face_boxes is a list of (x, y, w, h).
    """
    snapshots: list[tuple[dict, list[tuple[int, int, int, int]]]] = []
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo.hash, Photo.source_path, Photo.file_kind, Decision.action)
            .join(Decision, Photo.hash == Decision.photo_hash)
            .where(Decision.action.in_(_ENHANCE_ACTIONS))
        ).all()
        faces_by_hash: dict[str, list[tuple[int, int, int, int]]] = {}
        face_rows = sess.execute(
            select(Face.photo_hash, Face.bbox_x, Face.bbox_y, Face.bbox_w, Face.bbox_h)
        ).all()
        for digest, x, y, w, h in face_rows:
            faces_by_hash.setdefault(digest, []).append((int(x), int(y), int(w), int(h)))
        for digest, source_path, file_kind, action in rows:
            snapshots.append(
                (
                    {
                        "hash": digest,
                        "source_path": source_path,
                        "file_kind": file_kind,
                        "action": action,
                    },
                    faces_by_hash.get(digest, []),
                )
            )
    return snapshots


def _xmp_for(source_path: str) -> Path | None:
    src = Path(source_path)
    candidate = settings.xmp / (src.stem + ".xmp")
    return candidate if candidate.exists() else None


def _persist_report(photo_hash: str, report: QualityReport) -> None:
    """Upsert a quality_reports row for this photo. Wiped by `make reset`."""
    with session_scope() as sess:
        existing = sess.get(PhotoQualityReport, photo_hash)
        if existing is None:
            existing = PhotoQualityReport(photo_hash=photo_hash)
            sess.add(existing)
        existing.mean_luma = report.mean_luma
        existing.shadow_clip = report.shadow_clip
        existing.highlight_clip = report.highlight_clip
        existing.midtone_ratio = report.midtone_ratio
        existing.midtone_deviation = report.midtone_deviation
        existing.dr_p95_p5 = report.dr_p95_p5
        existing.local_dr_mean = report.local_dr_mean
        existing.rg_ratio = report.rg_ratio
        existing.bg_ratio = report.bg_ratio
        existing.avg_saturation = report.avg_saturation
        existing.oversat_ratio = report.oversat_ratio
        existing.skin_hue_var = report.skin_hue_var
        existing.lap_var = report.lap_var
        existing.edge_density = report.edge_density
        existing.hf_energy = report.hf_energy
        existing.luma_noise = report.luma_noise
        existing.chroma_noise = report.chroma_noise
        existing.score_exposure = report.score_exposure
        existing.score_dynamic_range = report.score_dynamic_range
        existing.score_color = report.score_color
        existing.score_sharpness = report.score_sharpness
        existing.score_noise = report.score_noise
        existing.score_q = report.score_q


def _load_linear_float(tiff_path: Path) -> np.ndarray:
    """Load a 16-bit linear TIFF (from darktable_cli) as float32 RGB in [0,1]."""
    arr = np.asarray(Image.open(tiff_path))
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Drop alpha if present.
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        else:
            raise ValueError(f"expected HxWx3, got {arr.shape} from {tiff_path}")
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * (1.0 / 65535.0)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _enhance_one(photo: dict, face_boxes: list[tuple[int, int, int, int]]) -> Path | None:
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
    rgb_f01 = _load_linear_float(full_tiff)
    native_h, native_w = rgb_f01.shape[:2]

    metrics = measure_all(rgb_f01, face_boxes=face_boxes if face_boxes else None)
    report = score_report(metrics)
    log.info(
        "engine quality %s Q=%.1f (E=%.1f D=%.1f C=%.1f S=%.1f N=%.1f)",
        src.name,
        report.score_q,
        report.score_exposure,
        report.score_dynamic_range,
        report.score_color,
        report.score_sharpness,
        report.score_noise,
    )
    _persist_report(photo["hash"], report)

    plan = plan_from_report(
        report,
        has_faces=bool(face_boxes),
        enhance_codeformer_w=settings.enhance_codeformer_w,
        enhance_realesrgan_fidelity=settings.enhance_realesrgan_fidelity,
        enhance_denoise_strength=settings.enhance_denoise_strength,
        backlit_shadow_lift=settings.enhance_backlit_shadow_lift,
        backlit_highlight_protect=settings.enhance_backlit_highlight_protect,
    )
    console.print(
        f"[dim]{src.name}[/dim] Q=[cyan]{report.score_q:.1f}[/cyan] -> "
        f"{len(plan.steps)} step(s)"
    )

    result_f01 = run_plan(rgb_f01, plan, native_size=(native_w, native_h))

    out = settings.photos / "exported" / (src.stem + ".tif")
    # write_tiff16 expects uint8 currently; the existing wrapper handles conversion.
    write_tiff16((result_f01 * 65535.0 + 0.5).clip(0, 65535).astype(np.uint16), out)

    try:
        full_tiff.unlink()
    except OSError:
        pass
    if photo.get("action") == "enhance_only" and out.exists():
        try:
            src.unlink()
            log.info("deleted no-RAW source after enhance: %s", src)
        except OSError as exc:
            log.warning("failed to delete no-RAW source %s: %s", src, exc)
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
        for photo, face_boxes in items:
            out = _enhance_one(photo, face_boxes)
            if out:
                enhanced += 1
                console.print(f"  -> {out}")
            else:
                skipped += 1
            progress.advance(task)
    console.print(
        f"[green]Enhancement complete:[/green] enhanced={enhanced} skipped={skipped}"
    )
