"""Step-major enhancement of a whole batch: develop+plan+pre-AI for every photo, then each AI
model once over the photos that need it, then post-AI+verify+write. Intermediates are float16
linear Rec.2020 .npy files under cache/enhance/."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rich.console import Console
from rich.progress import Progress

from app.arrays import Array
from app.config import settings
from app.db import session_scope
from app.enhancement.classical.lens_correct import correct_lens
from app.enhancement.denoise import ScunetModel
from app.enhancement.develop_full import darktable_cli, read_icc_profile
from app.enhancement.downsample import resize_float
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.metrics import luma_u8
from app.enhancement.engine.plan import EnhancementPlan, FaceInfo, QualityReport, StepSpec
from app.enhancement.engine.runner import (
    ai_steps,
    apply_ai_delta,
    apply_post_ai,
    apply_pre_ai,
    run_plan,
)
from app.enhancement.enhance_job import (
    EnhanceSummary,
    FaceBox,
    PhotoCandidate,
    _load_linear_float,
    persist_plan,
    persist_report,
    persist_verdict,
    preview_size,
)
from app.enhancement.face_restore import CodeFormerModel
from app.enhancement.geometry import scale_boxes
from app.enhancement.render_jpeg import render_paths, write_render
from app.enhancement.sidecar import resolve_xmp
from app.enhancement.upsample_final import parse_target
from app.enhancement.upscale import RealEsrganModel
from app.enhancement.verify import Verdict, safe_plan, verify
from app.ingest.exif import ExifData

log = logging.getLogger(__name__)
console = Console()

AI_ORDER: tuple[str, ...] = ("scunet_denoise", "realesrgan_upscale", "codeformer_restore")


class ModelLike(Protocol):
    def __enter__(self) -> ModelLike: ...
    def __exit__(self, *exc: object) -> None: ...
    def apply(self, rgb: Array, **params: Any) -> Array: ...


AI_MODELS: dict[str, Callable[[], ModelLike]] = {
    "scunet_denoise": ScunetModel,
    "realesrgan_upscale": RealEsrganModel,
    "codeformer_restore": CodeFormerModel,
}


@dataclass
class WorkItem:
    photo: PhotoCandidate
    face_boxes: list[FaceBox]
    intermediate: Path
    faces: list[FaceInfo] = field(default_factory=list)
    report: QualityReport | None = None
    plan: EnhancementPlan | None = None
    native_size: tuple[int, int] = (0, 0)
    icc: bytes | None = None
    ai_pending: list[StepSpec] = field(default_factory=list)
    ai_scaled: bool = False
    failed: str | None = None
    out: Path | None = None


def _face_infos(img: Array, boxes: list[FaceBox]) -> list[FaceInfo]:
    """Per-face sharpness on the developed frame: Laplacian variance of the
    sRGB-encoded luma of each crop. img is float32 linear RGB in [0, 1]."""
    import cv2

    h, w = img.shape[:2]
    faces: list[FaceInfo] = []
    for x, y, bw, bh in boxes:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + bw, w), min(y + bh, h)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img[y0:y1, x0:x1]
        lap = cv2.Laplacian(luma_u8(crop), ddepth=cv2.CV_32F, ksize=3)
        faces.append(FaceInfo((x0, y0, x1 - x0, y1 - y0), float(lap.var())))
    return faces


def plan_for(report: QualityReport, *, native_size: tuple[int, int], **kw: Any) -> EnhancementPlan:
    native_long_edge = max(native_size)
    target = parse_target(settings.enhance_target_res, native_size)
    target_scale = max(target) / native_long_edge
    return plan_from_report(
        report,
        denoise=settings.enhance_denoise,
        face_restore=settings.enhance_face_restore,
        backlit_recovery=settings.enhance_backlit_recovery,
        enhance_codeformer_w=settings.enhance_codeformer_w,
        enhance_realesrgan_fidelity=settings.enhance_realesrgan_fidelity,
        enhance_denoise_strength=settings.enhance_denoise_strength,
        backlit_shadow_lift=settings.enhance_backlit_shadow_lift,
        backlit_highlight_protect=settings.enhance_backlit_highlight_protect,
        native_long_edge=native_long_edge,
        target_scale=target_scale,
        sr_min_long_edge=settings.enhance_sr_min_long_edge,
        face_restore_max_px=settings.enhance_face_restore_max_px,
        **kw,
    )


def persist_all(item: WorkItem, verdict: Verdict | None = None) -> None:
    with session_scope() as sess:
        if item.report is not None:
            persist_report(sess, item.photo.hash, item.report)
        if item.plan is not None:
            persist_plan(sess, item.photo.hash, item.plan)
        if verdict is not None:
            persist_verdict(sess, item.photo.hash, verdict)


def _exif_for(photo: PhotoCandidate) -> ExifData:
    """The lens/camera fields lens correction needs, from the ingest DB snapshot."""
    return ExifData(
        camera_make=photo.camera_make,
        camera_body=photo.camera_body,
        lens=photo.lens,
        aperture=photo.aperture,
        focal_length=photo.focal_length,
    )


def _phase1(item: WorkItem, develop: Callable[..., Path]) -> None:
    src = Path(item.photo.source_path)
    xmp = resolve_xmp(src, photos_root=settings.photos, xmp_root=settings.xmp)
    dev = develop(src, xmp)
    try:
        item.icc = read_icc_profile(dev)
        if item.icc is None:
            log.warning(
                "developed TIFF for %s has no ICC profile; master will be untagged", src.name
            )
        img = _load_linear_float(dev)
    finally:
        with contextlib.suppress(OSError):
            dev.unlink()
    # Distortion + CA correction per-frame from EXIF, before measuring/planning so
    # every downstream step sees the geometrically-correct frame. No-op for lenses
    # lensfun cannot resolve. Shape is preserved, so native_size stays valid. EXIF
    # comes from the ingest DB snapshot (no per-frame disk read). Face boxes were
    # detected on the *uncorrected* preview, so a face near a frame edge sits a few
    # pixels off its true corrected position (bounded by the distortion at that
    # radius; sub-percent near center, low single-digit percent at the corners) —
    # acceptable, as CodeFormer re-aligns within the supplied region.
    img = correct_lens(img, _exif_for(item.photo), enabled=settings.enhance_lens_correction)
    h, w = img.shape[:2]
    item.native_size = (w, h)
    size = preview_size(Path(item.photo.preview_path) if item.photo.preview_path else None)
    if size:
        item.face_boxes = scale_boxes(item.face_boxes, size, (w, h))
    else:
        if item.face_boxes:
            log.warning(
                "preview missing for %s; ignoring %d face box(es)",
                src.name,
                len(item.face_boxes),
            )
        item.face_boxes = []
    item.faces = _face_infos(img, item.face_boxes)
    # `img` is the developed + lens-corrected frame, before any enhancement: the true
    # "before". Render it now (review-res + full-res) so the viewer has a baseline.
    paths = render_paths(item.photo.hash)
    write_render(
        img,
        paths["before"],
        long_edge=settings.review_long_edge,
        quality=settings.jpeg_quality_preview,
    )
    write_render(
        img,
        paths["before_full"],
        long_edge=0,
        quality=settings.jpeg_quality,
        source=item.photo.source_path,
    )
    item.report = score_report(measure_all(img, face_boxes=item.face_boxes or None))
    item.plan = plan_for(
        item.report,
        faces=item.faces,
        iso=item.photo.iso,
        native_size=item.native_size,
    )
    item.ai_pending = ai_steps(item.plan)
    persist_all(item)
    img = apply_pre_ai(img, item.plan)
    item.intermediate.parent.mkdir(parents=True, exist_ok=True)
    np.save(item.intermediate, img.astype(np.float16))


def _phase2(items: list[WorkItem]) -> None:
    for name in AI_ORDER:
        todo = [
            it for it in items if it.failed is None and any(s.name == name for s in it.ai_pending)
        ]
        if not todo:
            continue
        console.print(f"[cyan]AI step {name}: {len(todo)} photo(s)[/cyan]")
        try:
            model_cm = AI_MODELS[name]()
            model = model_cm.__enter__()
        except Exception as exc:  # noqa: BLE001 - a model that won't load fails only its photos
            log.exception("AI model %s failed to load", name)
            for it in todo:
                it.failed = f"{name}: model load failed: {exc}"
            continue
        try:
            for it in todo:
                try:
                    step = next(s for s in it.ai_pending if s.name == name)
                    img = np.load(it.intermediate).astype(np.float32)
                    if settings.enhance_ai_scale < 0.999 and not it.ai_scaled:
                        h, w = img.shape[:2]
                        s = settings.enhance_ai_scale
                        img = resize_float(img, (round(w * s), round(h * s)))
                        it.ai_scaled = True
                    params = dict(step.params)
                    if name == "codeformer_restore":
                        params["faces"] = scale_boxes(
                            it.face_boxes, it.native_size, (img.shape[1], img.shape[0])
                        )
                        params["min_similarity"] = settings.enhance_face_min_similarity
                    img = apply_ai_delta(
                        img,
                        partial(model.apply, **params),
                        scale=2 if name == "realesrgan_upscale" else 1,
                    )
                    np.save(it.intermediate, img.astype(np.float16))
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    it.failed = f"{name}: {exc}"
                    log.exception("AI step %s failed for %s", name, it.photo.source_path)
        finally:
            model_cm.__exit__(None, None, None)


def _take_retry(
    primary_q: float, primary_degraded: bool, retry_q: float, retry_degraded: bool
) -> bool:
    """Whether the safe-plan retry should replace the primary result.

    Prefer a *trustworthy* result: take the retry when it is not degraded and
    the primary is; when both share the same degraded state, fall back to
    quality (the retry wins on a tie). Never trade a clean primary for a
    degraded retry.
    """
    if not retry_degraded and primary_degraded:
        return True
    if primary_degraded == retry_degraded:
        return retry_q >= primary_q
    return False


def _phase3(item: WorkItem, develop: Callable[..., Path]) -> None:
    assert item.plan is not None and item.report is not None
    src = Path(item.photo.source_path)
    img = np.load(item.intermediate).astype(np.float32)
    if item.ai_pending:
        target = parse_target(settings.enhance_target_res, item.native_size)
        h, w = img.shape[:2]
        if (w, h) != target:
            img = resize_float(img, target)
    result = apply_post_ai(img, item.plan)
    after = score_report(measure_all(result, face_boxes=item.face_boxes or None))
    verdict = verify(item.report, after, result)
    plan = item.plan
    if verdict.degraded:
        log.warning(
            "%s degraded (%s); retrying with the safe plan", src.name, ",".join(verdict.reasons)
        )
        # The safe plan has no AI steps, so it can run from the original development again.
        dev = develop(src, resolve_xmp(src, photos_root=settings.photos, xmp_root=settings.xmp))
        try:
            base = _load_linear_float(dev)
        finally:
            with contextlib.suppress(OSError):
                dev.unlink()
        base = correct_lens(base, _exif_for(item.photo), enabled=settings.enhance_lens_correction)
        retry = safe_plan(plan)
        retry_img = run_plan(base, retry, native_size=item.native_size)
        retry_after = score_report(measure_all(retry_img, face_boxes=item.face_boxes or None))
        retry_verdict = verify(item.report, retry_after, retry_img)
        if _take_retry(
            after.score_q, verdict.degraded, retry_after.score_q, retry_verdict.degraded
        ):
            result, verdict, plan = retry_img, retry_verdict, retry
    item.plan = plan
    persist_all(item, verdict)
    # Write the "after" (enhanced) renders. No TIFF and no RAW disposition here:
    # the export stage moves/deletes originals once the curator picks a keep set.
    paths = render_paths(item.photo.hash)
    write_render(
        result,
        paths["after"],
        long_edge=settings.review_long_edge,
        quality=settings.jpeg_quality_preview,
    )
    write_render(
        result,
        paths["after_full"],
        long_edge=0,
        quality=settings.jpeg_quality,
        source=item.photo.source_path,
    )
    item.out = paths["after_full"]


def run_batch(
    items_in: list[tuple[PhotoCandidate, list[FaceBox]]],
    *,
    develop: Callable[..., Path] = darktable_cli,
) -> EnhanceSummary:
    if not items_in:
        console.print("[yellow]No photos to enhance.[/yellow]")
        return EnhanceSummary()
    work_dir = settings.cache / "enhance"
    items = [WorkItem(p, list(f), work_dir / f"{p.hash}.npy") for p, f in items_in]
    skipped = 0
    with Progress() as progress:
        task = progress.add_task("enhance: develop + plan", total=len(items))
        for it in items:
            src = Path(it.photo.source_path)
            kind = it.photo.file_kind
            if not src.exists():
                log.warning("skipping %s: source missing", src)
                it.failed = "skipped"
                skipped += 1
            elif kind not in (None, "raw"):
                log.warning(
                    "skipping %s: file_kind=%s is not RAW; the AI chain needs sensor data",
                    src,
                    kind,
                )
                it.failed = "skipped"
                skipped += 1
            else:
                try:
                    _phase1(it, develop)
                except Exception as exc:  # noqa: BLE001
                    it.failed = f"develop/plan: {exc}"
                    log.exception("enhance phase 1 failed for %s", src)
            progress.advance(task)
    _phase2(items)
    enhanced = 0
    with Progress() as progress:
        task = progress.add_task("enhance: finish + write", total=len(items))
        for it in items:
            if it.failed is None:
                try:
                    _phase3(it, develop)
                    enhanced += 1
                    console.print(f"  -> {it.out}")
                except Exception as exc:  # noqa: BLE001
                    it.failed = f"finish: {exc}"
                    log.exception("enhance phase 3 failed for %s", it.photo.source_path)
            with contextlib.suppress(OSError):
                it.intermediate.unlink()
            progress.advance(task)
    failed = sum(1 for it in items if it.failed not in (None, "skipped"))
    colour = "red" if failed else "green"
    console.print(
        f"[{colour}]Enhancement complete:[/{colour}] "
        f"enhanced={enhanced} skipped={skipped} failed={failed}"
    )
    return EnhanceSummary(enhanced=enhanced, skipped=skipped, failed=failed)
