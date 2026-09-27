"""Step-major enhancement of a whole batch: develop+plan+pre-AI for every photo, then each AI
model once over the photos that need it, then post-AI+verify+write. Intermediates are float16
linear Rec.2020 .npy files under cache/enhance/."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rich.console import Console
from rich.progress import Progress

from app.arrays import Array
from app.config import settings
from app.db import session_scope
from app.enhancement.denoise import ScunetModel
from app.enhancement.develop_full import darktable_cli, read_icc_profile
from app.enhancement.downsample import scale as lanczos_scale
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.plan import EnhancementPlan, QualityReport, StepSpec
from app.enhancement.engine.runner import (
    _from_u8,
    _to_u8,
    ai_steps,
    apply_post_ai,
    apply_pre_ai,
    run_plan,
)
from app.enhancement.enhance_job import (
    EnhanceSummary,
    FaceBox,
    PhotoCandidate,
    _load_linear_float,
    may_delete_source,
    persist_plan,
    persist_report,
    persist_verdict,
    preview_size,
)
from app.enhancement.face_restore import CodeFormerModel
from app.enhancement.geometry import scale_boxes
from app.enhancement.pack_tiff import copy_metadata, write_tiff16
from app.enhancement.sidecar import resolve_xmp
from app.enhancement.upsample_final import upsample_final
from app.enhancement.upscale import RealEsrganModel
from app.enhancement.verify import Verdict, safe_plan, verify
from app.paths import relative_subpath

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
    report: QualityReport | None = None
    plan: EnhancementPlan | None = None
    native_size: tuple[int, int] = (0, 0)
    icc: bytes | None = None
    ai_pending: list[StepSpec] = field(default_factory=list)
    ai_scaled: bool = False
    failed: str | None = None
    out: Path | None = None


def plan_for(report: QualityReport, **kw: Any) -> EnhancementPlan:
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
    h, w = img.shape[:2]
    item.native_size = (w, h)
    size = preview_size(Path(item.photo.preview_path) if item.photo.preview_path else None)
    item.face_boxes = scale_boxes(item.face_boxes, size, (w, h)) if size else []
    item.report = score_report(measure_all(img, face_boxes=item.face_boxes or None))
    item.plan = plan_for(item.report, has_faces=bool(item.face_boxes), iso=item.photo.iso)
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
        with AI_MODELS[name]() as model:
            for it in todo:
                try:
                    step = next(s for s in it.ai_pending if s.name == name)
                    img = np.load(it.intermediate).astype(np.float32)
                    u8 = _to_u8(img)
                    if settings.enhance_ai_scale < 0.999 and not it.ai_scaled:
                        u8 = lanczos_scale(u8, settings.enhance_ai_scale)
                        it.ai_scaled = True
                    u8 = model.apply(u8, **step.params)
                    np.save(it.intermediate, _from_u8(u8).astype(np.float16))
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    it.failed = f"{name}: {exc}"
                    log.exception("AI step %s failed for %s", name, it.photo.source_path)


def _phase3(item: WorkItem, develop: Callable[..., Path]) -> None:
    assert item.plan is not None and item.report is not None
    src = Path(item.photo.source_path)
    img = np.load(item.intermediate).astype(np.float32)
    if item.ai_pending:
        img = _from_u8(upsample_final(_to_u8(img), item.native_size))
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
        retry = safe_plan(plan)
        retry_img = run_plan(base, retry, native_size=item.native_size)
        retry_after = score_report(measure_all(retry_img, face_boxes=item.face_boxes or None))
        retry_verdict = verify(item.report, retry_after, retry_img)
        if retry_after.score_q >= after.score_q:
            result, verdict, plan = retry_img, retry_verdict, retry
    item.plan = plan
    persist_all(item, verdict)
    out = settings.photos / "exported" / relative_subpath(src, settings.photos).with_suffix(".tif")
    write_tiff16(result, out, icc=item.icc)
    copy_metadata(src, out)
    item.out = out
    if may_delete_source(item.photo, out, verdict):
        with contextlib.suppress(OSError):
            src.unlink()
            log.info("deleted no-RAW source after verified enhance: %s", src)
    elif item.photo.action == "enhance_only" and verdict.degraded:
        log.error(
            "KEEPING source RAW for %s: result degraded (%s)", src.name, ",".join(verdict.reasons)
        )


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
            if not src.exists() or (it.photo.file_kind not in (None, "raw")):
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
