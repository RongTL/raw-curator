"""Execute an EnhancementPlan against a float32 linear Rec.2020 image.

The batch pipeline (``app/enhancement/batch.py``) drives the steps directly:
``apply_pre_ai`` runs the classical corrections before the AI stage,
``apply_ai_delta`` merges each AI model's *change* into the full-precision
float master at an AI boundary, and ``apply_post_ai`` runs the classical
polish afterwards. ``run_plan`` wires those together for the verifier's
safe-plan retry and any legacy single-image caller.

Two impedance mismatches are handled at the AI boundary only:

1. **Bit depth**: classical steps stay on float32 linear RGB in [0, 1];
   AI steps (SCUNet, Real-ESRGAN, CodeFormer) take uint8 sRGB. The float
   state survives across consecutive classical steps without quantisation.

2. **VRAM hygiene**: after each GPU step we call
   ``torch.cuda.empty_cache()`` so the next model fits in 6 GB.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np

from app.arrays import Array
from app.config import settings
from app.enhancement.classical import (
    color,
    exposure,
    local_contrast,
    sharpen,
    tone_map,
    white_balance,
)
from app.enhancement.colorspace import linear_rec2020_to_srgb_u8, srgb_u8_to_linear_rec2020
from app.enhancement.denoise import scunet_denoise
from app.enhancement.downsample import lanczos_resize, resize_float
from app.enhancement.engine.plan import EnhancementPlan, StepSpec
from app.enhancement.face_restore import codeformer_restore
from app.enhancement.tone_balance import recover_backlit
from app.enhancement.upsample_final import parse_target
from app.enhancement.upscale import realesrgan_x2

log = logging.getLogger(__name__)


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _to_u8(rgb_lin: Array) -> Array:
    """AI boundary: linear Rec.2020 float -> display-referred sRGB uint8."""
    return linear_rec2020_to_srgb_u8(rgb_lin)


def _from_u8(rgb_u8: Array) -> Array:
    return srgb_u8_to_linear_rec2020(rgb_u8)


def apply_ai_delta(
    x_lin: Array,
    model_fn: Callable[[Array], Array],
    *,
    strength: float = 1.0,
    scale: int = 1,
) -> Array:
    """Run an 8-bit sRGB model and merge only its *change* (delta) into the linear float image.

    The model sees `_to_u8(x_lin)` and returns 8-bit sRGB; we compute the delta the model
    introduced (in linear space) and add it to the *full-precision* float base, so the master
    keeps 16-bit precision instead of being quantised to the model's 8-bit output. When
    `scale > 1` (Real-ESRGAN x2) the model must return a `scale`x-larger image; the base and the
    model's 8-bit reference are Lanczos-upsampled so the delta lines up with the model's larger
    output. A model that does not honour `scale` raises `ValueError`.

    `x_lin` is float32 linear Rec.2020 in [0, 1]; returns the same, clipped.
    """
    x8 = _to_u8(x_lin)
    y8 = model_fn(x8)
    h, w = x_lin.shape[:2]
    if scale > 1:
        if y8.shape[:2] != (h * scale, w * scale):
            raise ValueError(
                f"model returned {y8.shape[:2]} for scale={scale}; expected {(h * scale, w * scale)}"
            )
        base = resize_float(x_lin, (w * scale, h * scale))  # full-precision Lanczos upsample
        ref8 = lanczos_resize(x8, (w * scale, h * scale))  # sRGB-domain, like the model saw
    else:
        base, ref8 = x_lin, x8
    delta = _from_u8(y8) - _from_u8(ref8)
    return np.clip(base + strength * delta, 0.0, 1.0).astype(np.float32)


_PRE_AI = {
    "exposure_gamma",
    "shadow_lift",
    "highlight_recover",
    "backlit_recover",
    "highlight_rolloff",
    "white_balance",
    "saturation_adjust",
}
_AI = {"scunet_denoise", "realesrgan_upscale", "codeformer_restore"}
_POST_AI = {"unsharp_mask", "clahe_local_contrast"}


def _apply_classical(name: str, rgb_f01: Array, params: Mapping[str, Any]) -> Array:
    if name == "exposure_gamma":
        return exposure.gamma_correct(rgb_f01, gain=params.get("gain", 0.0))
    if name == "shadow_lift":
        return exposure.shadow_lift(rgb_f01, amount=params.get("amount", 0.3))
    if name == "highlight_recover":
        return exposure.highlight_recover(
            rgb_f01,
            amount=params.get("amount", 0.5),
            knee=params.get("knee", 0.78),
        )
    if name == "backlit_recover":
        # The existing module expects uint8 sRGB. Convert in/out.
        u8 = _to_u8(rgb_f01)
        out = recover_backlit(
            u8,
            shadow_lift=params.get("shadow_lift", 0.4),
            highlight_protect=params.get("highlight_protect", 0.15),
            force=params.get("force", False),
        )
        return _from_u8(out)
    if name == "highlight_rolloff":
        return tone_map.global_compress(rgb_f01, strength=params.get("strength", 0.5))
    if name == "white_balance":
        return white_balance.gray_world(
            rgb_f01,
            target_rg=params.get("target_rg", 1.0),
            target_bg=params.get("target_bg", 1.0),
            strength=params.get("strength", 1.0),
            neutral_only=bool(params.get("neutral_only", False)),
        )
    if name == "saturation_adjust":
        return color.adjust_saturation(
            rgb_f01,
            factor=params.get("factor", 1.0),
            protect_skin=params.get("protect_skin", True),
        )
    if name == "unsharp_mask":
        return sharpen.unsharp_mask(
            rgb_f01,
            amount=params.get("amount", 0.6),
            radius=params.get("radius", 1.4),
            threshold=params.get("threshold", 0.006),
        )
    if name == "clahe_local_contrast":
        return local_contrast.apply_clahe(
            rgb_f01,
            clip_limit=params.get("clip_limit", 2.0),
            tile_grid=tuple(params.get("tile_grid", (8, 8))),
        )
    raise ValueError(f"unknown classical step: {name}")


def _apply_ai(name: str, rgb_u8: Array, params: Mapping[str, Any], has_faces: bool) -> Array:
    if name == "scunet_denoise":
        return scunet_denoise(rgb_u8, strength=params.get("strength", 0.75))
    if name == "realesrgan_upscale":
        return realesrgan_x2(rgb_u8, fidelity=params.get("fidelity", 0.7))
    if name == "codeformer_restore":
        if not has_faces:
            return rgb_u8
        return codeformer_restore(rgb_u8, weight=params.get("weight", 0.85))
    raise ValueError(f"unknown AI step: {name}")


def apply_pre_ai(img: Array, plan: EnhancementPlan) -> Array:
    """Pre-AI classical steps, applied to float32 linear RGB at native resolution."""
    for step in plan.steps:
        if step.name in _PRE_AI:
            log.info("engine[pre-AI]  %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)
    return img


def apply_post_ai(img: Array, plan: EnhancementPlan) -> Array:
    """Post-AI classical steps, returning clipped float32 RGB in [0, 1]."""
    for step in plan.steps:
        if step.name in _POST_AI:
            log.info("engine[post-AI] %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def ai_steps(plan: EnhancementPlan) -> list[StepSpec]:
    """The plan's AI steps, in order."""
    return [s for s in plan.steps if s.name in _AI]


def run_plan(
    rgb_f01: Array,
    plan: EnhancementPlan,
    native_size: tuple[int, int],
) -> Array:
    """Execute the plan, returning float32 RGB in [0, 1] at native_size (W, H)."""
    img = apply_pre_ai(rgb_f01, plan)

    ai = ai_steps(plan)
    if ai:
        if settings.enhance_ai_scale < 0.999:
            h, w = img.shape[:2]
            s = settings.enhance_ai_scale
            img = resize_float(img, (round(w * s), round(h * s)))
        for step in ai:
            log.info("engine[ai]      %s %s -- %s", step.name, step.params, step.reason)
            img = apply_ai_delta(
                img,
                partial(_apply_ai, step.name, params=step.params, has_faces=plan.has_faces),
                scale=2 if step.name == "realesrgan_upscale" else 1,
            )
            _free_gpu()
        target = parse_target(settings.enhance_target_res, native_size)
        h, w = img.shape[:2]
        if (w, h) != target:
            img = resize_float(img, target)
    else:
        target_w, target_h = native_size
        h, w = img.shape[:2]
        if (w, h) != (target_w, target_h):
            img = resize_float(img, native_size)

    return apply_post_ai(img, plan)
