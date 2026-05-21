"""Execute an EnhancementPlan.

Step dispatch lives here. The runner manages three impedance mismatches:

1. **Bit depth**: classical steps operate on float32 RGB in [0, 1];
   AI steps (SCUNet, Real-ESRGAN, CodeFormer) take uint8 RGB.
   Conversions happen at AI boundaries only — the float state survives
   across consecutive classical steps without quantisation.

2. **Resolution**: AI steps need a downscaled copy that fits in 6 GB
   VRAM. Real-ESRGAN x2 already restores most of that resolution, and
   the final Lanczos upsample brings the result back to native.

3. **VRAM hygiene**: after each GPU step we call
   `torch.cuda.empty_cache()` so the next model fits in 6 GB.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings
from app.enhancement.classical import (
    color,
    exposure,
    local_contrast,
    sharpen,
    tone_map,
    white_balance,
)
from app.enhancement.denoise import scunet_denoise
from app.enhancement.downsample import scale as lanczos_scale
from app.enhancement.engine.plan import EnhancementPlan
from app.enhancement.face_restore import codeformer_restore
from app.enhancement.tone_balance import recover_backlit
from app.enhancement.upscale import realesrgan_x2
from app.enhancement.upsample_final import upsample_final

log = logging.getLogger(__name__)


def _free_gpu() -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _to_u8(rgb_f01: np.ndarray) -> np.ndarray:
    return (np.clip(rgb_f01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _from_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return (rgb_u8.astype(np.float32) * (1.0 / 255.0)).astype(np.float32)


_PRE_AI = {
    "exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover",
    "highlight_rolloff", "white_balance", "saturation_adjust",
}
_AI = {"scunet_denoise", "realesrgan_upscale", "codeformer_restore"}
_POST_AI = {"unsharp_mask", "clahe_local_contrast", "tone_map_final"}


def _apply_classical(name: str, rgb_f01: np.ndarray, params: dict) -> np.ndarray:
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
    if name == "tone_map_final":
        return tone_map.filmic_tone_map(
            rgb_f01,
            shoulder=params.get("shoulder", 0.88),
            toe=params.get("toe", 0.02),
        )
    raise ValueError(f"unknown classical step: {name}")


def _apply_ai(name: str, rgb_u8: np.ndarray, params: dict, has_faces: bool) -> np.ndarray:
    if name == "scunet_denoise":
        return scunet_denoise(rgb_u8, strength=params.get("strength", 0.75))
    if name == "realesrgan_upscale":
        return realesrgan_x2(rgb_u8, fidelity=params.get("fidelity", 0.7))
    if name == "codeformer_restore":
        if not has_faces:
            return rgb_u8
        return codeformer_restore(rgb_u8, faces=None, weight=params.get("weight", 0.85))
    raise ValueError(f"unknown AI step: {name}")


def run_plan(
    rgb_f01: np.ndarray,
    plan: EnhancementPlan,
    native_size: tuple[int, int],
) -> np.ndarray:
    """Execute the plan, returning float32 RGB in [0, 1] at native_size (W, H)."""
    img = np.clip(rgb_f01, 0.0, 1.0).astype(np.float32)

    # Pass 1: pre-AI classical steps at full native resolution.
    for step in plan.steps:
        if step.name in _PRE_AI:
            log.info("engine[pre-AI]  %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)

    ai_steps = [s for s in plan.steps if s.name in _AI]
    if ai_steps:
        u8 = _to_u8(img)
        if settings.enhance_ai_scale < 0.999:
            u8 = lanczos_scale(u8, settings.enhance_ai_scale)
        for step in ai_steps:
            log.info("engine[ai]      %s %s -- %s", step.name, step.params, step.reason)
            u8 = _apply_ai(step.name, u8, step.params, plan.has_faces)
            _free_gpu()
        u8 = upsample_final(u8, native_size)
        img = _from_u8(u8)
    else:
        target_w, target_h = native_size
        h, w = img.shape[:2]
        if (w, h) != (target_w, target_h):
            u8 = _to_u8(img)
            u8 = upsample_final(u8, native_size)
            img = _from_u8(u8)

    # Pass 2: post-AI classical steps at native resolution.
    for step in plan.steps:
        if step.name in _POST_AI:
            log.info("engine[post-AI] %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)

    return np.clip(img, 0.0, 1.0).astype(np.float32)
