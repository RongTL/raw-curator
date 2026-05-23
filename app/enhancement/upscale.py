"""Real-ESRGAN x2 upscaler with optional Lanczos fidelity blend.

Real-ESRGAN x2plus is a GAN — it produces sharp results but can
over-sharpen natural textures (skin, foliage, sky), giving a fake-detail
look. Blending the GAN output with a Lanczos upscale of the same input
pulls the result toward photographic naturalness while keeping most of
the resolution recovery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.enhancement.downsample import lanczos_resize

log = logging.getLogger(__name__)


def _lanczos_x2(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return lanczos_resize(rgb, (w * 2, h * 2))


def realesrgan_x2(
    rgb: np.ndarray,
    tile_size: int = 768,
    tile_pad: int = 16,
    fidelity: float = 1.0,
) -> np.ndarray:
    """x2 upscale with a fidelity blend.

    `fidelity` in [0, 1]: 1.0 is pure Real-ESRGAN (original behaviour),
    0.0 is pure Lanczos. Values around 0.5 keep most of the AI's detail
    recovery while softening its over-sharpened edges.
    """
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    if fidelity <= 0.0:
        return _lanczos_x2(rgb)

    try:
        from realesrgan import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except ImportError as exc:
        log.warning("realesrgan imports failed: %s — falling back to Lanczos x2", exc)
        return _lanczos_x2(rgb)
    weights = Path("/data/models/RealESRGAN_x2plus.pth")
    if not weights.exists():
        log.warning("realesrgan weights missing at %s — falling back to Lanczos x2", weights)
        return _lanczos_x2(rgb)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path=str(weights),
        model=model,
        tile=tile_size,
        tile_pad=tile_pad,
        pre_pad=0,
        half=True,
    )
    ai_out, _ = upsampler.enhance(rgb, outscale=2)

    if fidelity >= 1.0:
        return ai_out

    lanczos_out = _lanczos_x2(rgb)
    if lanczos_out.shape != ai_out.shape:
        from PIL import Image
        h, w = ai_out.shape[:2]
        lanczos_out = np.asarray(
            Image.fromarray(lanczos_out).resize((w, h), Image.Resampling.LANCZOS)
        )
    blended = ai_out.astype(np.float32) * fidelity + lanczos_out.astype(np.float32) * (1.0 - fidelity)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
