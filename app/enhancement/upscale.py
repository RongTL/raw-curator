"""Real-ESRGAN x2 upscaler with feathered tiling and optional Lanczos fidelity blend.

Tiling is ours, not the realesrgan library's: ``tiled_apply`` runs the model
on overlapping windows and cross-fades the overlaps, so no hard tile
boundary survives in smooth gradients such as skies.

Real-ESRGAN x2plus is a GAN — it produces sharp results but can
over-sharpen natural textures (skin, foliage, sky), giving a fake-detail
look. Blending the GAN output with a Lanczos upscale of the same input
pulls the result toward photographic naturalness while keeping most of
the resolution recovery.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.arrays import Array
from app.enhancement.downsample import lanczos_resize
from app.enhancement.tiling import tiled_apply
from app.enhancement.weights import realesrgan_weights

log = logging.getLogger(__name__)


def _lanczos_x2(rgb: Array) -> Array:
    h, w = rgb.shape[:2]
    return lanczos_resize(rgb, (w * 2, h * 2))


class RealEsrganModel:
    """Real-ESRGAN x2 loaded once for a batch. ``apply`` blends by ``fidelity`` like realesrgan_x2 did."""

    def __init__(self, tile_size: int = 768, overlap: int = 32) -> None:
        self.available = False
        self._tile_size = tile_size
        self._overlap = overlap
        self._upsampler: Any = None
        self._torch: Any = None

    def __enter__(self) -> RealEsrganModel:
        weights = realesrgan_weights()
        if not weights.exists():
            log.warning("realesrgan weights missing at %s — falling back to Lanczos x2", weights)
            return self
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            log.warning("realesrgan imports failed: %s — falling back to Lanczos x2", exc)
            return self
        self._torch = torch
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
        )
        self._upsampler = RealESRGANer(
            scale=2,
            model_path=str(weights),
            model=model,
            tile=0,
            tile_pad=0,
            pre_pad=0,
            half=True,
        )
        self.available = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._upsampler = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self.available = False

    def apply(self, rgb: Array, *, fidelity: float = 1.0, **_: object) -> Array:
        """x2 upscale with a fidelity blend.

        `fidelity` in [0, 1]: 1.0 is pure Real-ESRGAN (original behaviour),
        0.0 is pure Lanczos. Values around 0.5 keep most of the AI's detail
        recovery while softening its over-sharpened edges.
        """
        fidelity = float(np.clip(fidelity, 0.0, 1.0))
        if not self.available or fidelity <= 0.0:
            return _lanczos_x2(rgb)

        ai_out = tiled_apply(
            rgb,
            lambda t: self._upsampler.enhance(t, outscale=2)[0],
            tile=self._tile_size,
            overlap=self._overlap,
            scale=2,
        )

        if fidelity >= 1.0:
            return ai_out

        lanczos_out = _lanczos_x2(rgb)
        if lanczos_out.shape != ai_out.shape:
            from PIL import Image

            h, w = ai_out.shape[:2]
            lanczos_out = np.asarray(
                Image.fromarray(lanczos_out).resize((w, h), Image.Resampling.LANCZOS)
            )
        blended = ai_out.astype(np.float32) * fidelity + lanczos_out.astype(np.float32) * (
            1.0 - fidelity
        )
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def realesrgan_x2(
    rgb: Array,
    tile_size: int = 768,
    overlap: int = 32,
    fidelity: float = 1.0,
) -> Array:
    with RealEsrganModel(tile_size=tile_size, overlap=overlap) as m:
        return m.apply(rgb, fidelity=fidelity)
