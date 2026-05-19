"""Real-ESRGAN x2 upscaler."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def realesrgan_x2(rgb: np.ndarray, tile_size: int = 512, tile_pad: int = 16) -> np.ndarray:
    try:
        from realesrgan import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception:
        return rgb  # no-op fallback
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    weights = Path("/data/models/RealESRGAN_x2plus.pth")
    if not weights.exists():
        return rgb
    upsampler = RealESRGANer(
        scale=2,
        model_path=str(weights),
        model=model,
        tile=tile_size,
        tile_pad=tile_pad,
        pre_pad=0,
        half=True,
    )
    out, _ = upsampler.enhance(rgb, outscale=2)
    return out
