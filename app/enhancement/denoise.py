"""SCUNet denoiser. Lazy import — only loaded when this stage runs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# SCUNet has no native tiling; running a full 4200x2800 frame through it on a
# 6 GB card OOMs in the deeper batchnorm layers. We process in non-overlapping
# output tiles padded with reflective context so the model sees enough beyond
# each tile to avoid edge artifacts. multiple=64 satisfies SCUNet's window
# attention and 3-stage downsample divisibility.
_TILE = int(os.environ.get("RAWCURATOR_SCUNET_TILE", "384"))
_PAD = int(os.environ.get("RAWCURATOR_SCUNET_TILE_PAD", "32"))
_MULTIPLE = 64


def _tiled_forward(model, x, multiple: int = _MULTIPLE, tile: int = _TILE, pad: int = _PAD):
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _, _, h_in, w_in = x.shape
    out = torch.zeros_like(x)
    for ty in range(0, h_in, tile):
        for tx in range(0, w_in, tile):
            y0 = max(0, ty - pad)
            x0 = max(0, tx - pad)
            y1 = min(h_in, ty + tile + pad)
            x1 = min(w_in, tx + tile + pad)
            tile_in = x[:, :, y0:y1, x0:x1]
            th, tw = tile_in.shape[2], tile_in.shape[3]
            pad_h = (-th) % multiple
            pad_w = (-tw) % multiple
            if pad_h or pad_w:
                tile_in = F.pad(tile_in, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.no_grad():
                tile_out = model(tile_in).clamp(0.0, 1.0)
            tile_out = tile_out[:, :, :th, :tw]
            iy0, ix0 = ty, tx
            iy1, ix1 = min(h_in, ty + tile), min(w_in, tx + tile)
            sy0, sx0 = iy0 - y0, ix0 - x0
            sy1, sx1 = sy0 + (iy1 - iy0), sx0 + (ix1 - ix0)
            out[:, :, iy0:iy1, ix0:ix1] = tile_out[:, :, sy0:sy1, sx0:sx1]
            del tile_in, tile_out
    return out


def scunet_denoise(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Run SCUNet, then blend the denoised result with the input.

    `strength` in [0, 1]: 1.0 returns pure SCUNet, 0.0 returns the input
    unchanged. Values <1 retain a fraction of the original micro-texture
    so the output keeps natural sensor grain instead of looking plastic.
    """
    if strength <= 0.0:
        return rgb
    strength = float(min(1.0, strength))

    weights = Path("/data/models/scunet_color_real_psnr.pth")
    if not weights.exists():
        log.warning("scunet weights missing at %s — skipping denoise", weights)
        return rgb
    try:
        import torch  # type: ignore
        from basicsr.archs.scunet_arch import SCUNet  # type: ignore
    except ImportError as exc:
        log.warning("scunet imports failed: %s — skipping denoise", exc)
        return rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(device, dtype=dtype)
    ckpt = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(ckpt.get("params") or ckpt.get("params_ema") or ckpt)
    model.eval()

    x = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device, dtype=dtype)
    )
    y = _tiled_forward(model, x)
    denoised = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
    del model, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if strength >= 1.0:
        return denoised.astype(np.uint8)
    blended = denoised * strength + rgb.astype(np.float32) * (1.0 - strength)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
