"""SCUNet denoiser. Lazy import — only loaded when this stage runs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


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
    with torch.no_grad():
        y = model(x).clamp(0.0, 1.0)
    denoised = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
    del model, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if strength >= 1.0:
        return denoised.astype(np.uint8)
    blended = denoised * strength + rgb.astype(np.float32) * (1.0 - strength)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
