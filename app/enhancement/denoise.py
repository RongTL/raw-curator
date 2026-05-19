"""SCUNet denoiser. Lazy import — only loaded when this stage runs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def scunet_denoise(rgb: np.ndarray) -> np.ndarray:
    weights = Path("/data/models/scunet_color_real_psnr.pth")
    if not weights.exists():
        log.info("scunet weights missing at %s — skipping denoise", weights)
        return rgb
    try:
        import torch  # type: ignore
        from basicsr.archs.scunet_arch import SCUNet  # type: ignore
    except Exception as exc:  # noqa: BLE001
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
    out = (y.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
    del model, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
