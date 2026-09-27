"""SCUNet denoiser. Lazy import — only loaded when this stage runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from app.arrays import Array
from app.config import settings
from app.enhancement.weights import scunet_weights

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# SCUNet has no native tiling; running a full 4200x2800 frame through it on a
# 6 GB card OOMs in the deeper batchnorm layers. We process in non-overlapping
# output tiles (settings.scunet_tile) padded with reflective context
# (settings.scunet_tile_pad) so the model sees enough beyond each tile to avoid
# edge artifacts. multiple=64 satisfies SCUNet's window attention and 3-stage
# downsample divisibility.
_MULTIPLE = 64


def _tiled_forward(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    multiple: int = _MULTIPLE,
    tile: int | None = None,
    pad: int | None = None,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    tile = settings.scunet_tile if tile is None else tile
    pad = settings.scunet_tile_pad if pad is None else pad
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


class ScunetModel:
    """SCUNet loaded once for a batch. ``apply`` blends by ``strength`` like scunet_denoise did."""

    def __init__(self) -> None:
        self.available = False
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._dtype: Any = None

    def __enter__(self) -> ScunetModel:
        weights = scunet_weights()
        if not weights.exists():
            log.warning("scunet weights missing at %s — denoise disabled", weights)
            return self
        try:
            import torch

            from app.enhancement._scunet_arch import SCUNet
        except ImportError as exc:
            log.warning("scunet imports failed: %s — denoise disabled", exc)
            return self
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(  # type: ignore[no-untyped-call]
            self._device, dtype=self._dtype
        )
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("params") or ckpt.get("params_ema") or ckpt)
        model.eval()
        self._model = model
        self.available = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self.available = False

    def apply(self, rgb: Array, *, strength: float = 1.0, **_: object) -> Array:
        """Run SCUNet, then blend the denoised result with the input.

        `strength` in [0, 1]: 1.0 returns pure SCUNet, 0.0 returns the input
        unchanged. Values <1 retain a fraction of the original micro-texture
        so the output keeps natural sensor grain instead of looking plastic.
        """
        if strength <= 0.0 or not self.available:
            return rgb
        strength = float(min(1.0, strength))
        torch = self._torch
        x = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device, dtype=self._dtype)
        )
        y = _tiled_forward(self._model, x)
        denoised = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
        del x, y
        if strength >= 1.0:
            return denoised.astype(np.uint8)
        blended = denoised * strength + rgb.astype(np.float32) * (1.0 - strength)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def scunet_denoise(rgb: Array, strength: float = 1.0) -> Array:
    with ScunetModel() as m:
        return m.apply(rgb, strength=strength)
