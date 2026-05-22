"""Exposure corrections in float32 linear RGB [0,1] (spec §1.5).

Three operations, designed to be layered:
- `gamma_correct(rgb, gain)` — global brightening via a gamma curve.
  Positive gain brightens midtones more than blacks/highlights (natural to the eye);
  negative darkens.
- `shadow_lift(rgb, amount)` — luminance-masked lift focused on the
  bottom third of the histogram.
- `highlight_recover(rgb, amount, knee)` — soft roll-off of values above
  `knee` so detail near 1.0 is reclaimed without a hard clip.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def _luma(rgb: np.ndarray) -> np.ndarray:
    return _LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1] + _LUMA_B * rgb[..., 2]


def gamma_correct(rgb: np.ndarray, gain: float) -> np.ndarray:
    if abs(gain) < 1e-4:
        return rgb
    k = float(gain)
    f = np.clip(rgb, 1e-6, 1.0)
    return np.power(f, 1.0 / (1.0 + k)).astype(np.float32)


def shadow_lift(rgb: np.ndarray, amount: float = 0.35) -> np.ndarray:
    if amount <= 0.0:
        return rgb
    L = _luma(rgb)
    mask = np.exp(-((L - 0.18) ** 2) / (2.0 * 0.18 ** 2)).astype(np.float32)
    k = float(amount) * mask
    boosted = np.power(np.clip(rgb, 1e-6, 1.0), 1.0 / (1.0 + k[..., None]))
    return boosted.astype(np.float32)


def highlight_recover(rgb: np.ndarray, amount: float = 0.5, knee: float = 0.75) -> np.ndarray:
    if amount <= 0.0:
        return rgb
    knee = float(np.clip(knee, 0.1, 0.99))
    amt = float(np.clip(amount, 0.0, 1.0))
    f = np.clip(rgb, 0.0, None).astype(np.float32)
    over = np.maximum(0.0, f - knee)
    range_ = max(1e-6, 1.0 - knee) * (1.0 - amt * 0.85)
    rolled = over / (1.0 + over / range_)
    # Only re-map highlights (f > knee); leave shadows and midtones unchanged.
    # Without the np.where guard, pixels below knee collapse to `knee + 0 = knee`,
    # flattening every shadow/midtone to a constant 0.78 plateau.
    rolled_full = np.clip(knee + rolled, 0.0, 1.0)
    return np.where(f > knee, rolled_full, f).astype(np.float32)
