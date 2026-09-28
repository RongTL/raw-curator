"""Gray-world white balance (spec §3.1).

Computes the per-channel mean of the linear RGB image and scales R and B
so their ratios with G match the requested target. Default target is
(1.0, 1.0) — neutral. The `strength` knob (0..1) interpolates between
no-op and full correction, useful for not over-correcting deliberately
warm/cool scenes.
"""

from __future__ import annotations

import numpy as np

from app.arrays import Array
from app.enhancement.engine.metrics import neutral_mask


def gray_world(
    rgb: Array,
    target_rg: float = 1.0,
    target_bg: float = 1.0,
    strength: float = 1.0,
    neutral_only: bool = False,
) -> Array:
    if strength <= 0.0:
        return rgb
    if neutral_only:
        # Estimate the cast from the same near-neutral pixels the planner gated on,
        # so a saturated sky can't drag the gains. Fall back to the whole-frame
        # mean when too few neutrals exist (never divide by ~zero).
        mask = neutral_mask(rgb)
        avg = (
            rgb[mask].mean(axis=0)
            if float(mask.mean()) >= 0.02
            else rgb.reshape(-1, 3).mean(axis=0)
        )
    else:
        avg = rgb.reshape(-1, 3).mean(axis=0)
    g = max(float(avg[1]), 1e-6)
    rg = float(avg[0] / g)
    bg = float(avg[2] / g)
    if rg < 1e-6 or bg < 1e-6:
        return rgb
    scale_r = target_rg / rg
    scale_b = target_bg / bg
    s = float(np.clip(strength, 0.0, 1.0))
    scale_r = 1.0 * (1.0 - s) + scale_r * s
    scale_b = 1.0 * (1.0 - s) + scale_b * s
    out = rgb.copy()
    out[..., 0] = np.clip(rgb[..., 0] * scale_r, 0.0, 1.0)
    out[..., 2] = np.clip(rgb[..., 2] * scale_b, 0.0, 1.0)
    return out.astype(np.float32)
