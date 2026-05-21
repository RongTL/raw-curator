"""Gray-world white balance (spec §3.1).

Computes the per-channel mean of the linear RGB image and scales R and B
so their ratios with G match the requested target. Default target is
(1.0, 1.0) — neutral. The `strength` knob (0..1) interpolates between
no-op and full correction, useful for not over-correcting deliberately
warm/cool scenes.
"""

from __future__ import annotations

import numpy as np


def gray_world(
    rgb: np.ndarray,
    target_rg: float = 1.0,
    target_bg: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    if strength <= 0.0:
        return rgb
    flat = rgb.reshape(-1, 3)
    avg = flat.mean(axis=0)
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
