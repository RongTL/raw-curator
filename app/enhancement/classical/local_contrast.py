"""CLAHE local contrast (spec §2.4 / §7).

CLAHE on the L channel of LAB preserves color and avoids the global
histogram crush of plain equalisation. cv2's CLAHE is multi-threaded and
runs at full 24 MP in <600 ms on the Ryzen 3 3100.

`clip_limit` controls local contrast strength (1.5..3.0 typical). The
tile grid is the spatial granularity; smaller tiles create more local
contrast but risk halos around large smooth gradients.
"""

from __future__ import annotations

import numpy as np

from app.arrays import Array


def apply_clahe(
    rgb: Array,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> Array:
    if clip_limit <= 0.0:
        return rgb
    try:
        import cv2
    except ImportError:
        return rgb

    f = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
    lum = lab[..., 0]
    lum_u16 = np.clip(lum * 655.35, 0, 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    lum_eq = clahe.apply(lum_u16).astype(np.float32) / 655.35
    lab[..., 0] = lum_eq
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
