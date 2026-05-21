"""Final tone mapping (spec §7 step 8).

A gentle filmic curve with a shoulder for highlights and a toe for
shadows. Designed to be a no-op-like polish that prevents output
clipping after the upstream steps have done their work — not a heavy
HDR look. Applied per channel so it preserves perceived colour at the
high end (where filmic curves typically desaturate slightly, which is
the desired film-like behaviour).

`global_compress` is the §2.4 "excessive dynamic range" remedy and is
applied earlier in the chain on high-DR inputs.
"""

from __future__ import annotations

import numpy as np


def filmic_tone_map(
    rgb: np.ndarray,
    shoulder: float = 0.85,
    toe: float = 0.04,
) -> np.ndarray:
    s = float(np.clip(shoulder, 0.5, 0.99))
    t = float(np.clip(toe, 0.0, 0.2))
    f = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    over = np.maximum(0.0, f - s)
    range_high = max(1e-6, 1.0 - s)
    shoulder_term = over / (1.0 + over / range_high)
    f_shouldered = np.where(f > s, s + shoulder_term, f)

    if t > 0.0:
        toe_lift = t * np.exp(-(f_shouldered / 0.08) ** 2)
        f_shouldered = f_shouldered + toe_lift

    return np.clip(f_shouldered, 0.0, 1.0).astype(np.float32)


def global_compress(rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
    if strength <= 0.0:
        return rgb
    s = float(np.clip(strength, 0.0, 1.0))
    f = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    compressed = f / (1.0 + s * f * (1.0 - f))
    lo = compressed.min()
    hi = compressed.max()
    if hi - lo > 1e-6:
        compressed = (compressed - lo) / (hi - lo)
    return np.clip(compressed, 0.0, 1.0).astype(np.float32)
