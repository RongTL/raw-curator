"""Backlit recovery: edge-preserving shadow lift with highlight protection.

Sits between darktable develop and SCUNet. Auto-detects backlit images
(dense shadows + dense highlights, sparse mid-tones) and applies a
luminance-targeted lift so the subject becomes visible without crushing
the background's highlight detail.

Design constraints:
- Natural, not exaggerated. Default `shadow_lift=0.4` lifts a 20%-luma
  pixel by roughly 1/3 stop. Highlights above ~70% luma are protected.
- Edge-preserving: the lift mask is computed on a bilateral-filtered
  luminance so the gradient follows real edges instead of bleeding
  across them (no HDR halos).
- Idempotent on already-balanced images: the histogram gate skips the
  stage when there is no backlit signature.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722

_SHADOW_BIN_MAX = 0.15
_HIGHLIGHT_BIN_MIN = 0.80
_MIDTONE_BIN_MIN = 0.35
_MIDTONE_BIN_MAX = 0.65
_SHADOW_DENSITY_TRIGGER = 0.18
_HIGHLIGHT_DENSITY_TRIGGER = 0.10
_MIDTONE_DEFICIT_MAX = 0.25  # "bimodal" gate: backlit scenes have hollow midtones

_LIFT_MASK_CENTRE = 0.22
_LIFT_MASK_SIGMA = 0.18

_HIGHLIGHT_PROTECT_KNEE = 0.65
_HIGHLIGHT_PROTECT_RANGE = 0.30

# Mask-computation working size: bilateral on full 24 MP is ~20 s. We run
# it on a downscaled luma and bilinear-resize the mask back; this is the
# standard guided-upsampling trick for tone-mapping masks.
_MASK_WORK_MAX_EDGE = 768


def _rgb_to_luma(rgb_u8: np.ndarray) -> np.ndarray:
    f = rgb_u8.astype(np.float32) / 255.0
    return _LUMA_R * f[..., 0] + _LUMA_G * f[..., 1] + _LUMA_B * f[..., 2]


def is_backlit(luma01: np.ndarray) -> bool:
    """Histogram-based backlit detector.

    Triggers only when the histogram is bimodal: dense deep shadows AND
    dense bright highlights AND a hollow midtone band. The midtone-deficit
    gate prevents false positives on high-contrast-but-balanced scenes
    (midday landscape with bright sky + shadowed ground + plenty of midtone
    foliage).
    """
    bins = 20
    hist, _ = np.histogram(luma01, bins=bins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return False
    shadow_n = int(round(_SHADOW_BIN_MAX * bins))
    highlight_n = int(round((1.0 - _HIGHLIGHT_BIN_MIN) * bins))
    mid_lo = int(round(_MIDTONE_BIN_MIN * bins))
    mid_hi = int(round(_MIDTONE_BIN_MAX * bins))
    shadow_density = hist[:shadow_n].sum() / total
    highlight_density = hist[-highlight_n:].sum() / total
    midtone_density = hist[mid_lo:mid_hi].sum() / total
    return (
        shadow_density >= _SHADOW_DENSITY_TRIGGER
        and highlight_density >= _HIGHLIGHT_DENSITY_TRIGGER
        and midtone_density <= _MIDTONE_DEFICIT_MAX
    )


def _edge_preserving_luma(luma01: np.ndarray) -> np.ndarray | None:
    """Bilateral-filtered luminance, computed on a downscaled copy for speed.

    Returns None when cv2 is unavailable — callers must treat this as
    "recovery not possible" and return the input unchanged rather than
    fall back to a non-edge-preserving mask (which would produce halos).

    The mask is computed at <=768px on the long edge then bilinear-resized
    back to full resolution. This is the standard guided-upsampling trick
    for tone-mapping masks: shape is preserved by the bilateral, the
    high-frequency detail in the lift comes from the multiplication with
    the original luma at full resolution.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        log.warning(
            "cv2 unavailable (%s) — backlit recovery requires opencv-python; skipping", exc
        )
        return None

    h, w = luma01.shape
    as_u8 = (np.clip(luma01, 0.0, 1.0) * 255.0).astype(np.uint8)

    long_edge = max(h, w)
    if long_edge > _MASK_WORK_MAX_EDGE:
        scale_factor = _MASK_WORK_MAX_EDGE / long_edge
        small_h = max(1, int(round(h * scale_factor)))
        small_w = max(1, int(round(w * scale_factor)))
        small = cv2.resize(as_u8, (small_w, small_h), interpolation=cv2.INTER_AREA)
        smoothed_small = cv2.bilateralFilter(small, d=11, sigmaColor=28, sigmaSpace=48)
        smoothed = cv2.resize(smoothed_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        diameter = max(9, min(15, int(round(long_edge / 80.0)) | 1))
        smoothed = cv2.bilateralFilter(as_u8, d=diameter, sigmaColor=28, sigmaSpace=48)
    return smoothed.astype(np.float32) / 255.0


def _lift_mask(local_luma: np.ndarray) -> np.ndarray:
    return np.exp(-((local_luma - _LIFT_MASK_CENTRE) ** 2) / (2.0 * _LIFT_MASK_SIGMA**2))


def _highlight_protect(local_luma: np.ndarray, strength: float) -> np.ndarray:
    ramp = np.clip(
        (local_luma - _HIGHLIGHT_PROTECT_KNEE) / _HIGHLIGHT_PROTECT_RANGE, 0.0, 1.0
    )
    return 1.0 - ramp * np.clip(strength, 0.0, 1.0)


def recover_backlit(
    rgb: np.ndarray,
    shadow_lift: float = 0.4,
    highlight_protect: float = 0.15,
    force: bool = False,
) -> np.ndarray:
    """Apply backlit recovery if the image looks backlit (or `force=True`).

    Returns the input unchanged when not backlit and not forced, or when
    shadow_lift <= 0. Raises ValueError on unsupported array shapes.
    """
    if shadow_lift <= 0.0:
        return rgb
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"recover_backlit expects HxWx3 uint8 RGB, got {rgb.dtype} {rgb.shape}"
        )

    luma = _rgb_to_luma(rgb)
    if not force and not is_backlit(luma):
        return rgb

    local_luma = _edge_preserving_luma(luma)
    if local_luma is None:
        return rgb  # cv2 missing — skip cleanly rather than degrade
    mask = _lift_mask(local_luma) * _highlight_protect(local_luma, highlight_protect)

    # Lift in gamma space: y = x^(1/(1+k)). k=0 → identity; k>0 brightens midtones
    # more than blacks/highlights, which reads as natural to the eye.
    f = rgb.astype(np.float32) / 255.0
    k = shadow_lift * mask[..., None]
    lifted = np.power(np.clip(f, 1e-6, 1.0), 1.0 / (1.0 + k))

    return (np.clip(lifted, 0.0, 1.0) * 255.0).astype(np.uint8)
