"""Quality measurement per spec §1-§5.

Input contract: linear Rec.2020 float32 in [0, 1] (what develop_full produces).
A 24 MP image at float32 RGB is ~290 MB — comfortable on a
24 GB host. cv2 is used for the heavy-lifting kernels (Laplacian, Canny,
color conversion) — they are SIMD-vectorised and release the GIL, so they
pipeline well alongside Python work on a 4C/8T Ryzen 3 3100.

Metric coordinate systems:
- Exposure / DR / sharpness / noise metrics use an sRGB-encoded 8-bit
  luma so the absolute thresholds quoted in the spec tables apply
  directly (e.g. Laplacian variance "< 50 blurry / > 150 sharp").
- Color uses the original linear RGB — gray-world is well-defined there.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from app.arrays import Array
from app.enhancement.colorspace import luma, rgb_to_ycbcr

log = logging.getLogger(__name__)

# Spec thresholds for the midtone-deviation score (§1.3 general band).
_MIDTONE_BAND_CENTER = 0.625  # mid of 0.50..0.75
_MIDTONE_BAND_HALF = 0.125

# Sharpness / noise patch sizing.
_FLAT_PATCH = 16  # for noise σ over flat regions
_NOISE_FLAT_GRADIENT_MAX = 4.0  # 8-bit luma gradient ceiling for a "flat" patch
_FFT_CENTER_FRAC = 0.25  # the band we treat as DC/low-frequency


def as_linear_float01(img: Array) -> Array:
    """Normalize uint8 / uint16 / float to float32 [0, 1].

    Values are taken as linear light; the sRGB encoding for the spec's 8-bit
    thresholds happens in _luma_u8.
    """
    if img.dtype == np.float32:
        return np.clip(img, 0.0, 1.0)
    if img.dtype == np.float64:
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    if img.dtype == np.uint16:
        return img.astype(np.float32) * (1.0 / 65535.0)
    raise TypeError(f"unsupported dtype {img.dtype} for quality measurement")


def linear_to_srgb_u8(rgb_f01: Array) -> Array:
    """Encode linear-light float to sRGB-gamma 8-bit (matches what a viewer shows)."""
    a = 0.055
    f = np.clip(rgb_f01, 0.0, 1.0)
    low = f <= 0.0031308
    out = np.where(low, 12.92 * f, (1.0 + a) * np.power(np.maximum(f, 1e-6), 1.0 / 2.4) - a)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _luma_u8(rgb_f01: Array) -> Array:
    lum = luma(rgb_f01)
    # sRGB-encode the single-channel luma directly.
    a = 0.055
    f = np.clip(lum, 0.0, 1.0)
    enc = np.where(
        f <= 0.0031308, 12.92 * f, (1.0 + a) * np.power(np.maximum(f, 1e-6), 1.0 / 2.4) - a
    )
    return (enc * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# §1 Exposure
# ---------------------------------------------------------------------------


def exposure_metrics(rgb_f01: Array) -> dict[str, float]:
    lum = _luma_u8(rgb_f01)
    hist = np.bincount(lum.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return {
            "mean_luma": 0.0,
            "shadow_clip": 0.0,
            "highlight_clip": 0.0,
            "midtone_ratio": 0.0,
            "midtone_deviation": 1.0,
        }
    shadow_clip = float(hist[:6].sum() / total)
    highlight_clip = float(hist[250:].sum() / total)
    midtone_ratio = float(hist[64:193].sum() / total)
    if (
        _MIDTONE_BAND_CENTER - _MIDTONE_BAND_HALF
        <= midtone_ratio
        <= _MIDTONE_BAND_CENTER + _MIDTONE_BAND_HALF
    ):
        midtone_deviation = 0.0
    else:
        midtone_deviation = float(
            min(1.0, abs(midtone_ratio - _MIDTONE_BAND_CENTER) / _MIDTONE_BAND_HALF)
        )
    return {
        "mean_luma": float(lum.mean()),
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip,
        "midtone_ratio": midtone_ratio,
        "midtone_deviation": midtone_deviation,
    }


# ---------------------------------------------------------------------------
# §2 Dynamic Range
# ---------------------------------------------------------------------------


def dynamic_range_metrics(rgb_f01: Array) -> dict[str, float]:
    lum = _luma_u8(rgb_f01).astype(np.float32)
    p5, p95 = np.percentile(lum, [5, 95])
    dr = float(p95 - p5)
    height, width = lum.shape
    bh = height // 16
    bw = width // 16
    if bh == 0 or bw == 0:
        return {"dr_p95_p5": dr, "local_dr_mean": dr}
    crop = lum[: bh * 16, : bw * 16].reshape(bh, 16, bw, 16)
    block_max = crop.max(axis=(1, 3))
    block_min = crop.min(axis=(1, 3))
    return {"dr_p95_p5": dr, "local_dr_mean": float((block_max - block_min).mean())}


# ---------------------------------------------------------------------------
# §3 Color
# ---------------------------------------------------------------------------


def color_metrics(
    rgb_f01: Array,
    face_boxes: Sequence[tuple[int, int, int, int]] | None = None,
) -> dict[str, float | None]:
    rgb = rgb_f01.reshape(-1, 3)
    avg = rgb.mean(axis=0)
    g = max(float(avg[1]), 1e-6)
    rg_ratio = float(avg[0] / g)
    bg_ratio = float(avg[2] / g)

    mx = rgb_f01.max(axis=-1)
    mn = rgb_f01.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).astype(np.float32)
    avg_sat = float(sat.mean())
    oversat = float((sat > 0.85).mean())

    skin_hue_var: float | None = None
    if face_boxes:
        height, width, _ = rgb_f01.shape
        cb_chunks: list[Array] = []
        cr_chunks: list[Array] = []
        for x, y, w, h in face_boxes:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(width, int(x + w))
            y1 = min(height, int(y + h))
            if x1 <= x0 or y1 <= y0:
                continue
            ycbcr = rgb_to_ycbcr(rgb_f01[y0:y1, x0:x1])
            cb_chunks.append(ycbcr[..., 1].ravel())
            cr_chunks.append(ycbcr[..., 2].ravel())
        if cb_chunks:
            cb = np.concatenate(cb_chunks)
            cr = np.concatenate(cr_chunks)
            hue = np.arctan2(cr - 0.5, cb - 0.5)
            skin_hue_var = float(np.var(hue))

    return {
        "rg_ratio": rg_ratio,
        "bg_ratio": bg_ratio,
        "avg_saturation": avg_sat,
        "oversat_ratio": oversat,
        "skin_hue_var": skin_hue_var,
    }


# ---------------------------------------------------------------------------
# §4 Sharpness
# ---------------------------------------------------------------------------


def sharpness_metrics(rgb_f01: Array) -> dict[str, float]:
    lum = _luma_u8(rgb_f01)
    try:
        import cv2

        lap = cv2.Laplacian(lum, ddepth=cv2.CV_32F, ksize=3)
        lap_var = float(lap.var())
        edges = cv2.Canny(lum, 80, 160)
        edge_density = float((edges > 0).mean())
    except ImportError:
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = _conv2d_same(lum.astype(np.float32), kernel)
        lap_var = float(lap.var())
        edge_density = float((np.abs(lap) > 24).mean())

    # FFT high-frequency energy on a downscaled luma (24 MP FFT is wasteful).
    height, width = lum.shape
    max_edge = 1024
    long = max(height, width)
    if long > max_edge:
        step = int(np.ceil(long / max_edge))
        small = lum[::step, ::step]
    else:
        small = lum
    f = np.fft.fftshift(np.fft.fft2(small.astype(np.float32)))
    mag = np.abs(f)
    sh, sw = mag.shape
    cy, cx = sh // 2, sw // 2
    rh = int(sh * _FFT_CENTER_FRAC / 2)
    rw = int(sw * _FFT_CENTER_FRAC / 2)
    mask = np.ones_like(mag, dtype=bool)
    mask[cy - rh : cy + rh + 1, cx - rw : cx + rw + 1] = False
    hf_energy = float(mag[mask].mean()) if mask.any() else 0.0

    return {
        "lap_var": lap_var,
        "edge_density": edge_density,
        "hf_energy": hf_energy,
    }


def _conv2d_same(img: Array, kernel: Array) -> Array:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + img.shape[0], j : j + img.shape[1]]
    return out


# ---------------------------------------------------------------------------
# §5 Noise
# ---------------------------------------------------------------------------


def noise_metrics(rgb_f01: Array) -> dict[str, float]:
    lum = _luma_u8(rgb_f01)
    luma_noise = _flat_patch_std(lum)

    try:
        import cv2

        lab = cv2.cvtColor(np.clip(rgb_f01, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
        a = lab[..., 1]
        b = lab[..., 2]
        chroma = 0.5 * (_flat_patch_std(a, lum) + _flat_patch_std(b, lum))
    except ImportError:
        chroma = 0.0

    return {"luma_noise": float(luma_noise), "chroma_noise": float(chroma)}


def _flat_patch_std(image: Array, luma_for_flat: Array | None = None) -> float:
    """Estimate noise σ by averaging std over patches with near-zero gradient."""
    img = image.astype(np.float32)
    lum = (luma_for_flat if luma_for_flat is not None else image).astype(np.float32)
    height, width = lum.shape
    bh = height // _FLAT_PATCH
    bw = width // _FLAT_PATCH
    if bh == 0 or bw == 0:
        return 0.0
    lum_blocks = lum[: bh * _FLAT_PATCH, : bw * _FLAT_PATCH].reshape(
        bh, _FLAT_PATCH, bw, _FLAT_PATCH
    )
    gx = np.diff(lum_blocks, axis=3)
    gy = np.diff(lum_blocks, axis=1)
    grad_mag = np.sqrt(np.maximum(0.0, (gx**2).mean(axis=(1, 3)) + (gy**2).mean(axis=(1, 3))))
    flat_mask = grad_mag < _NOISE_FLAT_GRADIENT_MAX
    if not flat_mask.any():
        thr = np.percentile(grad_mag, 1)
        flat_mask = grad_mag <= thr
    img_blocks = img[: bh * _FLAT_PATCH, : bw * _FLAT_PATCH].reshape(
        bh, _FLAT_PATCH, bw, _FLAT_PATCH
    )
    patch_std = img_blocks.std(axis=(1, 3))
    return float(patch_std[flat_mask].mean()) if flat_mask.any() else 0.0


# ---------------------------------------------------------------------------
# Unified entry point — returns the raw metric dict (no scoring yet).
# ---------------------------------------------------------------------------


def measure_all(
    rgb: Array,
    face_boxes: Sequence[tuple[int, int, int, int]] | None = None,
) -> dict[str, float | None]:
    rgb_f01 = as_linear_float01(rgb)
    if rgb_f01.ndim != 3 or rgb_f01.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {rgb_f01.shape}")
    out: dict[str, float | None] = {}
    out.update(exposure_metrics(rgb_f01))
    out.update(dynamic_range_metrics(rgb_f01))
    out.update(color_metrics(rgb_f01, face_boxes=face_boxes))
    out.update(sharpness_metrics(rgb_f01))
    out.update(noise_metrics(rgb_f01))
    return out
