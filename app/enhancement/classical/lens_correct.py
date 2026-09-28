"""Per-frame lens distortion + chromatic-aberration correction via lensfun.

Reads the lens / camera / focal length / aperture from the frame's EXIF and
applies lensfun's combined geometry-distortion + transverse-CA correction with
``cv2.remap``. Works on float32 linear-Rec.2020 RGB in [0, 1] and returns the
same dtype and shape. Correction is best-effort: a frame whose lens lensfun
cannot resolve (its bundled DB plus the calibration XML under
``darktable/lensfun/``) passes through unchanged, and any lensfun failure is
logged and swallowed rather than aborting the frame.

Vignetting is deliberately not applied: the shipped full-frame profiles carry
distortion + CA only, and vignetting on a scene-referred linear master is better
left to the tone steps. This module replaces the darktable baseline sidecar,
which baked one frame's white balance into every develop.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.arrays import Array
from app.ingest.exif import ExifData

if TYPE_CHECKING:
    import lensfunpy

log = logging.getLogger(__name__)

# Calibration XML shipped with the repo (also copied into the system lensfun DB
# for the darktable GUI/CLI by the Containerfile). lensfunpy reads it directly so
# the same profiles serve both developers.
_XML_DIR = Path(__file__).resolve().parent.parent / "darktable" / "lensfun"


@lru_cache(maxsize=1)
def _database() -> lensfunpy.Database:
    """lensfunpy DB: the wheel's bundled data plus our extra calibration XML.

    Cached for the process — building it parses the whole database once.
    """
    import lensfunpy

    paths = sorted(str(p) for p in _XML_DIR.glob("*.xml"))
    return lensfunpy.Database(paths=paths or None)


def correct_lens(rgb: Array, exif: ExifData, *, enabled: bool = True) -> Array:
    """Distortion + CA correction for one frame; a no-op when it cannot apply.

    ``rgb`` is float32 RGB in [0, 1]; the return has the same shape and dtype.
    Returns ``rgb`` unchanged (the same object) when disabled, when EXIF lacks
    the lens or focal length, when no lensfun profile matches, or when lensfun is
    unavailable or errors — the frame is developed without lens correction rather
    than failing.
    """
    if not enabled:
        return rgb
    if not exif.lens or not exif.camera_body or not exif.focal_length:
        return rgb
    try:
        import lensfunpy

        db = _database()
        cams = db.find_cameras(exif.camera_make or "", exif.camera_body, loose_search=True)
        if not cams:
            return rgb
        cam = cams[0]
        lenses = db.find_lenses(cam, None, exif.lens, loose_search=True)
        if not lenses:
            return rgb
        lens = lenses[0]
        h, w = rgb.shape[:2]
        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(exif.focal_length, exif.aperture or 8.0, pixel_format=np.float32)
        # One combined map applies geometry distortion and transverse CA together,
        # so each channel is resampled exactly once (no compounded softening).
        coords = mod.apply_subpixel_geometry_distortion()
        if coords is None:
            return rgb
        # coords[..., c, 0] and [..., c, 1] are the per-channel x and y source maps.
        out = np.stack(
            [
                cv2.remap(rgb[..., c], coords[..., c, 0], coords[..., c, 1], cv2.INTER_LANCZOS4)
                for c in range(3)
            ],
            axis=-1,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception:
        log.exception("lens correction failed for lens=%r; leaving frame uncorrected", exif.lens)
        return rgb
