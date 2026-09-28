"""Engine-side lens distortion + chromatic-aberration correction via lensfun.

The correction is per-frame (read from EXIF), deterministic, and best-effort: a
frame whose lens lensfun cannot resolve passes through untouched. It replaces the
old darktable baseline sidecar, which baked one frame's white balance into every
develop.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.enhancement.classical.lens_correct import correct_lens
from app.ingest.exif import ExifData


def _synthetic(h: int = 400, w: int = 600) -> np.ndarray:
    """Horizontal gradient with a bright 3px border, float32 RGB in [0, 1].

    The border makes a geometric remap visible: distortion correction moves the
    edge pixels inward.
    """
    img = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))[..., None].repeat(3, axis=2)
    img[:3, :] = 1.0
    img[-3:, :] = 1.0
    img[:, :3] = 1.0
    img[:, -3:] = 1.0
    return img


RF24 = ExifData(
    camera_make="Canon",
    camera_body="Canon EOS R8",
    lens="RF24mm F1.8 MACRO IS STM",
    focal_length=24.0,
    aperture=8.0,
)


def test_disabled_is_noop() -> None:
    img = _synthetic()
    out = correct_lens(img, RF24, enabled=False)
    assert out is img


def test_missing_lens_is_noop() -> None:
    img = _synthetic()
    exif = ExifData(camera_make="Canon", camera_body="Canon EOS R8", focal_length=24.0)
    out = correct_lens(img, exif, enabled=True)
    assert out is img


def test_missing_focal_length_is_noop() -> None:
    img = _synthetic()
    exif = ExifData(
        camera_make="Canon", camera_body="Canon EOS R8", lens="RF24mm F1.8 MACRO IS STM"
    )
    out = correct_lens(img, exif, enabled=True)
    assert out is img


def test_unknown_lens_is_noop() -> None:
    pytest.importorskip("lensfunpy")  # otherwise this passes for the wrong reason (no lensfunpy)
    img = _synthetic()
    exif = ExifData(
        camera_make="Nikon",
        camera_body="Nikon Z9",
        lens="NONEXISTENT 9000mm f/0.1",
        focal_length=24.0,
    )
    out = correct_lens(img, exif, enabled=True)
    assert out is img


def test_known_lens_applies_correction() -> None:
    pytest.importorskip("lensfunpy")
    img = _synthetic()
    out = correct_lens(img, RF24, enabled=True)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert not np.array_equal(out, img)
