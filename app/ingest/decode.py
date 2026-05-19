"""Format-aware decoder: maps a source file to an 8-bit sRGB ndarray.

The ingest pipeline used to be RAW-only via rawpy. This module dispatches
based on file extension so JPEG, TIFF, HEIC, and PNG can flow through the
same downstream stages (filter, score, cluster, decide).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageOps

log = logging.getLogger(__name__)


class FileKind(str, Enum):
    RAW = "raw"
    JPEG = "jpeg"
    TIFF = "tiff"
    HEIC = "heic"
    PNG = "png"


RAW_EXTS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srw", ".raf", ".orf",
     ".rw2", ".dng", ".pef", ".raw", ".x3f", ".rwl", ".3fr", ".iiq", ".mef",
     ".mos", ".mrw", ".sr2", ".srf"}
)
JPEG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg"})
TIFF_EXTS: frozenset[str] = frozenset({".tif", ".tiff"})
HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif"})
PNG_EXTS: frozenset[str] = frozenset({".png"})


_HEIC_OK = False
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIC_OK = True
except ImportError:
    log.debug("pillow-heif not installed; .heic files will be skipped")


def heic_available() -> bool:
    return _HEIC_OK


def classify_kind(path: Path) -> FileKind | None:
    """Return the FileKind for `path`, or None if the extension isn't supported."""
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return FileKind.RAW
    if ext in JPEG_EXTS:
        return FileKind.JPEG
    if ext in TIFF_EXTS:
        return FileKind.TIFF
    if ext in HEIC_EXTS:
        return FileKind.HEIC if _HEIC_OK else None
    if ext in PNG_EXTS:
        return FileKind.PNG
    return None


def all_supported_exts() -> frozenset[str]:
    out = RAW_EXTS | JPEG_EXTS | TIFF_EXTS | PNG_EXTS
    if _HEIC_OK:
        out = out | HEIC_EXTS
    return frozenset(out)


def _develop_raw(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            output_bps=8,
            half_size=False,
            no_auto_bright=False,
            use_camera_wb=True,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
        )


def _load_tiff(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _load_via_pillow(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def decode_preview(path: Path) -> np.ndarray:
    """Return an 8-bit RGB ndarray, EXIF-rotated, ready for thumb/preview pipelines."""
    kind = classify_kind(path)
    if kind is None:
        raise ValueError(f"unsupported file type: {path.suffix} ({path})")
    if kind == FileKind.RAW:
        return _develop_raw(path)
    if kind == FileKind.TIFF:
        return _load_tiff(path)
    # JPEG, HEIC, PNG — Pillow handles all three with EXIF orientation.
    return _load_via_pillow(path)


def extract_thumb_bytes(path: Path) -> bytes | None:
    """For RAWs, hand back the embedded JPEG thumb (cheap). Else return None
    so the caller falls back to the decoded preview array."""
    kind = classify_kind(path)
    if kind == FileKind.RAW:
        from app.preview.rawpy_dev import extract_embedded_thumb

        return extract_embedded_thumb(path)
    return None
