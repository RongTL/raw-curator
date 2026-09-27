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

from app.arrays import Array
from app.enhancement.colorspace import linear_rec2020_to_srgb_u8

# Extension sets live in the constants-only `app.ingest.extensions` module so
# lightweight callers (progress polling, export) can use them without this
# module's heavy decode deps.
from app.ingest.extensions import HEIC_EXTS, JPEG_EXTS, PNG_EXTS, RAW_EXTS, TIFF_EXTS

log = logging.getLogger(__name__)


class FileKind(str, Enum):
    RAW = "raw"
    JPEG = "jpeg"
    TIFF = "tiff"
    HEIC = "heic"
    PNG = "png"


_HEIC_OK = False
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIC_OK = True
except ImportError:
    log.debug("pillow-heif not installed; .heic files will be skipped")


def heic_available() -> bool:
    return _HEIC_OK


# Magic-byte signatures for content sniffing. An extension can lie — e.g. a
# JPEG exported by Picasa/Google Photos saved with a .CR2 suffix — so for the
# unambiguous display formats we trust the bytes over the extension.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_HEIC_BRANDS: frozenset[bytes] = frozenset(
    {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"heif", b"mif1", b"msf1"}
)


def _kind_from_ext(ext: str) -> FileKind | None:
    if ext in RAW_EXTS:
        return FileKind.RAW
    if ext in JPEG_EXTS:
        return FileKind.JPEG
    if ext in TIFF_EXTS:
        return FileKind.TIFF
    if ext in HEIC_EXTS:
        return FileKind.HEIC
    if ext in PNG_EXTS:
        return FileKind.PNG
    return None


def _sniff_kind(path: Path) -> FileKind | None:
    """Detect format from magic bytes. Returns None when the signature is
    ambiguous — TIFF-family magic covers real TIFF *and* CR2/NEF/ARW/DNG raws —
    or the file can't be read, so the caller falls back to the extension."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head.startswith(_JPEG_MAGIC):
        return FileKind.JPEG
    if head.startswith(_PNG_MAGIC):
        return FileKind.PNG
    if len(head) >= 12 and head[4:8] == b"ftyp" and head[8:12] in _HEIC_BRANDS:
        return FileKind.HEIC
    return None


def classify_kind(path: Path) -> FileKind | None:
    """Return the FileKind for `path`, or None if it isn't supported.

    Content (magic bytes) wins over the extension when they disagree, so a
    misnamed file (e.g. a JPEG saved as .CR2) routes to the right decoder
    instead of crashing the RAW path in LibRaw.
    """
    ext_kind = _kind_from_ext(path.suffix.lower())
    sniffed = _sniff_kind(path)
    kind = sniffed if sniffed is not None else ext_kind
    if sniffed is not None and ext_kind is not None and sniffed != ext_kind:
        log.warning(
            "%s has extension %r but its content is %s; using content.",
            path,
            path.suffix,
            sniffed.value,
        )
    if kind == FileKind.HEIC and not _HEIC_OK:
        return None
    return kind


def all_supported_exts() -> frozenset[str]:
    out = RAW_EXTS | JPEG_EXTS | TIFF_EXTS | PNG_EXTS
    if _HEIC_OK:
        out = out | HEIC_EXTS
    return frozenset(out)


def develop_raw_rgb8(path: Path) -> Array:
    """Full-size 8-bit sRGB development via LibRaw (camera WB, auto-bright)."""
    with rawpy.imread(str(path)) as raw:
        return np.asarray(
            raw.postprocess(
                output_bps=8,
                half_size=False,
                no_auto_bright=False,
                use_camera_wb=True,
                gamma=(2.222, 4.5),
                output_color=rawpy.ColorSpace.sRGB,
            )
        )


def extract_embedded_thumb(path: Path) -> bytes | None:
    """The camera's embedded JPEG preview, or None if absent / not JPEG."""
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None
    return bytes(thumb.data)


LINEAR_REC2020_DESC = "Linear Rec2020 RGB"
_TIFF_ICC_TAG = 34675


def icc_description(icc: bytes) -> str | None:
    """The ICC 'desc' tag as text (v2 'desc' or v4 'mluc' record); None if absent/malformed."""
    if len(icc) < 132:
        return None
    count = int.from_bytes(icc[128:132], "big")
    count = min(count, max(0, (len(icc) - 132) // 12))  # a malformed count can't exceed the blob
    for i in range(count):
        off = 132 + 12 * i
        if icc[off : off + 4] != b"desc":
            continue
        start = int.from_bytes(icc[off + 4 : off + 8], "big")
        data = icc[start : start + int.from_bytes(icc[off + 8 : off + 12], "big")]
        if data[:4] == b"desc":
            n = int.from_bytes(data[8:12], "big")
            return data[12 : 12 + n].rstrip(b"\x00").decode("ascii", "replace")
        if data[:4] == b"mluc" and int.from_bytes(data[8:12], "big") > 0:
            n = int.from_bytes(data[20:24], "big")
            o = int.from_bytes(data[24:28], "big")
            return data[o : o + n].decode("utf-16-be", "replace").rstrip("\x00")
    return None


def _tiff_icc(path: Path) -> bytes | None:
    with tifffile.TiffFile(str(path)) as tf:
        tag = tf.pages[0].tags.get(_TIFF_ICC_TAG)  # type: ignore[union-attr]
        return bytes(tag.value) if tag is not None else None


def load_tiff_rgb8(path: Path) -> Array:
    """Any TIFF -> HxWx3 uint8 display sRGB. Our linear Rec.2020 masters are colour-converted;
    everything else is assumed sRGB-encoded (grayscale broadcast, alpha dropped, 16-bit >> 8)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    icc = _tiff_icc(path)
    if icc is not None and icc_description(icc) == LINEAR_REC2020_DESC:
        scale = 65535.0 if arr.dtype == np.uint16 else 255.0 if arr.dtype == np.uint8 else 1.0
        return linear_rec2020_to_srgb_u8(arr.astype(np.float32) / scale)
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_pillow_rgb8(path: Path) -> Array:
    """JPEG / HEIC / PNG -> HxWx3 uint8, EXIF orientation applied."""
    with Image.open(path) as opened:
        img: Image.Image = ImageOps.exif_transpose(opened) or opened
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)


def decode_preview(path: Path) -> Array:
    """Return an 8-bit RGB ndarray, EXIF-rotated, ready for thumb/preview pipelines."""
    kind = classify_kind(path)
    if kind is None:
        raise ValueError(f"unsupported file type: {path.suffix} ({path})")
    if kind == FileKind.RAW:
        return develop_raw_rgb8(path)
    if kind == FileKind.TIFF:
        return load_tiff_rgb8(path)
    # JPEG, HEIC, PNG — Pillow handles all three with EXIF orientation.
    return load_pillow_rgb8(path)


def extract_thumb_bytes(path: Path) -> bytes | None:
    """For RAWs, hand back the embedded JPEG thumb (cheap). Else return None
    so the caller falls back to the decoded preview array."""
    if classify_kind(path) == FileKind.RAW:
        return extract_embedded_thumb(path)
    return None
