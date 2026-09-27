"""Single-file conversion: any supported source -> share-ready JPEG (+ EXIF)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from app.ingest.decode import FileKind, classify_kind, decode_preview, heic_available
from app.ingest.extensions import ALL_SUPPORTED_EXTS, HEIC_EXTS
from app.preview.jpeg_writer import resize_long_edge, write_jpeg

log = logging.getLogger(__name__)

_CHROMA_420 = 2  # Pillow subsampling code; the usual choice for delivery JPEGs


def _copy_exif(source: Path, dest: Path) -> None:
    """Copy EXIF from source -> dest JPEG, force Orientation=1 (image is pre-rotated).

    Soft-fail: a JPEG without EXIF is still a valid deliverable.
    """
    if shutil.which("exiftool") is None:
        log.debug("exiftool not on PATH; skipping EXIF copy for %s", dest.name)
        return
    try:
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                str(source),
                "-EXIF:all",
                "--Orientation",
                "-XMP:all",
                "-IPTC:all",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["exiftool", "-overwrite_original", "-Orientation=1", "-n", str(dest)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.warning("EXIF copy failed for %s: %s", dest.name, exc)


def convert_image_to_jpeg(
    src: Path, dest: Path, *, quality: int, long_edge: int, progressive: bool
) -> None:
    """Decode any supported source (same path as ingest previews), optionally
    cap the long edge, encode 4:2:0 JPEG, and copy EXIF across.

    A JPEG source with no resize requested is byte-copied instead of
    re-encoded, to avoid generation loss.
    """
    kind = classify_kind(src)
    if kind is None:
        raise ValueError(f"unsupported file for JPEG export: {src.suffix} ({src})")
    if kind == FileKind.JPEG and long_edge <= 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return
    arr = resize_long_edge(decode_preview(src), long_edge)
    write_jpeg(arr, dest, quality=quality, progressive=progressive, subsampling=_CHROMA_420)
    _copy_exif(src, dest)


def is_convertible(path: Path) -> bool:
    """True if the path's extension is one we know how to read into a JPEG."""
    ext = path.suffix.lower()
    if ext in HEIC_EXTS and not heic_available():
        return False
    return ext in ALL_SUPPORTED_EXTS
