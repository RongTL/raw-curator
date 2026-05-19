"""Single-file conversion helpers: any supported source -> JPEG."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageOps

from app.ingest.decode import (
    HEIC_EXTS,
    JPEG_EXTS,
    PNG_EXTS,
    RAW_EXTS,
    TIFF_EXTS,
    FileKind,
    classify_kind,
    heic_available,
)

log = logging.getLogger(__name__)


RAW_EXTENSIONS = RAW_EXTS
TIFF_EXTENSIONS = TIFF_EXTS
JPEG_EXTENSIONS = JPEG_EXTS
HEIC_EXTENSIONS = HEIC_EXTS
PNG_EXTENSIONS = PNG_EXTS


def _resize_long_edge(arr: np.ndarray, long_edge: int) -> np.ndarray:
    if long_edge <= 0:
        return arr
    h, w = arr.shape[:2]
    cur = max(h, w)
    if cur <= long_edge:
        return arr
    scale = long_edge / cur
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _develop_raw(raw_path: Path) -> np.ndarray:
    with rawpy.imread(str(raw_path)) as raw:
        return raw.postprocess(
            output_bps=8,
            half_size=False,
            no_auto_bright=False,
            use_camera_wb=True,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
        )


def _load_tiff_as_rgb8(tiff_path: Path) -> np.ndarray:
    arr = tifffile.imread(str(tiff_path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_jpeg(arr: np.ndarray, dest: Path, quality: int, progressive: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(
        dest,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=progressive,
        subsampling=2,  # 4:2:0
    )


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
                "-TagsFromFile", str(source),
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


def _load_via_pillow(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def convert_raw_to_jpeg(
    raw: Path, dest: Path, *, quality: int, long_edge: int, progressive: bool
) -> None:
    arr = _develop_raw(raw)
    arr = _resize_long_edge(arr, long_edge)
    _save_jpeg(arr, dest, quality=quality, progressive=progressive)
    _copy_exif(raw, dest)


def convert_tiff_to_jpeg(
    tiff: Path, dest: Path, *, quality: int, long_edge: int, progressive: bool
) -> None:
    arr = _load_tiff_as_rgb8(tiff)
    arr = _resize_long_edge(arr, long_edge)
    _save_jpeg(arr, dest, quality=quality, progressive=progressive)
    _copy_exif(tiff, dest)


def convert_image_to_jpeg(
    src: Path, dest: Path, *, quality: int, long_edge: int, progressive: bool
) -> None:
    """Dispatcher: RAW/TIFF use specialised paths; JPEG/HEIC/PNG go through Pillow.

    If the source is already a JPEG and no resize is requested, we copy bytes
    instead of re-encoding (avoids generation loss).
    """
    kind = classify_kind(src)
    if kind == FileKind.RAW:
        convert_raw_to_jpeg(
            src, dest, quality=quality, long_edge=long_edge, progressive=progressive
        )
        return
    if kind == FileKind.TIFF:
        convert_tiff_to_jpeg(
            src, dest, quality=quality, long_edge=long_edge, progressive=progressive
        )
        return
    if kind == FileKind.JPEG and long_edge <= 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return
    if kind in {FileKind.JPEG, FileKind.HEIC, FileKind.PNG}:
        arr = _load_via_pillow(src)
        arr = _resize_long_edge(arr, long_edge)
        _save_jpeg(arr, dest, quality=quality, progressive=progressive)
        _copy_exif(src, dest)
        return
    raise ValueError(f"unsupported file for JPEG export: {src.suffix} ({src})")


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def is_tiff(path: Path) -> bool:
    return path.suffix.lower() in TIFF_EXTENSIONS


def is_convertible(path: Path) -> bool:
    """True if the path's extension is one we know how to read into a JPEG."""
    ext = path.suffix.lower()
    if ext in HEIC_EXTENSIONS and not heic_available():
        return False
    return ext in (RAW_EXTENSIONS | TIFF_EXTENSIONS | JPEG_EXTENSIONS | HEIC_EXTENSIONS | PNG_EXTENSIONS)
