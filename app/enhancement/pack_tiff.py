"""Write the enhanced image as a 16-bit LZW RGB TIFF master, tagged and with the RAW's metadata."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile

from app.arrays import Array

log = logging.getLogger(__name__)
_TIFF_ICC_TAG = 34675


def write_tiff16(arr: Array, out: Path, *, icc: bytes | None = None) -> None:
    """``arr`` may be float in [0, 1], uint8, or uint16; ``icc`` is embedded verbatim when given."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype == np.uint16:
        data = arr
    elif arr.dtype == np.uint8:
        data = (arr.astype(np.uint32) * 257).astype(np.uint16)
    else:
        data = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    extratags = [(_TIFF_ICC_TAG, "B", len(icc), icc, True)] if icc else []
    tifffile.imwrite(out, data, photometric="rgb", compression="lzw", extratags=extratags)


def copy_metadata(source: Path, dest: Path) -> bool:
    """Copy EXIF/XMP/IPTC from ``source`` onto ``dest`` and force Orientation=1 (the
    pixels are already upright). Soft-fail: a master without EXIF is still usable."""
    if shutil.which("exiftool") is None:
        log.warning("exiftool not on PATH; %s written without EXIF", dest.name)
        return False
    try:
        subprocess.run(
            [
                "exiftool",
                "-overwrite_original",
                "-TagsFromFile",
                str(source),
                "-EXIF:all",
                "-XMP:all",
                "-IPTC:all",
                "--Orientation",
                "-Orientation#=1",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "metadata copy failed for %s: %s", dest.name, exc.stderr.decode(errors="replace")[:200]
        )
        return False
    return True
