"""RAW -> RGB numpy array via rawpy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy


def develop_preview(raw_path: Path) -> np.ndarray:
    with rawpy.imread(str(raw_path)) as raw:
        return raw.postprocess(
            output_bps=8,
            half_size=True,
            no_auto_bright=False,
            use_camera_wb=True,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
        )


def extract_embedded_thumb(raw_path: Path) -> bytes | None:
    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None
    return thumb.data
