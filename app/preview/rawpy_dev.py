"""Embedded-JPEG thumbnail extraction via rawpy."""

from __future__ import annotations

from pathlib import Path

import rawpy


def extract_embedded_thumb(raw_path: Path) -> bytes | None:
    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None
    return thumb.data
