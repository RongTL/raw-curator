"""Write display-referred JPEGs from the enhancement engine's linear floats.

The before/after review needs viewable images the moment enhance finishes, and
the final export copies the full-res one. Four JPEGs per photo live under
``settings.enhanced_dir`` (cleaned by ``make reset``): review-res + full-res,
before (developed, un-enhanced) + after (AI-enhanced).
"""

from __future__ import annotations

from pathlib import Path

from app.arrays import Array
from app.config import settings
from app.enhancement.colorspace import linear_rec2020_to_srgb_u8
from app.enhancement.pack_tiff import copy_metadata
from app.preview.jpeg_writer import resize_long_edge, write_jpeg

_CHROMA_420 = 2

_SUFFIXES = {
    "before": "before.jpg",
    "after": "after.jpg",
    "before_full": "before.full.jpg",
    "after_full": "after.full.jpg",
}


def render_paths(photo_hash: str) -> dict[str, Path]:
    d = settings.enhanced_dir
    return {key: d / f"{photo_hash}.{suffix}" for key, suffix in _SUFFIXES.items()}


def write_render(
    linear_rgb: Array, dest: Path, *, long_edge: int, quality: int, source: str | None = None
) -> None:
    """linear Rec.2020 float32 [0,1] -> sRGB JPEG at ``dest`` (parents created).

    ``long_edge<=0`` keeps native size. ``source`` (a path) copies EXIF across
    with Orientation baked to 1 (the float image is already upright).
    """
    u8 = resize_long_edge(linear_rec2020_to_srgb_u8(linear_rgb), long_edge)
    write_jpeg(
        u8, dest, quality=quality, progressive=settings.jpeg_progressive, subsampling=_CHROMA_420
    )
    if source is not None:
        copy_metadata(Path(source), dest)
