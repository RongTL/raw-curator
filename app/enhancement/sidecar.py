"""Which darktable sidecar develops a given RAW.

Order: the user's sidecar in darktable's own naming (``IMG.CR3.xmp``) mirrored under
``xmp/<subfolder>/``, then the legacy ``xmp/<stem>.xmp`` lookup, then the shipped
baseline (Step 2), then none (darktable defaults).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.paths import relative_subpath

log = logging.getLogger(__name__)
_warned_legacy = False

BASELINE_XMP = Path(__file__).resolve().parent / "darktable" / "raw-curator-base.xmp"


def resolve_xmp(
    source: Path,
    *,
    photos_root: Path,
    xmp_root: Path,
    baseline: Path | None = None,
) -> Path | None:
    global _warned_legacy
    rel = relative_subpath(source, photos_root)
    mirrored = xmp_root / rel.parent / f"{source.name}.xmp"
    if mirrored.exists():
        return mirrored
    legacy = xmp_root / f"{source.stem}.xmp"
    if legacy.exists():
        if not _warned_legacy:
            log.warning(
                "using legacy sidecar name %s; prefer %s",
                legacy.name,
                mirrored.relative_to(xmp_root),
            )
            _warned_legacy = True
        return legacy
    if baseline is not None and baseline.exists():
        return baseline
    return None
