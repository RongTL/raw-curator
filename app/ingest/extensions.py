"""Supported input file extensions, importable without heavy decode deps.

`app.ingest.decode` re-exports these for its own dispatch; lightweight callers
(e.g. `app.orchestrator.progress`) import from here to avoid pulling in
numpy/rawpy/tifffile/Pillow.
"""

from __future__ import annotations

RAW_EXTS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srw", ".raf", ".orf",
     ".rw2", ".dng", ".pef", ".raw", ".x3f", ".rwl", ".3fr", ".iiq", ".mef",
     ".mos", ".mrw", ".sr2", ".srf"}
)
JPEG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg"})
TIFF_EXTS: frozenset[str] = frozenset({".tif", ".tiff"})
HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif"})
PNG_EXTS: frozenset[str] = frozenset({".png"})

# Superset used for display-only counting; runtime ingest may exclude HEIC
# when pillow-heif is unavailable (see decode.all_supported_exts()).
ALL_SUPPORTED_EXTS: frozenset[str] = (
    RAW_EXTS | JPEG_EXTS | TIFF_EXTS | HEIC_EXTS | PNG_EXTS
)
