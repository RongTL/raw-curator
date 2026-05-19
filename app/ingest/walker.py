"""Walk photos/incoming and yield candidate image file paths.

Originally RAW-only; now accepts every kind supported by `app.ingest.decode`.
The `RAW_SUFFIXES` constant is retained for callers that want a RAW-only filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from app.ingest.decode import RAW_EXTS, all_supported_exts

RAW_SUFFIXES: frozenset[str] = RAW_EXTS


def walk(root: Path, suffixes: Iterable[str] | None = None) -> Iterator[Path]:
    suffixes_lower = (
        {s.lower() for s in suffixes} if suffixes is not None else set(all_supported_exts())
    )
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes_lower:
            yield path
