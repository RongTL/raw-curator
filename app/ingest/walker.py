"""Walk photos/incoming and yield every file whose kind `app.ingest.decode` supports."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from app.ingest.decode import all_supported_exts


def walk(root: Path, suffixes: Iterable[str] | None = None) -> Iterator[Path]:
    suffixes_lower = (
        {s.lower() for s in suffixes} if suffixes is not None else set(all_supported_exts())
    )
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes_lower:
            yield path
