"""Walk photos/incoming and yield candidate RAW file paths."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

RAW_SUFFIXES: frozenset[str] = frozenset(
    {
        ".cr2",
        ".cr3",
        ".crw",
        ".nef",
        ".nrw",
        ".arw",
        ".srw",
        ".raf",
        ".orf",
        ".rw2",
        ".dng",
        ".pef",
        ".raw",
        ".x3f",
    }
)


def walk(root: Path, suffixes: Iterable[str] = RAW_SUFFIXES) -> Iterator[Path]:
    suffixes_lower = {s.lower() for s in suffixes}
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes_lower:
            yield path
