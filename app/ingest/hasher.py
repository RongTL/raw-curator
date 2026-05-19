"""Streaming xxh3 of a file."""

from __future__ import annotations

from pathlib import Path

import xxhash

CHUNK = 1 << 20


def xxh3_file(path: Path) -> str:
    h = xxhash.xxh3_128()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
