"""Map an absolute source path to its path relative to a known photos root.

Used by submit / enhance / export to mirror the subfolder layout found under
photos/incoming/ into the library/, exported/, and jpeg/ output trees.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE_ROOTS: tuple[str, ...] = ("incoming", "library", "exported")


def relative_subpath(
    src: Path, photos: Path, roots: tuple[str, ...] = SOURCE_ROOTS
) -> Path:
    """Return ``src`` relative to whichever known root it lives under.

    The returned path includes the filename, e.g.::

        photos/incoming/2025/wedding/IMG.CR3 -> 2025/wedding/IMG.CR3

    Falls back to ``Path(src.name)`` (and logs a warning) when ``src`` is under
    none of the known roots, so callers always get a usable relative path.
    """
    for root in roots:
        base = photos / root
        try:
            return src.relative_to(base)
        except ValueError:
            continue
    log.warning(
        "source %s is under no known root %s; flattening to basename",
        src,
        roots,
    )
    return Path(src.name)
