"""Container-path -> browser-URL helpers.

The DB stores absolute container paths like ``/data/cache/previews/<hash>.jpg``.
FastAPI mounts that directory at ``/cache``, so the browser-visible URL is
``/cache/previews/<hash>.jpg``.
"""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.paths import relative_subpath

_CACHE_PREFIX = str(settings.cache).rstrip("/") + "/"


def cache_url(container_path: str | None) -> str | None:
    if not container_path:
        return None
    if container_path.startswith(_CACHE_PREFIX):
        return "/cache/" + container_path[len(_CACHE_PREFIX) :]
    return container_path


def jpeg_url(source_path: str | None) -> str | None:
    """Browser URL for the share JPEG ``export-jpeg`` writes for this photo.

    The path is *derived* — it mirrors the incoming subfolder layout into
    ``photos/jpeg/`` (which is mounted at ``/jpeg``) — so it is returned even
    when the file does not exist yet; the UI treats a 404 as "not enhanced yet".
    """
    if not source_path:
        return None
    rel = relative_subpath(Path(source_path), settings.photos).with_suffix(".jpg")
    return "/jpeg/" + str(rel)


def render_url(photo_hash: str, which: str) -> str:
    """Browser URL for a before/after review JPEG (may 404 until enhance runs).

    ``which`` is ``before`` or ``after``; enhance writes both under
    ``cache/enhanced/`` (mounted at ``/cache``), so the URL is derived and
    returned even before the file exists.
    """
    return f"/cache/enhanced/{photo_hash}.{which}.jpg"
