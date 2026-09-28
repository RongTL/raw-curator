"""Container-path -> browser-URL helpers.

The DB stores absolute container paths like ``/data/cache/previews/<hash>.jpg``.
FastAPI mounts that directory at ``/cache``, so the browser-visible URL is
``/cache/previews/<hash>.jpg``.
"""

from __future__ import annotations

from app.config import settings

_CACHE_PREFIX = str(settings.cache).rstrip("/") + "/"


def cache_url(container_path: str | None) -> str | None:
    if not container_path:
        return None
    if container_path.startswith(_CACHE_PREFIX):
        return "/cache/" + container_path[len(_CACHE_PREFIX) :]
    return container_path


def render_url(photo_hash: str, which: str) -> str:
    """Browser URL for a before/after review JPEG (may 404 until enhance runs).

    ``which`` is ``before`` or ``after``; enhance writes both under
    ``cache/enhanced/`` (mounted at ``/cache``), so the URL is derived and
    returned even before the file exists.
    """
    return f"/cache/enhanced/{photo_hash}.{which}.jpg"
