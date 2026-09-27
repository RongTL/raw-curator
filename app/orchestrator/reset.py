"""Session reset: wipe every per-batch artefact and re-create the schema.

The defining operation of the ephemeral model. Reached via `raw-curator reset`
(which `make reset` calls) and via the UI's "New batch" action.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from app.config import settings

RESET_CONFIRM_TOKEN = "RESET"

_REPO_ROOT = Path(__file__).resolve().parents[2]  # where alembic.ini lives
_CACHE_TIERS = ("previews", "thumbs")
_OUTPUT_ROOTS = ("library", "exported")  # settings.jpeg_subdir is appended at runtime


def _empty_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def end_session(force: bool = False) -> None:
    """Delete the DB, the preview/thumb caches and the output trees, then
    run ``alembic upgrade head``. ``photos/incoming/`` is never touched."""
    output_roots = (*_OUTPUT_ROOTS, settings.jpeg_subdir)
    if not force:
        dirs = ",".join(output_roots)
        sys.stdout.write(
            f"About to wipe DB, cache/, and photos/{{{dirs}}}/.\n"
            "Have you saved anything you need to keep? [y/N] "
        )
        sys.stdout.flush()
        ans = sys.stdin.readline().strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    db = settings.db_path
    for f in (db, db.with_suffix(".db-wal"), db.with_suffix(".db-shm")):
        f.unlink(missing_ok=True)

    for tier in _CACHE_TIERS:
        _empty_dir(settings.cache / tier)
        (settings.cache / tier).mkdir(parents=True, exist_ok=True)

    for sub in output_roots:
        _empty_dir(settings.photos / sub)
        (settings.photos / sub).mkdir(parents=True, exist_ok=True)

    subprocess.run(["alembic", "upgrade", "head"], check=True, cwd=str(_REPO_ROOT))
    print("Session reset. Drop a new batch into photos/incoming/ to start fresh.")


async def reset_session() -> None:
    """Async wrapper for the UI: runs the blocking wipe in a worker thread.

    The wipe unlinks session.db; pooled connections still point at the old
    inode and would serve stale data, so the engine is disposed afterwards.
    """
    await asyncio.to_thread(end_session, True)

    from app.db import engine

    engine.dispose()
