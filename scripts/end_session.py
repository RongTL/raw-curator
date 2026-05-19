"""Wipe all session state. The defining operation of the ephemeral model."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from app.config import settings


def _empty_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def end_session(force: bool = False) -> None:
    if not force:
        sys.stdout.write(
            "About to wipe DB, cache/, and photos/{library,archive,quarantine,exported}/.\n"
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

    cache = settings.cache
    _empty_dir(cache / "previews")
    _empty_dir(cache / "thumbs")
    (cache / "previews").mkdir(parents=True, exist_ok=True)
    (cache / "thumbs").mkdir(parents=True, exist_ok=True)

    for sub in ("library", "archive", "quarantine", "exported"):
        _empty_dir(settings.photos / sub)
        (settings.photos / sub).mkdir(parents=True, exist_ok=True)

    cache.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    print("Session reset. Drop a new batch into photos/incoming/ to start fresh.")


if __name__ == "__main__":
    end_session(force="--force" in sys.argv)
