"""Apply staged decisions.

Binary routing:
- yes -> move RAW to photos/library/, action=keep_and_enhance
- no  -> leave RAW in place so `make enhance` can still read it;
         action=enhance_only. The source is deleted by enhance_job after
         the TIFF is successfully written.

Either way the row is marked applied=1; `make enhance` then picks up
every photo with an applied keep_and_enhance / enhance_only action.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from sqlalchemy import select, update

from app.config import settings
from app.db import session_scope
from app.decision.executor import Move, apply_moves
from app.decision.rules import resolve
from app.models import Decision, Photo

log = logging.getLogger(__name__)
console = Console()


def apply_decisions() -> None:
    moves: list[Move] = []
    updates: list[dict] = []

    with session_scope() as sess:
        rows = (
            sess.execute(
                select(Photo, Decision)
                .join(Decision, Photo.hash == Decision.photo_hash)
                .where(Decision.applied == 0)
            )
            .all()
        )
        if not rows:
            console.print("[yellow]No pending decisions.[/yellow]")
            return

        for photo, decision in rows:
            rule = resolve(decision.selected)
            if rule is None:
                log.info("skip %s: undecided", photo.hash)
                continue
            src = Path(photo.source_path)
            if not src.exists():
                log.warning("skip %s: source path missing %s", photo.hash, src)
                continue

            new_source_path = str(src)
            if rule.library_subdir is not None:
                dst = settings.photos / rule.library_subdir / src.name
                moves.append(Move(src=src, dst=dst))
                new_source_path = str(dst)

            updates.append(
                {
                    "photo_hash": photo.hash,
                    "action": rule.action,
                    "applied": 1,
                    "new_source_path": new_source_path,
                }
            )

        applied = apply_moves(moves)
        console.print(f"[green]Moved {len(applied)} file(s) to library/.[/green]")

        for u in updates:
            sess.execute(
                update(Decision)
                .where(Decision.photo_hash == u["photo_hash"])
                .values(action=u["action"], applied=u["applied"])
            )
            sess.execute(
                update(Photo)
                .where(Photo.hash == u["photo_hash"])
                .values(source_path=u["new_source_path"])
            )

    console.print(
        f"[green]Decisions applied:[/green] {len(updates)} photo(s) queued for enhancement."
    )
