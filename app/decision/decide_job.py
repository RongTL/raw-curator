"""Apply staged decisions: compute action from (selected, score_tier), move files, mark applied."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from sqlalchemy import select, update

from app.config import settings
from app.db import session_scope
from app.decision.executor import Move, apply_moves
from app.decision.rules import resolve, tier_from_scores
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
            tier = decision.score_tier
            if tier == "unset":
                tier = tier_from_scores(photo.technical_score, photo.aesthetic_score)
            rule = resolve(decision.selected, tier)
            if rule is None:
                log.info("skip %s: undecided", photo.hash)
                continue
            src = Path(photo.source_path)
            if not src.exists():
                log.warning("skip %s: source path missing %s", photo.hash, src)
                continue
            dst = settings.photos / rule.dest_subdir / src.name
            moves.append(Move(src=src, dst=dst))
            updates.append(
                {
                    "photo_hash": photo.hash,
                    "action": rule.action,
                    "score_tier": tier,
                    "applied": 1,
                    "new_source_path": str(dst),
                }
            )

        applied = apply_moves(moves)
        console.print(f"[green]Moved {len(applied)} file(s).[/green]")

        for u in updates:
            sess.execute(
                update(Decision)
                .where(Decision.photo_hash == u["photo_hash"])
                .values(action=u["action"], score_tier=u["score_tier"], applied=u["applied"])
            )
            sess.execute(
                update(Photo)
                .where(Photo.hash == u["photo_hash"])
                .values(source_path=u["new_source_path"])
            )

    console.print("[green]Decisions applied.[/green]")
