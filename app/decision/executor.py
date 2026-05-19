"""Atomic transactional file mover. All-or-nothing within one batch."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Move:
    src: Path
    dst: Path


def _safe_move(move: Move) -> Move:
    move.dst.parent.mkdir(parents=True, exist_ok=True)
    if move.dst.exists():
        raise FileExistsError(f"refusing to overwrite {move.dst}")
    try:
        move.src.rename(move.dst)
    except OSError:
        # Cross-device fallback.
        shutil.move(str(move.src), str(move.dst))
    return move


def apply_moves(moves: list[Move]) -> list[Move]:
    """Apply moves; on any failure, undo prior moves and re-raise."""
    done: list[Move] = []
    try:
        for m in moves:
            done.append(_safe_move(m))
        return done
    except Exception:
        log.exception("apply_moves failed; rolling back %d move(s)", len(done))
        for m in reversed(done):
            try:
                m.dst.rename(m.src)
            except Exception:
                log.exception("rollback failed for %s -> %s", m.dst, m.src)
        raise
