"""Process-wide logging configuration.

Called once from the Typer callback so every `raw-curator <stage>` process
(including the ones the UI spawns) emits its `log.info/warning` lines to
stderr, where the JobRunner captures them into the stage log. Without this,
Python's last-resort handler dropped everything below WARNING.
"""

from __future__ import annotations

import logging
import sys

from app.config import settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Idempotent: installs one stderr handler on the root logger and sets its level.

    ``level`` defaults to ``settings.log_level`` (RAWCURATOR_LOG_LEVEL).
    """
    root = logging.getLogger()
    resolved = logging.getLevelName((level or settings.log_level).upper())
    if not isinstance(resolved, int):
        raise ValueError(f"unknown log level: {level or settings.log_level!r}")
    root.setLevel(resolved)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
        root.addHandler(handler)
