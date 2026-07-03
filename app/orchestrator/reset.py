"""In-container session reset for the "New batch" UI action."""

from __future__ import annotations

import asyncio

from scripts.end_session import end_session

RESET_CONFIRM_TOKEN = "RESET"


async def reset_session() -> None:
    """Wipe DB + cache + working dirs and re-run migrations.

    `end_session` is blocking (file IO + an alembic subprocess), so it runs
    in a worker thread to keep the event loop responsive.
    """
    await asyncio.to_thread(end_session, True)

    from app.db import engine

    # The wipe unlinked session.db; pooled connections still point at the old
    # inode and would serve stale data. Drop them so new queries reopen the file.
    engine.dispose()
