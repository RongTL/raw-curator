"""SQLAlchemy engine + sqlite-vec extension loading."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import sqlite_vec
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def _load_sqlite_vec(dbapi_connection: sqlite3.Connection, _conn_record: object) -> None:
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


def make_engine(url: str | None = None) -> Engine:
    engine = create_engine(
        url or settings.db_url,
        future=True,
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _load_sqlite_vec)
    return engine


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    sess = SessionLocal()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()
