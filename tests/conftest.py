"""Test fixtures: ephemeral SQLite per test, no real RAW unless tests/data/ has files."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.db import make_engine
from app.models import Base


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[Session]:
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = make_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    sess = SessionLocal()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    real_raw_dir = Path(__file__).parent / "data"
    if not real_raw_dir.exists() or not any(real_raw_dir.iterdir()):
        skip_real_raw = pytest.mark.skip(reason="no real RAW fixtures in tests/data/")
        for item in items:
            if "real_raw" in item.keywords:
                item.add_marker(skip_real_raw)

    if not os.environ.get("RUN_GPU_TESTS"):
        skip_gpu = pytest.mark.skip(reason="set RUN_GPU_TESTS=1 to enable")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
