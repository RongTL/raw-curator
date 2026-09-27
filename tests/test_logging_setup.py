"""Process-wide logging is configured once, at the CLI entrypoint."""

from __future__ import annotations

import logging

from app.logging_setup import configure_logging


def test_configure_logging_installs_one_info_handler(monkeypatch) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.WARNING)

    configure_logging()
    configure_logging()  # idempotent: a second call must not stack handlers

    assert root.level == logging.INFO
    assert len(root.handlers) == 1


def test_configure_logging_respects_explicit_level(monkeypatch) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    configure_logging(level="DEBUG")
    assert root.level == logging.DEBUG
