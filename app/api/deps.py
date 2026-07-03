"""Lazily-constructed singletons shared by API routes (single-user app)."""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.monitor.stats import SystemMonitor
from app.orchestrator.autorun import AutoRun
from app.orchestrator.runner import JobRunner


@lru_cache(maxsize=1)
def get_runner() -> JobRunner:
    return JobRunner(log_dir=settings.cache / "logs")


@lru_cache(maxsize=1)
def get_autorun() -> AutoRun:
    return AutoRun(get_runner())


@lru_cache(maxsize=1)
def get_monitor() -> SystemMonitor:
    return SystemMonitor(
        mounts={
            "photos": settings.photos,
            "cache": settings.cache,
            "models": settings.models,
        },
        disk_warn_free_gb=settings.monitor_disk_warn_free_gb,
    )
