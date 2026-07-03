"""System monitor: psutil sampling, NVML graceful degradation, disk warnings."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from app.monitor import stats as stats_mod
from app.monitor.stats import SystemMonitor


def test_sample_shape_without_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    sample = mon.sample()
    assert sample["gpu"] is None
    assert 0 <= sample["cpu_pct"] <= 100 * 64
    assert sample["ram_total_mb"] > 0
    assert sample["disks"]["photos"]["total_gb"] > 0


def test_missing_mount_degrades_to_none(monkeypatch) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": Path("/definitely-not-a-mount")})
    sample = mon.sample()
    assert sample["disks"]["photos"] is None


def test_snapshot_includes_window_and_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    mon.sample()
    mon.sample()
    snap = mon.snapshot()
    assert len(snap["window"]) == 2
    assert snap["current"] == snap["window"][-1]
    assert isinstance(snap["warnings"], list)


def test_low_disk_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path}, disk_warn_free_gb=20.0)
    sample = {
        "disks": {"photos": {"used_gb": 497.0, "total_gb": 500.0, "free_gb": 3.0}},
    }
    warnings = mon._warnings(sample)
    assert any("free space" in w.lower() for w in warnings)


def test_disk_warning_respects_threshold_param(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path}, disk_warn_free_gb=20.0)
    sample = {
        "disks": {"photos": {"used_gb": 400.0, "total_gb": 500.0, "free_gb": 100.0}},
    }
    assert mon._warnings(sample) == []


def test_gpu_fatal_failure_disables_further_reads(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky() -> dict[str, Any] | None:
        calls["n"] += 1
        raise stats_mod.GpuUnavailableError("no nvml")

    monkeypatch.setattr(stats_mod, "_read_gpu", flaky)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    first = mon.sample()
    second = mon.sample()
    assert calls["n"] == 1  # second sample skipped the broken NVML path
    assert first["gpu"] is None
    assert second["gpu"] is None


def test_gpu_transient_failure_does_not_latch(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky() -> dict[str, Any] | None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(stats_mod, "_read_gpu", flaky)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    first = mon.sample()
    second = mon.sample()
    assert calls["n"] == 2  # transient None keeps probing on every sample
    assert first["gpu"] is None
    assert second["gpu"] is None


def test_gpu_present_path_keeps_reading(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}
    gpu_stats = {"util_pct": 42.0, "vram_used_mb": 2048, "vram_total_mb": 6144, "temp_c": 61}

    def healthy() -> dict[str, Any]:
        calls["n"] += 1
        return dict(gpu_stats)

    monkeypatch.setattr(stats_mod, "_read_gpu", healthy)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    first = mon.sample()
    second = mon.sample()
    assert first["gpu"]["util_pct"] == 42.0
    assert second["gpu"] == gpu_stats
    assert calls["n"] == 2  # a working GPU is read on every sample


def test_window_caps_at_window_samples(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "_read_gpu", lambda: None)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    for _ in range(stats_mod.WINDOW_SAMPLES + 5):
        mon.sample()
    snap = mon.snapshot()
    assert len(snap["window"]) == stats_mod.WINDOW_SAMPLES


def test_run_survives_sample_exception(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stats_mod, "SAMPLE_INTERVAL_SECONDS", 0.01)
    mon = SystemMonitor(mounts={"photos": tmp_path})
    calls = {"n": 0}

    def sometimes_broken() -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    monkeypatch.setattr(mon, "sample", sometimes_broken)

    async def scenario() -> None:
        task = asyncio.ensure_future(mon.run())
        try:
            while calls["n"] < 2:
                await asyncio.sleep(0.01)
        finally:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
    assert calls["n"] >= 2  # the loop survived the first raise and kept sampling
