"""System resource sampling: CPU/RAM/disk via psutil, GPU via NVML (optional).

NVML absence or failure must never break the stats endpoint: the GPU field
degrades to None and further NVML reads are skipped for the process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 1.0
WINDOW_SAMPLES = 60
_MB = 1024 * 1024
_GB = 1024 * 1024 * 1024


class GpuUnavailableError(RuntimeError):
    """NVML is absent or the driver is not loaded — GPU stats disabled for process lifetime."""


def _read_gpu() -> dict[str, Any] | None:
    """Read GPU stats via NVML.

    Raises GpuUnavailableError for fatal conditions (pynvml missing, NVML
    library/driver missing, nvmlInit failure) so the caller stops probing.
    Query-phase failures are transient: logged at debug, reported as None.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except ImportError as exc:
        raise GpuUnavailableError("pynvml is not installed") from exc
    except Exception as exc:  # init-phase NVML failure is fatal for this process
        raise GpuUnavailableError("NVML init failed (library or driver missing)") from exc

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "util_pct": float(util.gpu),
            "vram_used_mb": int(mem.used) // _MB,
            "vram_total_mb": int(mem.total) // _MB,
            "temp_c": int(temp),
        }
    except Exception:  # query-phase blip (e.g. driver under load); do not latch
        logger.debug("transient NVML read failure", exc_info=True)
        return None


class SystemMonitor:
    def __init__(self, mounts: dict[str, Path], disk_warn_free_gb: float = 50.0) -> None:
        self._mounts = mounts
        self._disk_warn_free_gb = disk_warn_free_gb
        self._window: deque[dict[str, Any]] = deque(maxlen=WINDOW_SAMPLES)
        self._gpu_available = True

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        sample = {
            "ts": time.time(),
            "cpu_pct": psutil.cpu_percent(),
            "cpu_per_core": psutil.cpu_percent(percpu=True),
            "ram_used_mb": (vm.total - vm.available) // _MB,
            "ram_total_mb": vm.total // _MB,
            "disks": {name: self._disk(path) for name, path in self._mounts.items()},
            "gpu": self._gpu(),
        }
        self._window.append(sample)
        return sample

    def snapshot(self) -> dict[str, Any]:
        current = self._window[-1] if self._window else self.sample()
        return {
            "current": current,
            "window": list(self._window),
            "warnings": self._warnings(current),
        }

    async def run(self) -> None:
        while True:
            try:
                self.sample()
            except Exception:  # noqa: BLE001 — sampling must never die
                logger.exception("system sampling failed")
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)

    def _disk(self, path: Path) -> dict[str, float] | None:
        try:
            usage = psutil.disk_usage(str(path))
        except OSError:
            return None
        return {
            "used_gb": usage.used / _GB,
            "total_gb": usage.total / _GB,
            "free_gb": usage.free / _GB,
        }

    def _gpu(self) -> dict[str, Any] | None:
        if not self._gpu_available:
            return None
        try:
            return _read_gpu()
        except GpuUnavailableError:
            self._gpu_available = False
            logger.info("GPU stats unavailable (NVML absent or driver not loaded); disabling")
            return None

    def _warnings(self, sample: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        photos = (sample.get("disks") or {}).get("photos")
        if photos is not None and photos["free_gb"] < self._disk_warn_free_gb:
            warnings.append(
                f"Low free space on photos volume: {photos['free_gb']:.0f} GB left "
                "— enhance/export may fill the disk"
            )
        return warnings
