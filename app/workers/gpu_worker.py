"""Stage-swapped GPU worker helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from rich.console import Console

log = logging.getLogger(__name__)
console = Console()


def cuda_available() -> bool:
    return torch.cuda.is_available()


def assert_gpu_or_skip() -> bool:
    if not cuda_available():
        console.print("[yellow]CUDA not available — skipping GPU scoring.[/yellow]")
        return False
    return True


def warmup() -> None:
    # Pin torch's intra-op pool to the SMT thread count so CPU-side
    # preprocessing in GPU stages uses the whole R3 3100 without
    # oversubscribing (torch defaults to physical cores only).
    from app.config import settings

    torch.set_num_threads(max(1, settings.cpu_workers))
    if cuda_available():
        torch.cuda.empty_cache()


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def memory_summary() -> str:
    if not cuda_available():
        return "no cuda"
    used = torch.cuda.memory_allocated() / 1024 / 1024
    return f"alloc={used:.0f} MiB"
