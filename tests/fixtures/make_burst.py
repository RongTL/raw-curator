"""Synthesize a burst of near-identical JPEGs with controlled EXIF timestamps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def make_burst(
    out_dir: Path,
    n: int = 5,
    base_time: datetime | None = None,
    spacing_seconds: float = 0.5,
    size: tuple[int, int] = (640, 480),
    color: tuple[int, int, int] = (128, 64, 200),
    jitter: int = 5,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_time = base_time or datetime(2026, 1, 1, 12, 0, 0)
    paths: list[Path] = []
    for i in range(n):
        arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        arr += (np.random.randint(-jitter, jitter, arr.shape)).astype(np.int16).clip(-jitter, jitter).astype(np.uint8)
        out = out_dir / f"burst_{i:03d}.jpg"
        Image.fromarray(arr).save(out, format="JPEG", quality=90)
        paths.append(out)
    return paths


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/data/burst")
    print(f"writing {target}")
    paths = make_burst(target)
    for p in paths:
        print(f"  {p}")
