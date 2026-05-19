"""Develop a RAW via darktable-cli to a full-resolution 16-bit linear TIFF."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def darktable_cli(
    raw: Path,
    xmp: Path | None = None,
    out_path: Path | None = None,
    out_bit_depth: int = 16,
    out_colorspace: str = "linear_rec2020",
) -> Path:
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        out_path = Path(tmp.name)
        tmp.close()
        # darktable-cli refuses to overwrite; drop the zero-byte placeholder.
        out_path.unlink(missing_ok=True)
    cmd = [
        "darktable-cli",
        str(raw),
    ]
    if xmp is not None:
        cmd.append(str(xmp))
    cmd.extend(
        [
            str(out_path),
            "--core",
            "--conf",
            f"plugins/imageio/format/tiff/bpp={out_bit_depth}",
            "--conf",
            "plugins/imageio/format/tiff/compress=0",
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
