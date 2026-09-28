"""Develop a RAW via darktable-cli to a 16-bit **linear Rec.2020** TIFF.

darktable-cli argument order is strict: positional input [xmp] output, then
darktable-cli options (``--icc-type``), then ``--core`` followed by darktable
core options (``--conf``). Without ``--icc-type`` darktable exports sRGB.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import tifffile

from app.config import settings

ICC_LINEAR_REC2020 = "LIN_REC2020"
_TIFF_ICC_TAG = 34675


def build_darktable_command(
    raw: Path,
    out: Path,
    xmp: Path | None = None,
    *,
    icc_type: str = ICC_LINEAR_REC2020,
    bpp: int = 16,
    workflow: str | None = None,
) -> list[str]:
    cmd = ["darktable-cli", str(raw)]
    if xmp is not None:
        cmd.append(str(xmp))
    cmd += [str(out), "--icc-type", icc_type, "--core"]
    cmd += [
        "--conf",
        f"plugins/imageio/format/tiff/bpp={bpp}",
        "--conf",
        "plugins/imageio/format/tiff/compress=0",
    ]
    if workflow:
        cmd += ["--conf", f"plugins/darkroom/workflow={workflow}"]
    return cmd


def darktable_cli(raw: Path, xmp: Path | None = None, out_path: Path | None = None) -> Path:
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        out_path = Path(tmp.name)
        tmp.close()
        out_path.unlink(missing_ok=True)  # darktable-cli refuses to overwrite
    cmd = build_darktable_command(raw, out_path, xmp, workflow=settings.darktable_workflow)
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def read_icc_profile(tiff: Path) -> bytes | None:
    with tifffile.TiffFile(str(tiff)) as tf:
        tag = tf.pages[0].tags.get(_TIFF_ICC_TAG)  # type: ignore[union-attr]
        return bytes(tag.value) if tag is not None else None
