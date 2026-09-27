"""darktable-cli invocation: option order matters (core options must follow --core)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from app.enhancement.develop_full import (
    ICC_LINEAR_REC2020,
    build_darktable_command,
    read_icc_profile,
)


def test_icc_type_precedes_core_and_xmp_sits_between_input_and_output() -> None:
    cmd = build_darktable_command(Path("/in/a.CR3"), Path("/out/a.tif"), Path("/x/a.CR3.xmp"))
    assert cmd[:4] == ["darktable-cli", "/in/a.CR3", "/x/a.CR3.xmp", "/out/a.tif"]
    assert cmd.index("--icc-type") < cmd.index("--core")
    assert cmd[cmd.index("--icc-type") + 1] == ICC_LINEAR_REC2020
    assert "plugins/imageio/format/tiff/bpp=16" in cmd


def test_workflow_conf_is_a_core_option() -> None:
    cmd = build_darktable_command(Path("a.CR3"), Path("a.tif"), workflow="scene-referred (sigmoid)")
    i = cmd.index("plugins/darkroom/workflow=scene-referred (sigmoid)")
    assert i > cmd.index("--core") and cmd[i - 1] == "--conf"


def test_read_icc_profile_returns_tag_bytes_or_none(tmp_path: Path) -> None:
    icc = Path("tests/fixtures/linear_rec2020.icc").read_bytes()
    arr = np.zeros((2, 2, 3), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "with.tif", arr, extratags=[(34675, "B", len(icc), icc, True)])
    tifffile.imwrite(tmp_path / "without.tif", arr)
    assert read_icc_profile(tmp_path / "with.tif") == icc
    assert read_icc_profile(tmp_path / "without.tif") is None


def test_default_workflow_is_sigmoid() -> None:
    from app.config import Settings

    assert Settings(_env_file=None).darktable_workflow == "scene-referred (sigmoid)"
