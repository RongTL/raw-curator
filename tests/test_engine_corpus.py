"""Plan-level expectations over the real corpus (skipped unless tests/data/corpus has RAWs)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from app.enhancement.develop_full import darktable_cli
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from tests.corpus_expectations import EXPECT

CORPUS = Path("tests/data/corpus")
pytestmark = pytest.mark.real_raw


def _develop_linear(raw: Path, tmp_path: Path) -> np.ndarray:
    out = darktable_cli(raw, out_path=tmp_path / "dev.tif")
    return tifffile.imread(str(out))[..., :3].astype(np.float32) / 65535.0


def _iso(raw: Path) -> int | None:
    proc = subprocess.run(
        ["exiftool", "-s3", "-ISO", str(raw)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout.strip()
    return int(out) if out else None


@pytest.mark.parametrize("stem", sorted(EXPECT))
def test_plan_matches_expectation(stem: str, tmp_path: Path) -> None:
    raw = CORPUS / f"{stem}.CR3"
    if not raw.exists():
        pytest.skip(f"{raw} not present")
    plan = plan_from_report(
        score_report(measure_all(_develop_linear(raw, tmp_path))), iso=_iso(raw)
    )
    names = {s.name for s in plan.steps}
    exp = EXPECT[stem]
    assert set(exp.get("present", ())) <= names, f"missing {set(exp.get('present', ())) - names}"
    assert not (
        set(exp.get("absent", ())) & names
    ), f"unexpected {set(exp.get('absent', ())) & names}"
    if "clahe_max" in exp:
        clips = [s.params["clip_limit"] for s in plan.steps if s.name == "clahe_local_contrast"]
        assert all(c <= exp["clahe_max"] for c in clips)
