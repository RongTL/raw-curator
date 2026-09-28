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
from app.enhancement.engine.plan import FaceInfo
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
    img = _develop_linear(raw, tmp_path)
    exp = EXPECT[stem]
    # A "faces" key lists (x, y, w, h) boxes in developed-image pixels; the harness
    # treats each as a sharp, large face so the planner's face-aware branches fire.
    faces = tuple(
        FaceInfo(box=(int(x), int(y), int(w), int(h)), lap_var=400.0)
        for (x, y, w, h) in exp.get("faces", ())
    )
    plan = plan_from_report(
        score_report(measure_all(img)),
        iso=_iso(raw),
        native_long_edge=max(img.shape[:2]),
        faces=faces,
    )
    names = {s.name for s in plan.steps}
    assert set(exp.get("present", ())) <= names, f"missing {set(exp.get('present', ())) - names}"
    assert not (
        set(exp.get("absent", ())) & names
    ), f"unexpected {set(exp.get('absent', ())) & names}"
    if "clahe_max" in exp:
        clips = [s.params["clip_limit"] for s in plan.steps if s.name == "clahe_local_contrast"]
        assert all(c <= exp["clahe_max"] for c in clips)


@pytest.mark.parametrize("stem", ["IMG_0098", "IMG_1124", "IMG_1177"])
def test_lens_correction_applies_to_corpus_frame(stem: str, tmp_path: Path) -> None:
    pytest.importorskip("lensfunpy")
    from app.enhancement.classical.lens_correct import correct_lens
    from app.ingest.exif import ExifReader

    raw = CORPUS / f"{stem}.CR3"
    if not raw.exists():
        pytest.skip(f"{raw} not present")
    img = _develop_linear(raw, tmp_path)
    with ExifReader() as ex:
        exif = ex.read(raw)
    out = correct_lens(img, exif, enabled=True)
    # The whole corpus is the RF 24/1.8 Macro, which our shipped XML resolves, so
    # the geometry/CA remap must fire and preserve shape/dtype/range.
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not np.array_equal(out, img)
