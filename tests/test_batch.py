"""Step-major batch: each model loads once; one failing frame does not stop the others."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from app.config import settings
from app.enhancement import batch
from app.enhancement.engine.plan import EnhancementPlan, StepSpec
from app.enhancement.enhance_job import PhotoCandidate
from tests.test_enhance_job import _report

ICC = Path("tests/fixtures/linear_rec2020.icc").read_bytes()


def _forced_plan(names: tuple[str, ...]) -> EnhancementPlan:
    """A plan with exactly these AI steps and default params."""
    return EnhancementPlan(
        steps=tuple(StepSpec(n) for n in names), report=_report(), has_faces=False
    )


class FakeModel:
    loads = 0
    applies: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        FakeModel.loads += 1
        return self

    def __exit__(self, *exc: object) -> None: ...

    def apply(self, rgb, **params):
        FakeModel.applies.append(self.name)
        return rgb


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "photos", tmp_path / "photos")
    monkeypatch.setattr(settings, "cache", tmp_path / "cache")
    monkeypatch.setattr(settings, "xmp", tmp_path / "xmp")
    monkeypatch.setattr(settings, "enhance_target_res", "native")
    (tmp_path / "photos" / "incoming").mkdir(parents=True)
    FakeModel.loads = 0
    FakeModel.applies = []
    monkeypatch.setattr(
        batch,
        "AI_MODELS",
        {
            "scunet_denoise": lambda: FakeModel("scunet"),
            "realesrgan_upscale": lambda: FakeModel("esrgan"),
            "codeformer_restore": lambda: FakeModel("cf"),
        },
    )
    monkeypatch.setattr(
        batch, "persist_all", lambda *a, **k: None
    )  # DB writes are covered elsewhere

    def fake_develop(raw: Path, xmp=None, out_path=None) -> Path:
        if raw.name.startswith("BAD"):
            raise RuntimeError("darktable-cli failed")
        out = tmp_path / f"{raw.stem}.dev.tif"
        noisy = np.random.default_rng(1).random((64, 96, 3)) * 0.3 + 0.2
        tifffile.imwrite(
            out,
            (noisy * 65535).astype(np.uint16),
            photometric="rgb",
            extratags=[(34675, "B", len(ICC), ICC, True)],
        )
        return out

    def candidate(name: str) -> tuple[PhotoCandidate, list]:
        p = tmp_path / "photos" / "incoming" / name
        p.write_bytes(b"raw")
        return PhotoCandidate(name, str(p), "raw", "keep_and_enhance"), []

    return tmp_path, fake_develop, candidate


def test_models_load_once_for_the_whole_batch(env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path, fake_develop, candidate = env
    monkeypatch.setattr(
        batch,
        "plan_for",
        lambda report, **kw: _forced_plan(("scunet_denoise", "realesrgan_upscale")),
    )
    summary = batch.run_batch(
        [candidate("A.CR3"), candidate("B.CR3"), candidate("C.CR3")], develop=fake_develop
    )
    assert (summary.enhanced, summary.failed) == (3, 0)
    assert FakeModel.loads == 2  # scunet once, esrgan once, never codeformer
    assert FakeModel.applies.count("scunet") == 3
    assert not list((tmp_path / "cache" / "enhance").glob("*.npy"))
    assert sorted(p.name for p in (tmp_path / "photos" / "exported").glob("*.tif")) == [
        "A.tif",
        "B.tif",
        "C.tif",
    ]


def test_one_failing_develop_does_not_stop_the_batch(env) -> None:
    tmp_path, fake_develop, candidate = env
    summary = batch.run_batch([candidate("BAD.CR3"), candidate("OK.CR3")], develop=fake_develop)
    assert (summary.enhanced, summary.failed) == (1, 1)
    assert (tmp_path / "photos" / "exported" / "OK.tif").exists()
    assert not list((tmp_path / "cache" / "enhance").glob("*.npy"))
