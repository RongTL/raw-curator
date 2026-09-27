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
        if self.name == "esrgan":  # Real-ESRGAN honours scale=2: return a 2x image
            return np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
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


def test_x2_master_lands_at_the_target_resolution(env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path, fake_develop, candidate = env
    monkeypatch.setattr(settings, "enhance_target_res", "200%")
    # env's FakeModel("esrgan") already returns a 2x image, honouring scale=2.
    monkeypatch.setattr(
        batch, "plan_for", lambda report, **kw: _forced_plan(("realesrgan_upscale",))
    )
    summary = batch.run_batch([candidate("A.CR3")], develop=fake_develop)
    assert summary.enhanced == 1
    master = tifffile.imread(tmp_path / "photos" / "exported" / "A.tif")
    # fixture develops a (64, 96, 3) frame; 200% target -> exactly 2x each dim.
    assert master.shape == (128, 192, 3)


def test_missing_source_is_skipped_with_a_warning(env, caplog: pytest.LogCaptureFixture) -> None:
    tmp_path, fake_develop, _ = env
    missing = PhotoCandidate(
        "Z", str(tmp_path / "photos" / "incoming" / "GONE.CR3"), "raw", "keep_and_enhance"
    )
    with caplog.at_level("WARNING"):
        summary = batch.run_batch([(missing, [])], develop=fake_develop)
    assert (summary.skipped, summary.enhanced) == (1, 0)
    assert "source missing" in caplog.text


def test_non_raw_source_is_skipped_with_a_warning(env, caplog: pytest.LogCaptureFixture) -> None:
    tmp_path, fake_develop, _ = env
    p = tmp_path / "photos" / "incoming" / "S.JPG"
    p.write_bytes(b"jpeg")
    jpeg = PhotoCandidate("S", str(p), "jpeg", "keep_and_enhance")
    with caplog.at_level("WARNING"):
        summary = batch.run_batch([(jpeg, [])], develop=fake_develop)
    assert summary.skipped == 1
    assert "is not RAW" in caplog.text


def test_model_load_failure_fails_only_its_photos(env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path, fake_develop, _ = env

    class RaisingEnter:
        def __enter__(self):
            raise RuntimeError("cuda oom on model load")

        def __exit__(self, *exc: object) -> None: ...

        def apply(self, rgb, **params):
            return rgb

    monkeypatch.setattr(
        batch,
        "AI_MODELS",
        {
            "scunet_denoise": lambda: RaisingEnter(),
            "realesrgan_upscale": lambda: FakeModel("esrgan"),
            "codeformer_restore": lambda: FakeModel("cf"),
        },
    )

    def plan_by_iso(report, **kw):
        if kw.get("iso") == 100:  # A needs SCUNet (which will fail to load)
            return _forced_plan(("scunet_denoise", "realesrgan_upscale"))
        return _forced_plan(("realesrgan_upscale",))  # B needs only ESRGAN

    monkeypatch.setattr(batch, "plan_for", plan_by_iso)

    def cand(name: str, iso: int | None) -> tuple[PhotoCandidate, list]:
        p = tmp_path / "photos" / "incoming" / name
        p.write_bytes(b"raw")
        return PhotoCandidate(name, str(p), "raw", "keep_and_enhance", iso=iso), []

    summary = batch.run_batch([cand("A.CR3", 100), cand("B.CR3", None)], develop=fake_develop)
    assert (summary.enhanced, summary.failed) == (1, 1)
    assert (tmp_path / "photos" / "exported" / "B.tif").exists()
    assert not (tmp_path / "photos" / "exported" / "A.tif").exists()
    assert not list((tmp_path / "cache" / "enhance").glob("*.npy"))


def test_preview_missing_warns_before_dropping_face_boxes(
    env, caplog: pytest.LogCaptureFixture
) -> None:
    tmp_path, fake_develop, _ = env
    p = tmp_path / "photos" / "incoming" / "F.CR3"
    p.write_bytes(b"raw")
    photo = PhotoCandidate("F", str(p), "raw", "keep_and_enhance", preview_path=None)
    with caplog.at_level("WARNING"):
        summary = batch.run_batch([(photo, [(10, 10, 20, 20)])], develop=fake_develop)
    assert summary.enhanced == 1
    assert "preview missing" in caplog.text
