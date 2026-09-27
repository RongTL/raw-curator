# Enhancement Engine Correctness Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Auto Enhancement Engine measure and edit in linear Rec.2020, write proper masters, record and verify every recipe, load each AI model once per batch, and stop the always-on steps that fight the photograph.

**Architecture:** darktable develops each RAW to a tagged 16-bit linear Rec.2020 TIFF (baseline XMP unless the user supplies one). The engine measures, plans and runs classical steps in float32 linear; AI steps see display-referred sRGB uint8 and hand back a *delta* that is merged into the float image. A batch runs step-major (all pre-AI, then one model at a time, then post-AI). Each result is re-measured; degraded results retry a safe plan and never delete the source RAW.

**Tech Stack:** Python 3.12, numpy, tifffile, Pillow, exiftool (in the image), darktable-cli 4.6.1 (in the image), SQLAlchemy 2 + Alembic, torch (existing model wrappers), pytest. All tooling runs in the container: `make test`, `make lint`, `make typecheck`, or `/tmp/rc.sh test ...` for a single file.

**Spec:** `docs/superpowers/specs/2026-09-27-enhancement-correctness-design.md` (approved, D1–D4 made).

## Global Constraints

- Working space: linear Rec.2020, `float32` in `[0, 1]`, from darktable output to the final write. Anything `uint8` is display-referred sRGB. The only conversions live in `app/enhancement/colorspace.py`.
- Master TIFF: 16-bit, LZW, the ICC profile darktable emitted embedded (`ProfileDescription` = `Linear Rec2020 RGB`), EXIF/XMP copied from the RAW, `Orientation=1`.
- A source RAW is deleted only for `action == "enhance_only"`, only after the output TIFF exists, and never when verification says *degraded*.
- 6 GB VRAM: exactly one AI model resident at a time; `torch.cuda.empty_cache()` after each model unload.
- Migrations are additive (`0004`); `make reset` recreates from scratch.
- `ruff check`, `ruff format --check`, `mypy --strict` stay green; arrays are annotated with `app.arrays.Array`.
- Look-changing tasks (Task 4, 15, 17, 19–24) are merged only after the owner has seen the contact sheet from Task 13.
- Tests that need real RAWs use the `real_raw` marker and `tests/data/corpus/`; everything else runs on synthetic arrays.

## Review Focus

Inputs the spec implies but no task's tests would otherwise exercise, most likely to bite first. Each has a pinned test in the owning task.

1. darktable exits non-zero on one frame in a step-major batch → that photo is *failed*, the batch continues, no intermediate leaks (Task 12).
2. The developed TIFF carries no ICC tag (older darktable, user XMP that overrides output profile) → the master is still written, untagged, with a warning; JPEG export treats it as sRGB (Task 5, Task 6).
3. The cached preview JPEG is missing when face boxes are scaled → boxes are dropped and the photo proceeds with `has_faces=False` (Task 7).
4. `Q_before` is already low (e.g. 20) → verification thresholds are relative, so a 3-point drop is not *degraded*, but a collapsed image still is (Task 10).
5. `enhance_target_res=200%` with the delta merge → the float base is Lanczos-upsampled ×2 before the ×2 delta is added, and the output is 2× native (Task 19).

---

## File structure

| Path | Responsibility |
|---|---|
| `app/enhancement/colorspace.py` (modify) | luma/YCbCr (existing) + Rec.2020↔sRGB matrices, sRGB OETF/EOTF, uint8 boundary helpers |
| `app/enhancement/develop_full.py` (modify) | `build_darktable_command`, `darktable_cli`, `read_icc_profile` |
| `app/enhancement/pack_tiff.py` (modify) | `write_tiff16(..., icc=)`, `copy_metadata` |
| `app/enhancement/geometry.py` (create) | `scale_boxes` |
| `app/enhancement/sidecar.py` (create) | `resolve_xmp` (user XMP → baseline) |
| `app/enhancement/verify.py` (create) | `Verdict`, `verify`, `safe_plan` |
| `app/enhancement/batch.py` (create) | step-major orchestration, intermediates, model registry |
| `app/enhancement/denoise.py`, `upscale.py`, `face_restore.py` (modify) | `ScunetModel`, `RealEsrganModel`, `CodeFormerModel` load-once classes; functional wrappers become adapters |
| `app/enhancement/engine/plan.py` (modify) | `plan_to_json`; `QualityReport` gains Step-3 metrics |
| `app/enhancement/engine/metrics.py` (modify) | `as_linear_float01`; neutral-pixel WB, OKLCh chroma, block sharpness metrics |
| `app/enhancement/engine/decision.py` (modify) | no always-on tone map; ISO, target-scale, face-size gates; restraint rules |
| `app/enhancement/engine/runner.py` (modify) | split into `apply_pre_ai` / `apply_post_ai` / `apply_ai_delta`; sRGB boundary |
| `app/enhancement/enhance_job.py` (modify) | `PhotoCandidate` gains `preview_path`, `iso`, faces with embeddings; delegates to `batch.run_batch` |
| `app/ingest/decode.py` (modify) | profile-aware `load_tiff_rgb8`, `icc_description` |
| `app/models.py`, `db/migrations/versions/0004_plan_and_verify.py` | new nullable columns on `quality_reports` |
| `app/api/serializers.py`, `app/api/static/js/review.js` (modify) | recipe + verification in the detail panel |
| `app/config.py` (modify) | `darktable_workflow`, `enhance_sr_min_long_edge`, `enhance_face_restore_max_px` |
| `app/enhancement/darktable/raw-curator-base.xmp` (create, owner-authored) | baseline development |
| `scripts/contact_sheet.py` (create) | before/after review sheets |
| `tests/fixtures/linear_rec2020.icc` (create) | darktable's profile bytes for tests |
| `tests/test_colorspace.py`, `tests/test_develop_full.py`, `tests/test_pack_tiff.py`, `tests/test_geometry.py`, `tests/test_sidecar.py`, `tests/test_verify.py`, `tests/test_batch.py`, `tests/test_engine_corpus.py`, `tests/corpus_expectations.py` | tests |

---

## Step 1 — colour pipeline and correctness

### Task 1: Rec.2020 ↔ sRGB transforms

**Files:**
- Modify: `app/enhancement/colorspace.py`
- Test: `tests/test_colorspace.py`

**Interfaces:**
- Produces: `encode_srgb(lin: Array) -> Array`, `decode_srgb(enc: Array) -> Array`, `rec2020_to_srgb_linear(lin: Array) -> Array`, `srgb_to_rec2020_linear(lin: Array) -> Array`, `linear_rec2020_to_srgb_u8(lin: Array) -> Array` (uint8), `srgb_u8_to_linear_rec2020(u8: Array) -> Array` (float32). All take HxWx3.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_colorspace.py`)

```python
from app.enhancement.colorspace import (
    decode_srgb, encode_srgb, linear_rec2020_to_srgb_u8, rec2020_to_srgb_linear,
    srgb_to_rec2020_linear, srgb_u8_to_linear_rec2020,
)


def test_srgb_transfer_round_trip_and_anchors() -> None:
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32).reshape(1, 11, 1).repeat(3, axis=2)
    assert np.allclose(decode_srgb(encode_srgb(x)), x, atol=1e-6)
    assert abs(float(encode_srgb(np.array([[[0.18, 0.18, 0.18]]], dtype=np.float32))[0, 0, 0]) - 0.4613) < 1e-3
    assert float(encode_srgb(np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32))[0, 0, 0]) == 1.0


def test_gamut_matrices_are_inverses_and_keep_white() -> None:
    white = np.ones((1, 1, 3), dtype=np.float32)
    assert np.allclose(rec2020_to_srgb_linear(white), white, atol=2e-3)
    rng = np.random.default_rng(1).random((4, 5, 3), dtype=np.float32)
    assert np.allclose(srgb_to_rec2020_linear(rec2020_to_srgb_linear(rng)), rng, atol=1e-4)


def test_u8_boundary_round_trip_is_within_one_code() -> None:
    lin = np.random.default_rng(2).random((6, 6, 3), dtype=np.float32) * 0.8 + 0.05
    u8 = linear_rec2020_to_srgb_u8(lin)
    assert u8.dtype == np.uint8
    back = srgb_u8_to_linear_rec2020(u8)
    assert np.abs(linear_rec2020_to_srgb_u8(back).astype(int) - u8.astype(int)).max() <= 1
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_colorspace.py` → ImportError on the new names.

- [ ] **Step 3: Implement** (append to `app/enhancement/colorspace.py`)

```python
# Rec.2020 (D65) <-> sRGB (D65) linear-light 3x3 matrices (BT.2087 / IEC 61966-2-1).
REC2020_TO_SRGB = np.array(
    [[1.6604910, -0.5876411, -0.0728499],
     [-0.1245505, 1.1328999, -0.0083494],
     [-0.0181508, -0.1005789, 1.1187297]], dtype=np.float32)
SRGB_TO_REC2020 = np.array(
    [[0.6274040, 0.3292820, 0.0433136],
     [0.0690970, 0.9195400, 0.0113612],
     [0.0163916, 0.0880132, 0.8955950]], dtype=np.float32)


def encode_srgb(lin: Array) -> Array:
    """Linear light -> sRGB-encoded, both float in [0, 1] (clips)."""
    f = np.clip(lin, 0.0, 1.0).astype(np.float32)
    return np.where(f <= 0.0031308, 12.92 * f, 1.055 * np.power(f, 1.0 / 2.4) - 0.055).astype(np.float32)


def decode_srgb(enc: Array) -> Array:
    f = np.clip(enc, 0.0, 1.0).astype(np.float32)
    return np.where(f <= 0.04045, f / 12.92, np.power((f + 0.055) / 1.055, 2.4)).astype(np.float32)


def rec2020_to_srgb_linear(lin: Array) -> Array:
    return (lin @ REC2020_TO_SRGB.T).astype(np.float32)


def srgb_to_rec2020_linear(lin: Array) -> Array:
    return (lin @ SRGB_TO_REC2020.T).astype(np.float32)


def linear_rec2020_to_srgb_u8(lin: Array) -> Array:
    """The AI / JPEG boundary: gamut-map by clipping, sRGB-encode, quantise."""
    return (encode_srgb(rec2020_to_srgb_linear(lin)) * 255.0 + 0.5).astype(np.uint8)


def srgb_u8_to_linear_rec2020(u8: Array) -> Array:
    return srgb_to_rec2020_linear(decode_srgb(u8.astype(np.float32) / 255.0))
```

- [ ] **Step 4: Run tests** — `/tmp/rc.sh test tests/test_colorspace.py` → PASS.
- [ ] **Step 5: Commit** — `git add app/enhancement/colorspace.py tests/test_colorspace.py && git commit -m "feat(colorspace): Rec.2020<->sRGB matrices and sRGB transfer helpers"`

### Task 2: darktable command builder and ICC read-back

**Files:**
- Modify: `app/enhancement/develop_full.py` (rewrite)
- Modify: `app/config.py` (add `darktable_workflow`)
- Test: `tests/test_develop_full.py` (create)

**Interfaces:**
- Produces: `ICC_LINEAR_REC2020 = "LIN_REC2020"`; `build_darktable_command(raw: Path, out: Path, xmp: Path | None = None, *, icc_type: str = ICC_LINEAR_REC2020, bpp: int = 16, workflow: str | None = None) -> list[str]`; `darktable_cli(raw: Path, xmp: Path | None = None, out_path: Path | None = None) -> Path`; `read_icc_profile(tiff: Path) -> bytes | None`.
- Settings: `darktable_workflow: str = "scene-referred (filmic)"` (Task 15 flips the default).

- [ ] **Step 1: Write the failing tests**

```python
"""darktable-cli invocation: option order matters (core options must follow --core)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from app.enhancement.develop_full import ICC_LINEAR_REC2020, build_darktable_command, read_icc_profile


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
```

- [ ] **Step 2: Create the ICC fixture** (bytes darktable emitted for `--icc-type LIN_REC2020`):

```bash
/tmp/rc.sh sh 'exiftool -icc_profile -b tests/data/_review/cs/linear.tif > tests/fixtures/linear_rec2020.icc && exiftool tests/fixtures/linear_rec2020.icc | grep -E "Profile Description|File Size"'
```
Expected: `Profile Description: Linear Rec2020 RGB`, a few hundred bytes. If `tests/data/_review/cs/linear.tif` is gone, regenerate it with `darktable-cli tests/data/corpus/IMG_0030.CR3 linear.tif --icc-type LIN_REC2020 --core --conf plugins/imageio/format/tiff/bpp=16`.

- [ ] **Step 3: Run tests to verify failure** — `/tmp/rc.sh test tests/test_develop_full.py` → ImportError.

- [ ] **Step 4: Implement** — replace `app/enhancement/develop_full.py`:

```python
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
    cmd += ["--conf", f"plugins/imageio/format/tiff/bpp={bpp}", "--conf", "plugins/imageio/format/tiff/compress=0"]
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
        tag = tf.pages[0].tags.get(_TIFF_ICC_TAG)
        return bytes(tag.value) if tag is not None else None
```

Add to `Settings` in `app/config.py`, after `log_level`:

```python
    # darktable pixel workflow for the develop step; Task 15 switches to sigmoid.
    darktable_workflow: str = "scene-referred (filmic)"
```

- [ ] **Step 5: Run tests** — `/tmp/rc.sh test tests/test_develop_full.py tests/test_enhancement_weights.py` → PASS.
- [ ] **Step 6: Commit** — `git add app/enhancement/develop_full.py app/config.py tests/test_develop_full.py tests/fixtures/linear_rec2020.icc && git commit -m "fix(develop): export linear Rec.2020 from darktable and read the ICC back"`

### Task 3: Linear input contract and the sRGB boundary for AI steps

**Files:**
- Modify: `app/enhancement/engine/metrics.py` (rename `to_float01` → `as_linear_float01`, doc)
- Modify: `app/enhancement/engine/runner.py` (`_to_u8`/`_from_u8` use colorspace)
- Modify: `app/enhancement/enhance_job.py` (`_load_linear_float` docstring, use `as_linear_float01`)
- Test: `tests/test_engine_metrics.py`, `tests/test_engine_runner_boundary.py` (create)

**Interfaces:**
- Produces: `metrics.as_linear_float01(img: Array) -> Array` (same behaviour as the old name); runner's `_to_u8(rgb_lin) -> uint8 sRGB`, `_from_u8(u8) -> float32 linear Rec.2020`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_engine_metrics.py` replace the import `to_float01` with `as_linear_float01` and the test body names accordingly (three call sites at lines 20, 99–105). Create `tests/test_engine_runner_boundary.py`:

```python
"""The AI boundary converts linear Rec.2020 float to display sRGB uint8 and back."""

from __future__ import annotations

import numpy as np

from app.enhancement.engine import runner


def test_boundary_encodes_linear_to_display_srgb() -> None:
    mid_grey_linear = np.full((2, 2, 3), 0.18, dtype=np.float32)
    u8 = runner._to_u8(mid_grey_linear)
    assert u8.dtype == np.uint8
    assert 115 <= int(u8[0, 0, 0]) <= 120  # 0.18 linear ~= 118/255 in sRGB, not 46/255


def test_boundary_round_trip_is_close_in_linear() -> None:
    lin = np.random.default_rng(3).random((4, 4, 3), dtype=np.float32) * 0.9
    back = runner._from_u8(runner._to_u8(lin))
    assert np.abs(back - lin).max() < 0.01
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_engine_metrics.py tests/test_engine_runner_boundary.py` → ImportError / first assertion fails (`_to_u8` of 0.18 gives 46).

- [ ] **Step 3: Implement**

`metrics.py`: rename `def to_float01(` → `def as_linear_float01(` and its call in `measure_all`; docstring: `"""Normalize uint8 / uint16 / float to float32 [0, 1]. Values are taken as linear light; the sRGB encoding for the spec's 8-bit thresholds happens in _luma_u8."""`. Replace the module docstring paragraph "Operates on full-resolution float32 linear RGB" with an explicit line: `Input contract: linear Rec.2020 float32 in [0, 1] (what develop_full produces).`

`runner.py`:

```python
from app.enhancement.colorspace import linear_rec2020_to_srgb_u8, srgb_u8_to_linear_rec2020


def _to_u8(rgb_lin: Array) -> Array:
    """AI boundary: linear Rec.2020 float -> display-referred sRGB uint8."""
    return linear_rec2020_to_srgb_u8(rgb_lin)


def _from_u8(rgb_u8: Array) -> Array:
    return srgb_u8_to_linear_rec2020(rgb_u8)
```

`enhance_job.py`: replace `_load_linear_float` entirely (Pillow cannot read 48-bit RGB TIFFs reliably; use tifffile):

```python
def _load_linear_float(tiff_path: Path) -> Array:
    """Load darktable's 16-bit linear Rec.2020 TIFF as float32 in [0, 1]."""
    arr = tifffile.imread(str(tiff_path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {arr.shape} from {tiff_path}")
    return as_linear_float01(arr)
```

(`import tifffile`; `from app.enhancement.engine.metrics import as_linear_float01`; drop the now-unused `PIL.Image` import if nothing else uses it — Task 7 adds `preview_size`, which does.) Add to `tests/test_enhance_job.py`:

```python
def test_load_linear_float_reads_16bit_rgb_tiff(tmp_path: Path) -> None:
    import tifffile
    from app.enhancement.enhance_job import _load_linear_float

    tifffile.imwrite(tmp_path / "t.tif", np.full((3, 3, 3), 32768, dtype=np.uint16), photometric="rgb")
    out = _load_linear_float(tmp_path / "t.tif")
    assert out.dtype == np.float32 and abs(float(out[0, 0, 0]) - 0.5) < 1e-3
```

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_engine_metrics.py tests/test_engine_runner_boundary.py tests/test_enhance_job.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "fix(engine): treat input as linear Rec.2020; AI boundary encodes to display sRGB"`

### Task 4: No always-on tone map

**Files:**
- Modify: `app/enhancement/engine/decision.py` (drop the final `tone_map_final` append)
- Modify: `app/enhancement/engine/runner.py` (`_POST_AI` loses `tone_map_final`; `_apply_classical` keeps the branch)
- Test: `tests/test_engine_decision.py`

- [ ] **Step 1: Update the tests** — in `test_balanced_input_plan_is_minimal` replace `assert names[-1] == "tone_map_final"` with `assert "tone_map_final" not in names`; in `test_plan_order_matches_spec_section_7` change `post = {"unsharp_mask", "clahe_local_contrast", "tone_map_final"}` to `post = {"unsharp_mask", "clahe_local_contrast"}`. Add:

```python
def test_tone_map_is_never_planned() -> None:
    for report in (_baseline(), _baseline(mean_luma=60.0), _baseline(highlight_clip=0.2)):
        assert "tone_map_final" not in _step_names(plan_from_report(report))
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_engine_decision.py` → the three tests fail.
- [ ] **Step 3: Implement** — delete the `# §7.8 Final tone map` block in `decision.py` (the `steps.append(StepSpec(name="tone_map_final", ...))`); in `runner.py` set `_POST_AI = {"unsharp_mask", "clahe_local_contrast"}`. Leave `filmic_tone_map` and the `_apply_classical` branch so a future guard can use it. Update the module docstring line 8 of `decision.py` (`8. Final tone mapping`) to `8. (tone mapping is darktable's job; not planned here)`.
- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_engine_decision.py tests/test_classical_exposure_tone_map.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "fix(engine): drop the second tone map; darktable owns tone mapping"`

### Task 5: TIFF master with ICC profile and copied metadata

**Files:**
- Modify: `app/enhancement/pack_tiff.py` (rewrite)
- Test: `tests/test_pack_tiff.py` (create)

**Interfaces:**
- Produces: `write_tiff16(arr: Array, out: Path, *, icc: bytes | None = None) -> None`; `copy_metadata(source: Path, dest: Path) -> bool` (False when exiftool is missing or fails; logs a warning).

- [ ] **Step 1: Write the failing tests**

```python
"""Master TIFF: 16-bit, ICC embedded when available, EXIF carried over from the RAW."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.enhancement.pack_tiff import copy_metadata, write_tiff16

ICC = Path("tests/fixtures/linear_rec2020.icc").read_bytes()


def test_write_tiff16_embeds_icc_when_given(tmp_path: Path) -> None:
    arr = np.full((3, 3, 3), 0.5, dtype=np.float32)
    write_tiff16(arr, tmp_path / "m.tif", icc=ICC)
    with tifffile.TiffFile(tmp_path / "m.tif") as tf:
        page = tf.pages[0]
        assert page.dtype == np.uint16
        assert bytes(page.tags[34675].value) == ICC


def test_write_tiff16_without_icc_writes_untagged(tmp_path: Path) -> None:
    write_tiff16(np.zeros((2, 2, 3), dtype=np.uint16), tmp_path / "u.tif")
    with tifffile.TiffFile(tmp_path / "u.tif") as tf:
        assert 34675 not in tf.pages[0].tags


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not installed")
def test_copy_metadata_carries_exif_and_forces_upright(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    Image.new("RGB", (8, 8)).save(src)
    subprocess.run(["exiftool", "-overwrite_original", "-Make=Canon", "-Orientation#=6", str(src)], check=True, capture_output=True)
    dest = tmp_path / "m.tif"
    write_tiff16(np.zeros((2, 2, 3), dtype=np.uint16), dest)
    assert copy_metadata(src, dest) is True
    out = subprocess.run(["exiftool", "-s", "-s", "-s", "-Make", "-Orientation#", str(dest)], check=True, capture_output=True, text=True).stdout.split()
    assert out == ["Canon", "1"]
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_pack_tiff.py` → TypeError (`icc` kwarg) / ImportError.

- [ ] **Step 3: Implement** — replace `app/enhancement/pack_tiff.py`:

```python
"""Write the enhanced image as a 16-bit LZW RGB TIFF master, tagged and with the RAW's metadata."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile

from app.arrays import Array

log = logging.getLogger(__name__)
_TIFF_ICC_TAG = 34675


def write_tiff16(arr: Array, out: Path, *, icc: bytes | None = None) -> None:
    """``arr`` may be float in [0, 1], uint8, or uint16; ``icc`` is embedded verbatim when given."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype == np.uint16:
        data = arr
    elif arr.dtype == np.uint8:
        data = (arr.astype(np.uint32) * 257).astype(np.uint16)
    else:
        data = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    extratags = [(_TIFF_ICC_TAG, "B", len(icc), icc, True)] if icc else []
    tifffile.imwrite(out, data, photometric="rgb", compression="lzw", extratags=extratags)


def copy_metadata(source: Path, dest: Path) -> bool:
    """Copy EXIF/XMP/IPTC from ``source`` onto ``dest`` and force Orientation=1 (the
    pixels are already upright). Soft-fail: a master without EXIF is still usable."""
    if shutil.which("exiftool") is None:
        log.warning("exiftool not on PATH; %s written without EXIF", dest.name)
        return False
    try:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-TagsFromFile", str(source), "-EXIF:all", "-XMP:all", "-IPTC:all",
             "--Orientation", "-Orientation#=1", str(dest)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning("metadata copy failed for %s: %s", dest.name, exc.stderr.decode(errors="replace")[:200])
        return False
    return True
```

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_pack_tiff.py` → PASS.
- [ ] **Step 5: Commit** — `git add app/enhancement/pack_tiff.py tests/test_pack_tiff.py && git commit -m "feat(enhance): tagged 16-bit masters with the RAW's metadata"`

### Task 6: Profile-aware TIFF loading (ingest and JPEG export)

**Files:**
- Modify: `app/ingest/decode.py` (`icc_description`, `load_tiff_rgb8`)
- Test: `tests/test_decode.py`

**Interfaces:**
- Produces: `decode.icc_description(icc: bytes) -> str | None`; `decode.LINEAR_REC2020_DESC = "Linear Rec2020 RGB"`; `load_tiff_rgb8` converts linear Rec.2020 masters to display sRGB, treats every other TIFF as sRGB-encoded (today's behaviour).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_decode.py`)

```python
ICC = Path("tests/fixtures/linear_rec2020.icc").read_bytes()


def test_icc_description_parses_darktable_profile() -> None:
    assert decode.icc_description(ICC) == "Linear Rec2020 RGB"
    assert decode.icc_description(b"not an icc") is None


def test_load_tiff_rgb8_converts_linear_rec2020_masters(tmp_path: Path) -> None:
    lin = np.full((4, 4, 3), int(0.18 * 65535), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "lin.tif", lin, photometric="rgb", extratags=[(34675, "B", len(ICC), ICC, True)])
    out = decode.load_tiff_rgb8(tmp_path / "lin.tif")
    assert out.dtype == np.uint8 and 115 <= int(out[0, 0, 1]) <= 120  # encoded, not bit-shifted (46)


def test_load_tiff_rgb8_keeps_untagged_tiffs_as_srgb(tmp_path: Path) -> None:
    srgb = np.full((4, 4, 3), 30000, dtype=np.uint16)
    tifffile.imwrite(tmp_path / "srgb.tif", srgb, photometric="rgb")
    assert int(decode.load_tiff_rgb8(tmp_path / "srgb.tif")[0, 0, 0]) == 30000 >> 8
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_decode.py` → AttributeError `icc_description`.

- [ ] **Step 3: Implement** in `decode.py` (imports: `from app.enhancement.colorspace import linear_rec2020_to_srgb_u8`):

```python
LINEAR_REC2020_DESC = "Linear Rec2020 RGB"
_TIFF_ICC_TAG = 34675


def icc_description(icc: bytes) -> str | None:
    """The ICC 'desc' tag as text (v2 'desc' or v4 'mluc' record); None if absent/malformed."""
    if len(icc) < 132:
        return None
    count = int.from_bytes(icc[128:132], "big")
    for i in range(count):
        off = 132 + 12 * i
        if icc[off : off + 4] != b"desc":
            continue
        start = int.from_bytes(icc[off + 4 : off + 8], "big")
        data = icc[start : start + int.from_bytes(icc[off + 8 : off + 12], "big")]
        if data[:4] == b"desc":
            n = int.from_bytes(data[8:12], "big")
            return data[12 : 12 + n].rstrip(b"\x00").decode("ascii", "replace")
        if data[:4] == b"mluc" and int.from_bytes(data[8:12], "big") > 0:
            n = int.from_bytes(data[20:24], "big")
            o = int.from_bytes(data[24:28], "big")
            return data[o : o + n].decode("utf-16-be", "replace").rstrip("\x00")
    return None


def _tiff_icc(path: Path) -> bytes | None:
    with tifffile.TiffFile(str(path)) as tf:
        tag = tf.pages[0].tags.get(_TIFF_ICC_TAG)
        return bytes(tag.value) if tag is not None else None


def load_tiff_rgb8(path: Path) -> Array:
    """Any TIFF -> HxWx3 uint8 display sRGB. Our linear Rec.2020 masters are colour-converted;
    everything else is assumed sRGB-encoded (grayscale broadcast, alpha dropped, 16-bit >> 8)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    icc = _tiff_icc(path)
    if icc is not None and icc_description(icc) == LINEAR_REC2020_DESC:
        scale = 65535.0 if arr.dtype == np.uint16 else 255.0 if arr.dtype == np.uint8 else 1.0
        return linear_rec2020_to_srgb_u8(arr.astype(np.float32) / scale)
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    if arr.dtype != np.uint8:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr
```

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_decode.py tests/test_jpeg_export.py` → PASS (export goes through `decode_preview`, so share JPEGs from new masters are now correct).
- [ ] **Step 5: Commit** — `git commit -am "feat(decode): colour-convert linear Rec.2020 masters when loading TIFFs"`

### Task 7: Face boxes in TIFF coordinates

**Files:**
- Create: `app/enhancement/geometry.py`
- Modify: `app/enhancement/enhance_job.py` (`PhotoCandidate.preview_path`; scale in `_enhance_one` after load)
- Test: `tests/test_geometry.py` (create), `tests/test_enhance_job.py`

**Interfaces:**
- Produces: `geometry.scale_boxes(boxes: Sequence[tuple[int,int,int,int]], src_size: tuple[int,int], dst_size: tuple[int,int]) -> list[tuple[int,int,int,int]]` (sizes are `(w, h)`; output clamped to `dst_size`); `enhance_job.preview_size(path: Path) -> tuple[int,int] | None`.
- `PhotoCandidate(hash, source_path, file_kind, action, preview_path: str | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_geometry.py
from app.enhancement.geometry import scale_boxes


def test_scale_boxes_maps_preview_to_native_and_clamps() -> None:
    boxes = [(300, 200, 100, 50), (2900, 1950, 200, 100)]
    out = scale_boxes(boxes, src_size=(3000, 2000), dst_size=(6000, 4000))
    assert out[0] == (600, 400, 200, 100)
    assert out[1] == (5800, 3900, 200, 100)  # clamped to the frame


def test_scale_boxes_identity_when_sizes_match() -> None:
    assert scale_boxes([(1, 2, 3, 4)], (10, 10), (10, 10)) == [(1, 2, 3, 4)]
```

Append to `tests/test_enhance_job.py`:

```python
def test_preview_size_missing_file_means_no_faces(tmp_path: Path) -> None:
    from app.enhancement.enhance_job import preview_size

    assert preview_size(tmp_path / "nope.jpg") is None
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_geometry.py tests/test_enhance_job.py` → ImportError.

- [ ] **Step 3: Implement**

```python
# app/enhancement/geometry.py
"""Pixel-geometry helpers shared by the enhancement stages."""

from __future__ import annotations

from collections.abc import Sequence

Box = tuple[int, int, int, int]  # x, y, w, h


def scale_boxes(boxes: Sequence[Box], src_size: tuple[int, int], dst_size: tuple[int, int]) -> list[Box]:
    """Rescale boxes from one image size to another (sizes are (w, h)); clamp to the target."""
    sx = dst_size[0] / max(src_size[0], 1)
    sy = dst_size[1] / max(src_size[1], 1)
    out: list[Box] = []
    for x, y, w, h in boxes:
        nx, ny = int(round(x * sx)), int(round(y * sy))
        nw, nh = int(round(w * sx)), int(round(h * sy))
        nx = min(max(nx, 0), dst_size[0] - 1)
        ny = min(max(ny, 0), dst_size[1] - 1)
        nw = min(nw, dst_size[0] - nx)
        nh = min(nh, dst_size[1] - ny)
        out.append((nx, ny, nw, nh))
    return out
```

In `enhance_job.py`: add `preview_path: str | None = None` to `PhotoCandidate`; select `Photo.preview_path` in `_candidates` and pass it; add

```python
def preview_size(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as im:
        return im.size
```

and in `_enhance_one`, right after `native_h, native_w = rgb_f01.shape[:2]`:

```python
    src_size = preview_size(Path(photo.preview_path) if photo.preview_path else None)
    if src_size is None and face_boxes:
        log.warning("preview missing for %s; ignoring %d face box(es)", src.name, len(face_boxes))
    face_boxes = scale_boxes(face_boxes, src_size, (native_w, native_h)) if src_size else []
```

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_geometry.py tests/test_enhance_job.py` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "fix(enhance): scale face boxes from preview to native coordinates"`

### Task 8: Sidecar resolution by darktable convention

**Files:**
- Create: `app/enhancement/sidecar.py`
- Modify: `app/enhancement/enhance_job.py` (replace `_xmp_for`)
- Test: `tests/test_sidecar.py` (create)

**Interfaces:**
- Produces: `resolve_xmp(source: Path, *, photos_root: Path, xmp_root: Path, baseline: Path | None = None) -> Path | None`. Order: `xmp_root/<rel subdir>/<name>.<ext>.xmp` → legacy `xmp_root/<stem>.xmp` (warns once per process) → `baseline` if it exists → `None`.

- [ ] **Step 1: Write the failing tests**

```python
"""User sidecars follow darktable's <name>.<ext>.xmp convention, mirrored by subfolder."""

from __future__ import annotations

import logging
from pathlib import Path

from app.enhancement.sidecar import resolve_xmp


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    photos = tmp_path / "photos"; xmp = tmp_path / "xmp"
    src = photos / "incoming" / "trip" / "IMG_0001.CR3"
    src.parent.mkdir(parents=True); src.write_bytes(b"")
    return photos, xmp, src


def test_mirrored_darktable_name_wins(tmp_path: Path) -> None:
    photos, xmp, src = _tree(tmp_path)
    want = xmp / "trip" / "IMG_0001.CR3.xmp"; want.parent.mkdir(parents=True); want.write_text("")
    (xmp / "IMG_0001.xmp").write_text("")  # legacy also present
    assert resolve_xmp(src, photos_root=photos, xmp_root=xmp) == want


def test_legacy_stem_lookup_still_works_with_warning(tmp_path: Path, caplog) -> None:
    photos, xmp, src = _tree(tmp_path)
    xmp.mkdir(); legacy = xmp / "IMG_0001.xmp"; legacy.write_text("")
    with caplog.at_level(logging.WARNING):
        assert resolve_xmp(src, photos_root=photos, xmp_root=xmp) == legacy
    assert "legacy" in caplog.text


def test_baseline_used_when_no_user_sidecar(tmp_path: Path) -> None:
    photos, xmp, src = _tree(tmp_path)
    base = tmp_path / "base.xmp"; base.write_text("")
    assert resolve_xmp(src, photos_root=photos, xmp_root=xmp, baseline=base) == base
    assert resolve_xmp(src, photos_root=photos, xmp_root=xmp, baseline=tmp_path / "missing.xmp") is None
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_sidecar.py` → ImportError.

- [ ] **Step 3: Implement**

```python
"""Which darktable sidecar develops a given RAW.

Order: the user's sidecar in darktable's own naming (``IMG.CR3.xmp``) mirrored under
``xmp/<subfolder>/``, then the legacy ``xmp/<stem>.xmp`` lookup, then the shipped
baseline (Step 2), then none (darktable defaults).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.paths import relative_subpath

log = logging.getLogger(__name__)
_warned_legacy = False


def resolve_xmp(source: Path, *, photos_root: Path, xmp_root: Path, baseline: Path | None = None) -> Path | None:
    global _warned_legacy
    rel = relative_subpath(source, photos_root)
    mirrored = xmp_root / rel.parent / f"{source.name}.xmp"
    if mirrored.exists():
        return mirrored
    legacy = xmp_root / f"{source.stem}.xmp"
    if legacy.exists():
        if not _warned_legacy:
            log.warning("using legacy sidecar name %s; prefer %s", legacy.name, mirrored.relative_to(xmp_root))
            _warned_legacy = True
        return legacy
    if baseline is not None and baseline.exists():
        return baseline
    return None
```

In `enhance_job.py` delete `_xmp_for` and call `resolve_xmp(src, photos_root=settings.photos, xmp_root=settings.xmp)`; Task 17 adds `baseline=`.

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_sidecar.py tests/test_enhance_job.py` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): resolve sidecars by darktable naming, mirrored per subfolder"`

### Task 9: Persist the plan (migration 0004, serializer, UI)

**Files:**
- Create: `db/migrations/versions/0004_plan_and_verify.py`
- Modify: `app/models.py` (`PhotoQualityReport` columns), `app/enhancement/engine/plan.py` (`plan_to_json`), `app/enhancement/enhance_job.py` (`persist_plan`), `app/api/serializers.py`, `app/api/static/js/review.js`
- Test: `tests/test_schema.py`, `tests/test_enhance_job.py`, `tests/test_api_serializers.py`

**Interfaces:**
- Columns (all nullable): `plan_json Text`, `verify_json Text`, `score_q_after Float`, `degraded Integer default 0`, and reserved Step-3 metrics `neutral_fraction Float`, `rg_neutral Float`, `bg_neutral Float`, `mean_chroma Float`, `lap_var_top Float`.
- `plan_to_json(plan: EnhancementPlan) -> str`; `enhance_job.persist_plan(sess: Session, photo_hash: str, plan: EnhancementPlan) -> None` (row must exist; created by `persist_report`).
- `quality_report_payload` adds keys `plan` (parsed JSON or None), `verify` (parsed JSON or None), `score_q_after`, `degraded` (bool).

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py` add `"quality_reports"` to `EXPECTED_TABLES` and:

```python
def test_quality_reports_has_plan_and_verify_columns(tmp_db: Session) -> None:
    cols = {c["name"] for c in inspect(tmp_db.bind).get_columns("quality_reports")}
    assert {"plan_json", "verify_json", "score_q_after", "degraded", "neutral_fraction", "lap_var_top"} <= cols
```

`tests/test_enhance_job.py` append:

```python
def test_persist_plan_stores_steps_as_json(tmp_db: Session) -> None:
    import json
    from app.enhancement.enhance_job import persist_plan
    from app.enhancement.engine.decision import plan_from_report

    tmp_db.add(Photo(hash="h2", source_path="/x/b.cr3")); tmp_db.flush()
    report = _report(luma_noise=6.0)
    persist_report(tmp_db, "h2", report)
    persist_plan(tmp_db, "h2", plan_from_report(report))
    tmp_db.flush()
    row = tmp_db.get(PhotoQualityReport, "h2")
    steps = json.loads(row.plan_json)["steps"]
    assert any(s["name"] == "scunet_denoise" for s in steps)
    assert all({"name", "params", "reason"} <= set(s) for s in steps)
```

`tests/test_api_serializers.py` append:

```python
def test_quality_report_payload_exposes_recipe_and_verdict() -> None:
    from app.api.serializers import quality_report_payload
    from app.models import PhotoQualityReport

    qr = PhotoQualityReport(photo_hash="abc123", mean_luma=1, shadow_clip=0, highlight_clip=0, midtone_ratio=0,
        midtone_deviation=0, dr_p95_p5=0, local_dr_mean=0, rg_ratio=1, bg_ratio=1, avg_saturation=0,
        oversat_ratio=0, lap_var=0, edge_density=0, hf_energy=0, luma_noise=0, chroma_noise=0,
        score_exposure=0, score_dynamic_range=0, score_color=0, score_sharpness=0, score_noise=0, score_q=50.0,
        plan_json='{"steps": [{"name": "unsharp_mask", "params": {"amount": 0.4}, "reason": "soft"}]}',
        verify_json='{"degraded": false, "reasons": []}', score_q_after=55.0, degraded=0)
    out = quality_report_payload(qr)
    assert out["plan"]["steps"][0]["name"] == "unsharp_mask"
    assert out["verify"] == {"degraded": False, "reasons": []}
    assert out["score_q_after"] == 55.0 and out["degraded"] is False
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_schema.py tests/test_enhance_job.py tests/test_api_serializers.py` → failures on missing columns / imports.

- [ ] **Step 3: Implement**

Migration `db/migrations/versions/0004_plan_and_verify.py`:

```python
"""quality_reports: recipe, verification, and Step-3 metric columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLS = [
    sa.Column("plan_json", sa.Text(), nullable=True),
    sa.Column("verify_json", sa.Text(), nullable=True),
    sa.Column("score_q_after", sa.Float(), nullable=True),
    sa.Column("degraded", sa.Integer(), server_default="0"),
    sa.Column("neutral_fraction", sa.Float(), nullable=True),
    sa.Column("rg_neutral", sa.Float(), nullable=True),
    sa.Column("bg_neutral", sa.Float(), nullable=True),
    sa.Column("mean_chroma", sa.Float(), nullable=True),
    sa.Column("lap_var_top", sa.Float(), nullable=True),
]


def upgrade() -> None:
    for col in _COLS:
        op.add_column("quality_reports", col)


def downgrade() -> None:
    with op.batch_alter_table("quality_reports") as batch:
        for col in reversed(_COLS):
            batch.drop_column(col.name)
```

`app/models.py` (`PhotoQualityReport`, before `measured_at`):

```python
    # Recipe + verification (Task 9/10) and Step-3 metrics; nullable so old rows load.
    plan_json: Mapped[str | None] = mapped_column(Text)
    verify_json: Mapped[str | None] = mapped_column(Text)
    score_q_after: Mapped[float | None] = mapped_column(Float)
    degraded: Mapped[bool] = mapped_column(Integer, default=0)
    neutral_fraction: Mapped[float | None] = mapped_column(Float)
    rg_neutral: Mapped[float | None] = mapped_column(Float)
    bg_neutral: Mapped[float | None] = mapped_column(Float)
    mean_chroma: Mapped[float | None] = mapped_column(Float)
    lap_var_top: Mapped[float | None] = mapped_column(Float)
```

`engine/plan.py`:

```python
import json

def plan_to_json(plan: EnhancementPlan) -> str:
    return json.dumps({
        "has_faces": plan.has_faces,
        "note": plan.note,
        "steps": [{"name": s.name, "params": dict(s.params), "reason": s.reason} for s in plan.steps],
    })
```

`enhance_job.py`:

```python
def persist_plan(sess: Session, photo_hash: str, plan: EnhancementPlan) -> None:
    row = sess.get(PhotoQualityReport, photo_hash)
    if row is None:
        raise ValueError(f"persist_report must run before persist_plan for {photo_hash}")
    row.plan_json = plan_to_json(plan)
```

Call it in `_enhance_one` right after `plan = plan_from_report(...)` inside a `session_scope()`.

`serializers.py` `quality_report_payload`:

```python
    out["plan"] = json.loads(qr.plan_json) if qr.plan_json else None
    out["verify"] = json.loads(qr.verify_json) if qr.verify_json else None
    out["score_q_after"] = qr.score_q_after
    out["degraded"] = bool(qr.degraded)
```

(and exclude `plan_json`/`verify_json` from the generic field loop by keeping `_REPORT_FIELDS` as the `QualityReport` dataclass fields, which do not include them.)

`review.js` `EngineQualityPanel`, after the `<details>` block:

```javascript
      ${qr.plan && html`
        <details class="mt-2 text-xs text-zinc-500" open>
          <summary class="cursor-pointer text-zinc-400">
            recipe · ${qr.plan.steps.length} step${qr.plan.steps.length === 1 ? "" : "s"}
            ${qr.score_q_after != null ? html` · Q after ${qr.score_q_after.toFixed(0)}` : ""}
            ${qr.degraded ? html`<span class="ml-1 px-1 rounded bg-rose-900 text-rose-200">degraded</span>` : ""}
          </summary>
          <ol class="mt-1 list-decimal list-inside">
            ${qr.plan.steps.map((s) => html`<li key=${s.name}>${s.name}
              <span class="text-zinc-600">${Object.entries(s.params).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(2) : v}`).join(" ")}</span></li>`)}
          </ol>
        </details>`}
```

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_schema.py tests/test_enhance_job.py tests/test_api_serializers.py tests/test_review_routes.py` → PASS. Also `/tmp/rc.sh sh 'alembic upgrade head && alembic downgrade 0003 && alembic upgrade head'` → no errors.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): persist the recipe per photo and show it in the detail panel"`

### Task 10: Verification, safe-plan retry, and the deletion gate

**Files:**
- Create: `app/enhancement/verify.py`
- Modify: `app/enhancement/enhance_job.py` (`_enhance_one` tail)
- Test: `tests/test_verify.py` (create), `tests/test_enhance_job.py`

**Interfaces:**
- Produces: `Verdict(degraded: bool, reasons: tuple[str, ...], q_before: float, q_after: float)` frozen dataclass with `.to_json() -> str`; `verify(before: QualityReport, after: QualityReport, result: Array) -> Verdict`; `safe_plan(plan: EnhancementPlan) -> EnhancementPlan` keeping only `{"exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover", "highlight_rolloff"}` and noting `"safe retry"`.
- `enhance_job.persist_verdict(sess, photo_hash, verdict) -> None` sets `verify_json`, `score_q_after`, `degraded`.
- `enhance_job.may_delete_source(photo: PhotoCandidate, out: Path, verdict: Verdict) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
"""Verification decides whether an enhanced frame is trustworthy enough to replace its RAW."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.enhancement.engine.decision import plan_from_report
from app.enhancement.verify import safe_plan, verify
from tests.test_enhance_job import _report

GOOD = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32) * 0.6 + 0.2


def test_small_quality_drop_is_not_degraded_even_from_a_low_base() -> None:
    v = verify(_report(score_q=20.0), _report(score_q=17.0), GOOD)
    assert v.degraded is False and v.reasons == ()


def test_quality_drop_over_five_is_degraded() -> None:
    v = verify(_report(score_q=80.0), _report(score_q=74.0), GOOD)
    assert v.degraded is True and "q_drop" in v.reasons


def test_new_clipping_is_degraded() -> None:
    v = verify(_report(highlight_clip=0.01), _report(highlight_clip=0.04), GOOD)
    assert "highlight_clip" in v.reasons


def test_collapsed_image_is_degraded_regardless_of_scores() -> None:
    flat = np.full((16, 16, 3), 0.02, dtype=np.float32)
    v = verify(_report(), _report(score_q=99.0), flat)
    assert v.degraded is True and {"mean_luma", "std"} & set(v.reasons)


def test_safe_plan_keeps_only_tone_steps() -> None:
    plan = plan_from_report(_report(luma_noise=6.0, lap_var=50.0, highlight_clip=0.05), has_faces=True)
    safe = safe_plan(plan)
    names = {s.name for s in safe.steps}
    assert names <= {"exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover", "highlight_rolloff"}
    assert "highlight_recover" in names and safe.note.startswith("safe retry")
```

Append to `tests/test_enhance_job.py`:

```python
def test_degraded_result_never_deletes_the_source(tmp_path: Path) -> None:
    from app.enhancement import enhance_job
    from app.enhancement.verify import Verdict

    src = tmp_path / "IMG.CR3"; src.write_bytes(b"raw"); out = tmp_path / "IMG.tif"; out.write_bytes(b"tif")
    photo = enhance_job.PhotoCandidate("h", str(src), "raw", "enhance_only")
    assert enhance_job.may_delete_source(photo, out, Verdict(True, ("q_drop",), 80.0, 70.0)) is False
    assert enhance_job.may_delete_source(photo, out, Verdict(False, (), 80.0, 82.0)) is True
    keep = enhance_job.PhotoCandidate("h", str(src), "raw", "keep_and_enhance")
    assert enhance_job.may_delete_source(keep, out, Verdict(False, (), 80.0, 82.0)) is False
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_verify.py tests/test_enhance_job.py` → ImportError.

- [ ] **Step 3: Implement**

```python
# app/enhancement/verify.py
"""Post-enhancement verification: did the chain make the frame worse?"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from app.arrays import Array
from app.enhancement.engine.plan import EnhancementPlan, QualityReport

Q_DROP_LIMIT = 5.0
SAFE_STEPS = frozenset({"exposure_gamma", "shadow_lift", "highlight_recover", "backlit_recover", "highlight_rolloff"})


@dataclass(frozen=True)
class Verdict:
    degraded: bool
    reasons: tuple[str, ...]
    q_before: float
    q_after: float

    def to_json(self) -> str:
        return json.dumps({"degraded": self.degraded, "reasons": list(self.reasons),
                           "q_before": self.q_before, "q_after": self.q_after})


def verify(before: QualityReport, after: QualityReport, result: Array) -> Verdict:
    reasons: list[str] = []
    if after.score_q < before.score_q - Q_DROP_LIMIT:
        reasons.append("q_drop")
    if after.highlight_clip > 2.0 * before.highlight_clip + 0.01:
        reasons.append("highlight_clip")
    if float(result.mean()) < 0.03:
        reasons.append("mean_luma")
    if float(result.std()) < 0.02:
        reasons.append("std")
    return Verdict(bool(reasons), tuple(reasons), before.score_q, after.score_q)


def safe_plan(plan: EnhancementPlan) -> EnhancementPlan:
    steps = tuple(s for s in plan.steps if s.name in SAFE_STEPS)
    return replace(plan, steps=steps, note=f"safe retry ({len(steps)} tone step(s))")
```

`enhance_job.py`:

```python
def persist_verdict(sess: Session, photo_hash: str, verdict: Verdict) -> None:
    row = sess.get(PhotoQualityReport, photo_hash)
    if row is None:
        raise ValueError(f"no quality report for {photo_hash}")
    row.verify_json = verdict.to_json()
    row.score_q_after = verdict.q_after
    row.degraded = verdict.degraded


def may_delete_source(photo: PhotoCandidate, out: Path, verdict: Verdict) -> bool:
    return photo.action == "enhance_only" and out.exists() and not verdict.degraded
```

Replace the tail of `_enhance_one` from `result_f01 = run_plan(...)` to the end with:

```python
    result_f01 = run_plan(rgb_f01, plan, native_size=(native_w, native_h))
    after = score_report(measure_all(result_f01, face_boxes=face_boxes or None))
    verdict = verify(report, after, result_f01)
    if verdict.degraded:
        log.warning("%s degraded (%s); retrying with the safe plan", src.name, ",".join(verdict.reasons))
        retry = safe_plan(plan)
        retry_f01 = run_plan(rgb_f01, retry, native_size=(native_w, native_h))
        retry_after = score_report(measure_all(retry_f01, face_boxes=face_boxes or None))
        retry_verdict = verify(report, retry_after, retry_f01)
        if retry_after.score_q >= after.score_q:
            result_f01, verdict, plan = retry_f01, retry_verdict, retry
    with session_scope() as sess:
        persist_plan(sess, photo.hash, plan)
        persist_verdict(sess, photo.hash, verdict)

    out = settings.photos / "exported" / relative_subpath(src, settings.photos).with_suffix(".tif")
    write_tiff16(result_f01, out, icc=icc)
    copy_metadata(src, out)
    with contextlib.suppress(OSError):
        full_tiff.unlink()

    if may_delete_source(photo, out, verdict):
        try:
            src.unlink()
            log.info("deleted no-RAW source after verified enhance: %s", src)
        except OSError as exc:
            log.warning("failed to delete no-RAW source %s: %s", src, exc)
    elif photo.action == "enhance_only" and verdict.degraded:
        log.error("KEEPING source RAW for %s: result degraded (%s). Inspect %s.", src.name, ",".join(verdict.reasons), out)
    return out
```

where `icc = read_icc_profile(full_tiff)` is captured right after `darktable_cli(...)`; if it is `None`, `log.warning("developed TIFF for %s has no ICC profile; master will be untagged", src.name)`.

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_verify.py tests/test_enhance_job.py` → PASS; then the full suite.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): verify results, retry a safe plan, and gate RAW deletion on the verdict"`

### Task 11: Load-once model classes

**Files:**
- Modify: `app/enhancement/denoise.py`, `app/enhancement/upscale.py`, `app/enhancement/face_restore.py`
- Test: `tests/test_ai_models.py` (create; CPU-only paths)

**Interfaces:**
- Produces three context managers with the same shape: `ScunetModel()`, `RealEsrganModel()`, `CodeFormerModel()`; each has `available: bool` (weights + imports present), `apply(rgb_u8: Array, **params) -> Array` (returns the input unchanged when not available), and `close()`; `__enter__`/`__exit__` load/unload with `torch.cuda.empty_cache()` on exit.
- The existing functions `scunet_denoise(rgb, strength)`, `realesrgan_x2(rgb, ..., fidelity)`, `codeformer_restore(rgb, weight)` become one-shot adapters: `with ScunetModel() as m: return m.apply(rgb, strength=strength)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Model wrappers load once and degrade to identity when weights are absent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.enhancement.denoise import ScunetModel
from app.enhancement.face_restore import CodeFormerModel
from app.enhancement.upscale import RealEsrganModel

RGB = np.random.default_rng(0).integers(0, 255, (32, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("cls", [ScunetModel, CodeFormerModel])
def test_missing_weights_means_identity(cls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with cls() as m:
        assert m.available is False
        assert m.apply(RGB, strength=0.8, weight=0.8) is RGB


def test_realesrgan_without_weights_falls_back_to_lanczos_x2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with RealEsrganModel() as m:
        out = m.apply(RGB, fidelity=0.7)
    assert out.shape == (64, 96, 3)


def test_zero_strength_short_circuits_without_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "models", tmp_path)
    with ScunetModel() as m:
        assert m.apply(RGB, strength=0.0) is RGB
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_ai_models.py` → ImportError.

- [ ] **Step 3: Implement** — pattern for `denoise.py` (the other two follow it; keep each file's existing tiling / blending code inside `apply`):

```python
class ScunetModel:
    """SCUNet loaded once for a batch. ``apply`` blends by ``strength`` like scunet_denoise did."""

    def __init__(self) -> None:
        self.available = False
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._dtype: Any = None

    def __enter__(self) -> ScunetModel:
        weights = scunet_weights()
        if not weights.exists():
            log.warning("scunet weights missing at %s — denoise disabled", weights)
            return self
        try:
            import torch
            from app.enhancement._scunet_arch import SCUNet
        except ImportError as exc:
            log.warning("scunet imports failed: %s — denoise disabled", exc)
            return self
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(  # type: ignore[no-untyped-call]
            self._device, dtype=self._dtype
        )
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("params") or ckpt.get("params_ema") or ckpt)
        model.eval()
        self._model = model
        self.available = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self.available = False

    def apply(self, rgb: Array, *, strength: float = 1.0, **_: object) -> Array:
        if strength <= 0.0 or not self.available:
            return rgb
        strength = float(min(1.0, strength))
        torch = self._torch
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=self._dtype)
        y = _tiled_forward(self._model, x)
        denoised = y.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0
        del x, y
        if strength >= 1.0:
            return denoised.astype(np.uint8)
        return np.clip(denoised * strength + rgb.astype(np.float32) * (1.0 - strength), 0.0, 255.0).astype(np.uint8)


def scunet_denoise(rgb: Array, strength: float = 1.0) -> Array:
    with ScunetModel() as m:
        return m.apply(rgb, strength=strength)
```

`RealEsrganModel.apply(rgb, *, fidelity=1.0, **_)`: when `not self.available` return `_lanczos_x2(rgb)`; otherwise the existing blend. `CodeFormerModel.apply(rgb, *, weight=0.7, **_)`: when not available return `rgb`; otherwise the existing body with `self._net`/`self._helper` built in `__enter__` (the `FaceRestoreHelper` is re-created per call because it holds per-image state; the `CodeFormer` net is loaded once).

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_ai_models.py tests/test_enhancement_weights.py` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "refactor(enhance): load-once model classes for SCUNet, Real-ESRGAN and CodeFormer"`

### Task 12: Step-major batch execution

**Files:**
- Create: `app/enhancement/batch.py`
- Modify: `app/enhancement/engine/runner.py` (split `run_plan`), `app/enhancement/enhance_job.py` (delegate)
- Test: `tests/test_batch.py` (create)

**Interfaces:**
- `runner.apply_pre_ai(img: Array, plan: EnhancementPlan) -> Array`; `runner.apply_post_ai(img: Array, plan: EnhancementPlan) -> Array`; `runner.ai_steps(plan) -> list[StepSpec]`; `run_plan` stays and is composed of these plus the AI loop (used by the safe-plan retry, which has no AI steps).
- `batch.AI_MODELS: dict[str, Callable[[], ModelLike]]` = `{"scunet_denoise": ScunetModel, "realesrgan_upscale": RealEsrganModel, "codeformer_restore": CodeFormerModel}`; `ModelLike` is a `Protocol` with `__enter__/__exit__/apply`.
- `batch.run_batch(items: list[tuple[PhotoCandidate, list[FaceBox]]], *, develop=darktable_cli) -> EnhanceSummary`. Intermediates: `settings.cache / "enhance" / f"{hash}.npy"` float16 linear; removed on success and on failure.
- `enhance_job.run_enhancement()` → `warmup(); return run_batch(_candidates())`.

- [ ] **Step 1: Write the failing tests**

```python
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
    return EnhancementPlan(steps=tuple(StepSpec(n) for n in names), report=_report(), has_faces=False)


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
    (tmp_path / "photos" / "incoming").mkdir(parents=True)
    FakeModel.loads = 0; FakeModel.applies = []
    monkeypatch.setattr(batch, "AI_MODELS", {"scunet_denoise": lambda: FakeModel("scunet"), "realesrgan_upscale": lambda: FakeModel("esrgan"), "codeformer_restore": lambda: FakeModel("cf")})
    monkeypatch.setattr(batch, "persist_all", lambda *a, **k: None)  # DB writes are covered elsewhere

    def fake_develop(raw: Path, xmp=None, out_path=None) -> Path:
        if raw.name.startswith("BAD"):
            raise RuntimeError("darktable-cli failed")
        out = tmp_path / f"{raw.stem}.dev.tif"
        noisy = (np.random.default_rng(1).random((64, 96, 3)) * 0.3 + 0.2)
        tifffile.imwrite(out, (noisy * 65535).astype(np.uint16), photometric="rgb", extratags=[(34675, "B", len(ICC), ICC, True)])
        return out

    def candidate(name: str) -> tuple[PhotoCandidate, list]:
        p = tmp_path / "photos" / "incoming" / name; p.write_bytes(b"raw")
        return PhotoCandidate(name, str(p), "raw", "keep_and_enhance"), []

    return tmp_path, fake_develop, candidate


def test_models_load_once_for_the_whole_batch(env, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path, fake_develop, candidate = env
    monkeypatch.setattr(batch, "plan_for", lambda report, **kw: _forced_plan(("scunet_denoise", "realesrgan_upscale")))
    summary = batch.run_batch([candidate("A.CR3"), candidate("B.CR3"), candidate("C.CR3")], develop=fake_develop)
    assert (summary.enhanced, summary.failed) == (3, 0)
    assert FakeModel.loads == 2  # scunet once, esrgan once, never codeformer
    assert FakeModel.applies.count("scunet") == 3
    assert not list((tmp_path / "cache" / "enhance").glob("*.npy"))
    assert sorted(p.name for p in (tmp_path / "photos" / "exported").glob("*.tif")) == ["A.tif", "B.tif", "C.tif"]


def test_one_failing_develop_does_not_stop_the_batch(env) -> None:
    tmp_path, fake_develop, candidate = env
    summary = batch.run_batch([candidate("BAD.CR3"), candidate("OK.CR3")], develop=fake_develop)
    assert (summary.enhanced, summary.failed) == (1, 1)
    assert (tmp_path / "photos" / "exported" / "OK.tif").exists()
    assert not list((tmp_path / "cache" / "enhance").glob("*.npy"))
```

- [ ] **Step 2: Run to verify failure** — `/tmp/rc.sh test tests/test_batch.py` → ImportError.

- [ ] **Step 3: Implement**

`runner.py` split (keep `_apply_classical`, `_apply_ai`, sets):

```python
def apply_pre_ai(img: Array, plan: EnhancementPlan) -> Array:
    for step in plan.steps:
        if step.name in _PRE_AI:
            log.info("engine[pre-AI]  %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)
    return img


def apply_post_ai(img: Array, plan: EnhancementPlan) -> Array:
    for step in plan.steps:
        if step.name in _POST_AI:
            log.info("engine[post-AI] %s %s -- %s", step.name, step.params, step.reason)
            img = _apply_classical(step.name, img, step.params)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def ai_steps(plan: EnhancementPlan) -> list[StepSpec]:
    return [s for s in plan.steps if s.name in _AI]
```

and `run_plan` becomes: `img = apply_pre_ai(...)`; the existing AI loop; `return apply_post_ai(img, plan)`.

`batch.py`:

```python
"""Step-major enhancement of a whole batch: develop+plan+pre-AI for every photo, then each AI
model once over the photos that need it, then post-AI+verify+write. Intermediates are float16
linear Rec.2020 .npy files under cache/enhance/."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rich.console import Console
from rich.progress import Progress

from app.arrays import Array
from app.config import settings
from app.db import session_scope
from app.enhancement.denoise import ScunetModel
from app.enhancement.develop_full import darktable_cli, read_icc_profile
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from app.enhancement.engine.plan import EnhancementPlan, QualityReport, StepSpec
from app.enhancement.engine.runner import _from_u8, _to_u8, ai_steps, apply_post_ai, apply_pre_ai, run_plan
from app.enhancement.enhance_job import (EnhanceSummary, FaceBox, PhotoCandidate, _load_linear_float,
                                         may_delete_source, persist_plan, persist_report, persist_verdict, preview_size)
from app.enhancement.face_restore import CodeFormerModel
from app.enhancement.geometry import scale_boxes
from app.enhancement.pack_tiff import copy_metadata, write_tiff16
from app.enhancement.sidecar import resolve_xmp
from app.enhancement.upsample_final import upsample_final
from app.enhancement.upscale import RealEsrganModel
from app.enhancement.verify import Verdict, safe_plan, verify
from app.paths import relative_subpath

log = logging.getLogger(__name__)
console = Console()

AI_ORDER: tuple[str, ...] = ("scunet_denoise", "realesrgan_upscale", "codeformer_restore")


class ModelLike(Protocol):
    def __enter__(self) -> ModelLike: ...
    def __exit__(self, *exc: object) -> None: ...
    def apply(self, rgb: Array, **params: Any) -> Array: ...


AI_MODELS: dict[str, Callable[[], ModelLike]] = {
    "scunet_denoise": ScunetModel, "realesrgan_upscale": RealEsrganModel, "codeformer_restore": CodeFormerModel,
}


@dataclass
class WorkItem:
    photo: PhotoCandidate
    face_boxes: list[FaceBox]
    intermediate: Path
    report: QualityReport | None = None
    plan: EnhancementPlan | None = None
    native_size: tuple[int, int] = (0, 0)
    icc: bytes | None = None
    ai_pending: list[StepSpec] = field(default_factory=list)
    failed: str | None = None
    out: Path | None = None


def plan_for(report: QualityReport, **kw: Any) -> EnhancementPlan:
    return plan_from_report(
        report, denoise=settings.enhance_denoise, face_restore=settings.enhance_face_restore,
        backlit_recovery=settings.enhance_backlit_recovery, enhance_codeformer_w=settings.enhance_codeformer_w,
        enhance_realesrgan_fidelity=settings.enhance_realesrgan_fidelity,
        enhance_denoise_strength=settings.enhance_denoise_strength,
        backlit_shadow_lift=settings.enhance_backlit_shadow_lift,
        backlit_highlight_protect=settings.enhance_backlit_highlight_protect, **kw,
    )


def persist_all(item: WorkItem, verdict: Verdict | None = None) -> None:
    with session_scope() as sess:
        if item.report is not None:
            persist_report(sess, item.photo.hash, item.report)
        if item.plan is not None:
            persist_plan(sess, item.photo.hash, item.plan)
        if verdict is not None:
            persist_verdict(sess, item.photo.hash, verdict)


def _phase1(item: WorkItem, develop: Callable[..., Path]) -> None:
    src = Path(item.photo.source_path)
    xmp = resolve_xmp(src, photos_root=settings.photos, xmp_root=settings.xmp)
    dev = develop(src, xmp)
    try:
        item.icc = read_icc_profile(dev)
        if item.icc is None:
            log.warning("developed TIFF for %s has no ICC profile; master will be untagged", src.name)
        img = _load_linear_float(dev)
    finally:
        with contextlib.suppress(OSError):
            dev.unlink()
    h, w = img.shape[:2]
    item.native_size = (w, h)
    size = preview_size(Path(item.photo.preview_path) if item.photo.preview_path else None)
    item.face_boxes = scale_boxes(item.face_boxes, size, (w, h)) if size else []
    item.report = score_report(measure_all(img, face_boxes=item.face_boxes or None))
    item.plan = plan_for(item.report, has_faces=bool(item.face_boxes))
    item.ai_pending = ai_steps(item.plan)
    persist_all(item)
    img = apply_pre_ai(img, item.plan)
    item.intermediate.parent.mkdir(parents=True, exist_ok=True)
    np.save(item.intermediate, img.astype(np.float16))


def _phase2(items: list[WorkItem]) -> None:
    for name in AI_ORDER:
        todo = [it for it in items if it.failed is None and any(s.name == name for s in it.ai_pending)]
        if not todo:
            continue
        console.print(f"[cyan]AI step {name}: {len(todo)} photo(s)[/cyan]")
        with AI_MODELS[name]() as model:
            for it in todo:
                try:
                    step = next(s for s in it.ai_pending if s.name == name)
                    img = np.load(it.intermediate).astype(np.float32)
                    u8 = _to_u8(img)
                    if settings.enhance_ai_scale < 0.999 and name == AI_ORDER[0]:
                        from app.enhancement.downsample import scale as lanczos_scale
                        u8 = lanczos_scale(u8, settings.enhance_ai_scale)
                    u8 = model.apply(u8, **step.params)
                    np.save(it.intermediate, _from_u8(u8).astype(np.float16))
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    it.failed = f"{name}: {exc}"
                    log.exception("AI step %s failed for %s", name, it.photo.source_path)


def _phase3(item: WorkItem) -> None:
    assert item.plan is not None and item.report is not None
    src = Path(item.photo.source_path)
    img = np.load(item.intermediate).astype(np.float32)
    if item.ai_pending:
        img = _from_u8(upsample_final(_to_u8(img), item.native_size))
    result = apply_post_ai(img, item.plan)
    after = score_report(measure_all(result, face_boxes=item.face_boxes or None))
    verdict = verify(item.report, after, result)
    plan = item.plan
    if verdict.degraded:
        log.warning("%s degraded (%s); retrying with the safe plan", src.name, ",".join(verdict.reasons))
        # The safe plan has no AI steps, so it can run from the original development again.
        dev = darktable_cli(src, resolve_xmp(src, photos_root=settings.photos, xmp_root=settings.xmp))
        try:
            base = _load_linear_float(dev)
        finally:
            with contextlib.suppress(OSError):
                dev.unlink()
        retry = safe_plan(plan)
        retry_img = run_plan(base, retry, native_size=item.native_size)
        retry_after = score_report(measure_all(retry_img, face_boxes=item.face_boxes or None))
        retry_verdict = verify(item.report, retry_after, retry_img)
        if retry_after.score_q >= after.score_q:
            result, verdict, plan = retry_img, retry_verdict, retry
    item.plan = plan
    persist_all(item, verdict)
    out = settings.photos / "exported" / relative_subpath(src, settings.photos).with_suffix(".tif")
    write_tiff16(result, out, icc=item.icc)
    copy_metadata(src, out)
    item.out = out
    if may_delete_source(item.photo, out, verdict):
        with contextlib.suppress(OSError):
            src.unlink()
            log.info("deleted no-RAW source after verified enhance: %s", src)
    elif item.photo.action == "enhance_only" and verdict.degraded:
        log.error("KEEPING source RAW for %s: result degraded (%s)", src.name, ",".join(verdict.reasons))


def run_batch(items_in: list[tuple[PhotoCandidate, list[FaceBox]]], *, develop: Callable[..., Path] = darktable_cli) -> EnhanceSummary:
    if not items_in:
        console.print("[yellow]No photos to enhance.[/yellow]")
        return EnhanceSummary()
    work_dir = settings.cache / "enhance"
    items = [WorkItem(p, list(f), work_dir / f"{p.hash}.npy") for p, f in items_in]
    skipped = 0
    with Progress() as progress:
        task = progress.add_task("enhance: develop + plan", total=len(items))
        for it in items:
            src = Path(it.photo.source_path)
            if not src.exists() or (it.photo.file_kind not in (None, "raw")):
                it.failed = "skipped"; skipped += 1
            else:
                try:
                    _phase1(it, develop)
                except Exception as exc:  # noqa: BLE001
                    it.failed = f"develop/plan: {exc}"
                    log.exception("enhance phase 1 failed for %s", src)
            progress.advance(task)
    _phase2(items)
    enhanced = 0
    with Progress() as progress:
        task = progress.add_task("enhance: finish + write", total=len(items))
        for it in items:
            if it.failed is None:
                try:
                    _phase3(it); enhanced += 1
                    console.print(f"  -> {it.out}")
                except Exception as exc:  # noqa: BLE001
                    it.failed = f"finish: {exc}"
                    log.exception("enhance phase 3 failed for %s", it.photo.source_path)
            with contextlib.suppress(OSError):
                it.intermediate.unlink()
            progress.advance(task)
    failed = sum(1 for it in items if it.failed not in (None, "skipped"))
    colour = "red" if failed else "green"
    console.print(f"[{colour}]Enhancement complete:[/{colour}] enhanced={enhanced} skipped={skipped} failed={failed}")
    return EnhanceSummary(enhanced=enhanced, skipped=skipped, failed=failed)
```

`enhance_job.run_enhancement`:

```python
def run_enhancement() -> EnhanceSummary:
    from app.enhancement.batch import run_batch  # local: batch imports this module

    warmup()
    return run_batch(_candidates())
```

Delete `_enhance_one` (its logic now lives in the three phases) and update `tests/test_enhance_job.py::test_run_enhancement_keeps_going_after_one_photo_fails` to monkeypatch `app.enhancement.batch.run_batch` instead of `_enhance_one`, asserting it receives the candidates.

- [ ] **Step 4: Run** — `/tmp/rc.sh test tests/test_batch.py tests/test_enhance_job.py` → PASS, then the full suite and `mypy`.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): step-major batch execution with one model load per step"`

### Task 13: Corpus harness and contact sheets

**Files:**
- Create: `tests/corpus_expectations.py`, `tests/test_engine_corpus.py`, `scripts/contact_sheet.py`
- Modify: `CLAUDE.md` (Conventions: how to run the harness and review sheets)

**Interfaces:**
- `corpus_expectations.EXPECT: dict[str, dict[str, object]]` keyed by stem, e.g. `{"IMG_0958": {"absent": {"white_balance"}}, "IMG_0037": {"absent": {"scunet_denoise"}}, "IMG_1176": {"clahe_max": 2.0}, "IMG_0030": {"present": {"highlight_recover"}}}`. Step 1 ships only the entries that hold before Step 3 (`IMG_0030 present highlight_recover`, and `absent tone_map_final` for all); Tasks 18, 23, 24 add theirs.
- `scripts/contact_sheet.py --out tests/data/_review/sheets/<tag>` renders, per corpus frame, darktable default vs enhanced (full frame + 100% centre crop) from `photos/exported/`.

- [ ] **Step 1: Write the harness**

```python
# tests/test_engine_corpus.py
"""Plan-level expectations over the real corpus (skipped unless tests/data/corpus has RAWs)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from app.enhancement.develop_full import build_darktable_command
from app.enhancement.engine import measure_all, score_report
from app.enhancement.engine.decision import plan_from_report
from tests.corpus_expectations import EXPECT

CORPUS = Path("tests/data/corpus")
pytestmark = pytest.mark.real_raw


def _develop_linear(raw: Path) -> np.ndarray:
    out = Path(tempfile.mkdtemp()) / "dev.tif"
    subprocess.run(build_darktable_command(raw, out), check=True, capture_output=True)
    return tifffile.imread(str(out))[..., :3].astype(np.float32) / 65535.0


@pytest.mark.parametrize("stem", sorted(EXPECT))
def test_plan_matches_expectation(stem: str) -> None:
    raw = CORPUS / f"{stem}.CR3"
    if not raw.exists():
        pytest.skip(f"{raw} not present")
    plan = plan_from_report(score_report(measure_all(_develop_linear(raw))))
    names = {s.name for s in plan.steps}
    exp = EXPECT[stem]
    assert set(exp.get("present", ())) <= names, f"missing {set(exp.get('present', ())) - names}"
    assert not (set(exp.get("absent", ())) & names), f"unexpected {set(exp.get('absent', ())) & names}"
    if "clahe_max" in exp:
        clips = [s.params["clip_limit"] for s in plan.steps if s.name == "clahe_local_contrast"]
        assert all(c <= exp["clahe_max"] for c in clips)
```

```python
# tests/corpus_expectations.py
"""What the planner must decide for specific corpus frames. Extend as rules land."""

EXPECT: dict[str, dict[str, object]] = {
    "IMG_0030": {"present": {"highlight_recover"}, "absent": {"tone_map_final"}},  # backlit sunset
    "IMG_1177": {"absent": {"tone_map_final"}},  # bright monument
}
```

`scripts/contact_sheet.py`:

```python
"""Before/after contact sheets for owner review: darktable default vs enhanced master."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import settings
from app.ingest.decode import load_tiff_rgb8

PREVIEWS = Path("tests/data/_review/previews")  # <stem>_dt_default.jpg from analyze_plans.py


def _crop_center(im: Image.Image, size: int = 600) -> Image.Image:
    w, h = im.size
    return im.crop(((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    for tif in sorted((settings.photos / "exported").rglob("*.tif")):
        before_path = PREVIEWS / f"{tif.stem}_dt_default.jpg"
        if not before_path.exists():
            continue
        after_full = Image.fromarray(load_tiff_rgb8(tif))
        before_full = Image.open(before_path).convert("RGB").resize(after_full.size)
        tiles = [before_full.copy(), after_full.copy(), _crop_center(before_full), _crop_center(after_full)]
        for t in tiles[:2]:
            t.thumbnail((900, 900))
        sheet = Image.new("RGB", (1800, 1500), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * 900, (i // 2) * 900 if i < 2 else 900))
        d = ImageDraw.Draw(sheet); d.text((8, 8), f"{tif.stem}  before | after  (bottom: 100% centre crops)", fill=(255, 255, 0))
        sheet.save(out_dir / f"{tif.stem}.jpg", quality=85)
        print(out_dir / f"{tif.stem}.jpg")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness** — `/tmp/rc.sh test tests/test_engine_corpus.py -v` → the two entries PASS (they develop two frames, ~10 s).
- [ ] **Step 3: Produce the Step 1 sheet** — run enhance on the corpus and render sheets; this needs the GPU and takes a while:

```bash
# one-off: point a scratch photos/ at the corpus and run leg-1 + enhance in the container
/tmp/rc.sh sh 'rm -rf /tmp/rc-photos && mkdir -p /tmp/rc-photos/incoming && cp tests/data/corpus/*.CR3 /tmp/rc-photos/incoming/ && alembic upgrade head && raw-curator ingest && raw-curator score --stage faces && python -c "
from app.db import session_scope; from app.decision.bulk import stage_all
with session_scope() as s: stage_all(s, \"yes\")" && raw-curator submit && raw-curator enhance && python scripts/contact_sheet.py --out tests/data/_review/sheets/step1'
```

(`rc.sh` sets `RAWCURATOR_PHOTOS=/tmp/rc-photos`; for the GPU add `--device nvidia.com/gpu=all` to the `podman run` in `/tmp/rc.sh` or run the same commands through `podman-compose run --rm app`.) Then open `tests/data/_review/sheets/step1/*.jpg` and hand them to the owner.

- [ ] **Step 4: Document** — add to `CLAUDE.md` Conventions: "`tests/test_engine_corpus.py` asserts planner decisions on real frames in `tests/data/corpus/` (`real_raw`); look-changing changes ship with a sheet from `scripts/contact_sheet.py` reviewed by the owner."
- [ ] **Step 5: Commit** — `git add -A && git commit -m "test(enhance): corpus expectations harness and before/after contact sheets"`

**Gate:** owner reviews `sheets/step1`. Step 1 PR opens after that.

---

## Step 2 — baseline development

### Task 14: Workflow default → sigmoid

**Files:**
- Modify: `app/config.py` (`darktable_workflow` default), `.env.example`, `README.md` config table
- Test: `tests/test_develop_full.py`

- [ ] **Step 1: Write the failing test**

```python
def test_default_workflow_is_sigmoid(monkeypatch) -> None:
    from app.config import Settings

    assert Settings(_env_file=None).darktable_workflow == "scene-referred (sigmoid)"
```

- [ ] **Step 2: Run to verify failure** — expects `filmic`.
- [ ] **Step 3: Implement** — change the default to `"scene-referred (sigmoid)"`; add `#RAWCURATOR_DARKTABLE_WORKFLOW=scene-referred (sigmoid)` with a one-line comment to `.env.example`; README row: `RAWCURATOR_DARKTABLE_WORKFLOW | scene-referred (sigmoid) | darktable pixel workflow for develop; filmic is the 4.6 default look`.
- [ ] **Step 4: Run** — PASS. Regenerate the darktable-default previews used by the sheets: rerun `tests/data/_review/analyze_plans.py` (it writes `previews/*_dt_default.jpg`).
- [ ] **Step 5: Commit** — `git commit -am "feat(develop): sigmoid workflow by default"`

### Task 15: Author the baseline sidecar (owner task, GUI)

**Files:**
- Create: `app/enhancement/darktable/raw-curator-base.xmp` (checked in), `app/enhancement/darktable/README.md`

Nothing here can be unit-tested until the file exists; the acceptance test is Task 16.

- [ ] **Step 1: Launch darktable from the image with the corpus mounted** (host X socket; run as the owner):

```bash
xhost +local: && podman run --rm -it --entrypoint darktable \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/rong/projects/raw-curator/tests/data/corpus:/corpus:z \
  -v /home/rong/projects/raw-curator/app/enhancement/darktable:/styles:z \
  localhost/raw-curator:latest /corpus/IMG_1177.CR3 --conf plugins/darkroom/workflow="scene-referred (sigmoid)"
```

- [ ] **Step 2: In the darkroom, on IMG_1177 (bright, ISO 100) set the baseline and nothing image-specific:**
  - lens correction: correction method *lensfun*, camera/lens *auto*, corrections *all*.
  - chromatic aberrations (raw): enabled, defaults.
  - highlight reconstruction: method *inpaint opposed*.
  - denoise (profiled): enabled, mode *wavelets auto*, profile *auto* (the ISO-dependent profile), default strength.
  - color calibration: adaptation *CAT16*, illuminant *as shot in camera*.
  - exposure: leave the scene-referred default (+0.7 EV with compensation), no manual change.
  - sigmoid: contrast 1.5, skew 0, per-channel preservation *per channel*.
  - Do **not** touch crop, rotate, local edits, or white balance sliders.
- [ ] **Step 3: Save as a style** named `raw-curator-base` (styles module → create), then in lighttable select IMG_1177 → *history stack → write sidecar files*. Copy the resulting `/corpus/IMG_1177.CR3.xmp` to `/styles/raw-curator-base.xmp` and also *export* the style to `/styles/raw-curator-base.dtstyle` (kept for humans; the pipeline uses the XMP).
- [ ] **Step 4: Strip image-specific fields** from the XMP: remove `darktable:history_end` value? No — keep it; remove `xmpMM:DerivedFrom`, `darktable:raw_params`, `darktable:auto_presets_applied` stays. Verify it applies to a *different* frame: `darktable-cli /corpus/IMG_0098.CR3 /styles/raw-curator-base.xmp /tmp/t.tif --icc-type LIN_REC2020 --core` exits 0 and the output is noticeably cleaner in shadows than without the XMP.
- [ ] **Step 5: Commit** — `git add app/enhancement/darktable && git commit -m "feat(develop): checked-in darktable baseline sidecar (lens, CA, highlights, profiled denoise, sigmoid)"` with `README.md` in that folder listing the module settings above and the re-authoring procedure.

### Task 16: Apply the baseline when no user sidecar exists

**Files:**
- Modify: `app/enhancement/sidecar.py` (`BASELINE_XMP` constant), `app/enhancement/batch.py` (pass `baseline=`), `app/config.py` (`darktable_baseline: bool = True`)
- Test: `tests/test_sidecar.py`, `tests/test_engine_corpus.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_default_baseline_path_points_at_the_shipped_sidecar() -> None:
    from app.enhancement.sidecar import BASELINE_XMP
    assert BASELINE_XMP.name == "raw-curator-base.xmp" and BASELINE_XMP.exists()
```

and in `tests/test_engine_corpus.py`:

```python
@pytest.mark.parametrize("stem", ["IMG_0098", "IMG_1124", "IMG_1177"])
def test_baseline_develops_every_kind_of_frame(stem: str, tmp_path: Path) -> None:
    from app.enhancement.sidecar import BASELINE_XMP
    raw = CORPUS / f"{stem}.CR3"
    if not raw.exists():
        pytest.skip()
    out = tmp_path / "b.tif"
    subprocess.run(build_darktable_command(raw, out, BASELINE_XMP), check=True, capture_output=True)
    arr = tifffile.imread(str(out))
    assert arr.dtype == np.uint16 and arr.shape[2] == 3 and max(arr.shape[:2]) >= 5900
```

- [ ] **Step 2: Run to verify failure** — ImportError on `BASELINE_XMP`.
- [ ] **Step 3: Implement** — `sidecar.py`: `BASELINE_XMP = Path(__file__).resolve().parent / "darktable" / "raw-curator-base.xmp"`; `batch._phase1` and the retry path call `resolve_xmp(..., baseline=BASELINE_XMP if settings.darktable_baseline else None)`; `Settings.darktable_baseline: bool = True` documented in `.env.example`/README ("set false to develop with darktable defaults only").
- [ ] **Step 4: Run** — sidecar + corpus tests PASS. Render `sheets/step2` as in Task 13 Step 3.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(develop): apply the shipped baseline sidecar unless the user provides one"`

### Task 17: ISO-aware denoise trigger

**Files:**
- Modify: `app/enhancement/engine/decision.py` (`iso: int | None = None` keyword), `app/enhancement/enhance_job.py` (`PhotoCandidate.iso`, selected from `Photo.iso`), `app/enhancement/batch.py` (`plan_for(..., iso=item.photo.iso)`)
- Test: `tests/test_engine_decision.py`, `tests/corpus_expectations.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_low_iso_mild_noise_does_not_trigger_scunet() -> None:
    plan = plan_from_report(_baseline(luma_noise=3.0), iso=100)
    assert "scunet_denoise" not in _step_names(plan)


def test_high_iso_mild_noise_triggers_scunet() -> None:
    plan = plan_from_report(_baseline(luma_noise=3.0), iso=3200)
    assert "scunet_denoise" in _step_names(plan)


def test_strong_noise_triggers_regardless_of_iso() -> None:
    assert "scunet_denoise" in _step_names(plan_from_report(_baseline(luma_noise=5.0), iso=100))
    assert "scunet_denoise" in _step_names(plan_from_report(_baseline(luma_noise=5.0), iso=None))
```

Add to `EXPECT`: `"IMG_0037": {"absent": {"scunet_denoise"}}` (ISO 100), `"IMG_0098": {"present": {"scunet_denoise"}}` (ISO 12800). The corpus test must then pass `iso` from EXIF: extend `_develop_linear` callers to read ISO with `exiftool -s3 -ISO`.

- [ ] **Step 2: Run to verify failure** — TypeError on `iso=`.
- [ ] **Step 3: Implement** — in `plan_from_report` signature add `iso: int | None = None,` after `backlit_recovery`; replace the noise rule with:

```python
    n = max(report.luma_noise, report.chroma_noise * 0.5)
    noisy = n > 4.0 or (n > 2.0 and (iso is None or iso >= 800))
    if denoise and noisy:
```

`PhotoCandidate` gains `iso: int | None = None`; `_candidates` selects `Photo.iso`.

- [ ] **Step 4: Run** — decision + corpus tests PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(engine): ISO-aware denoise trigger"`

**Gate:** owner reviews `sheets/step2`. Step 2 PR opens after that.

---

## Step 3 — the detail path

### Task 18: AI as a delta merged into the float image

**Files:**
- Modify: `app/enhancement/engine/runner.py` (`apply_ai_delta`), `app/enhancement/batch.py` (`_phase2`, `_phase3` use it)
- Test: `tests/test_engine_runner_boundary.py`

**Interfaces:**
- `runner.apply_ai_delta(x_lin: Array, model_fn: Callable[[Array], Array], *, strength: float = 1.0, scale: int = 1) -> Array`: `x8 = _to_u8(x_lin)`; `y8 = model_fn(x8)`; `base = x_lin` (or Lanczos ×`scale` of it); `delta = _from_u8(y8) − _from_u8(x8 upsampled ×scale if scale>1 else x8)`; returns `clip(base + strength × delta)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_identity_model_leaves_the_float_image_untouched() -> None:
    lin = np.random.default_rng(4).random((8, 8, 3), dtype=np.float32) * 0.9
    out = runner.apply_ai_delta(lin, lambda u8: u8)
    assert np.allclose(out, lin, atol=1e-6)  # quantisation cancels: base keeps full precision


def test_delta_is_scaled_by_strength() -> None:
    lin = np.full((4, 4, 3), 0.2, dtype=np.float32)
    brighter = lambda u8: np.clip(u8.astype(int) + 40, 0, 255).astype(np.uint8)
    full = runner.apply_ai_delta(lin, brighter, strength=1.0)
    half = runner.apply_ai_delta(lin, brighter, strength=0.5)
    assert np.allclose(half - lin, (full - lin) * 0.5, atol=1e-4)


def test_x2_model_output_is_merged_onto_an_upsampled_base() -> None:
    lin = np.random.default_rng(5).random((6, 6, 3), dtype=np.float32) * 0.5
    x2 = lambda u8: np.repeat(np.repeat(u8, 2, axis=0), 2, axis=1)
    out = runner.apply_ai_delta(lin, x2, scale=2)
    assert out.shape == (12, 12, 3)
```

- [ ] **Step 2: Run to verify failure** — AttributeError.
- [ ] **Step 3: Implement**

```python
def apply_ai_delta(x_lin: Array, model_fn: Callable[[Array], Array], *, strength: float = 1.0, scale: int = 1) -> Array:
    """Run an 8-bit sRGB model and merge only its *change* into the linear float image."""
    x8 = _to_u8(x_lin)
    y8 = model_fn(x8)
    if scale > 1:
        h, w = x_lin.shape[:2]
        base = _from_u8(_to_u8(lanczos_resize(_to_u8(x_lin), (w * scale, h * scale))))  # sRGB-domain Lanczos, like the model saw
        ref8 = lanczos_resize(x8, (w * scale, h * scale))
    else:
        base, ref8 = x_lin, x8
    delta = _from_u8(y8) - _from_u8(ref8)
    return np.clip(base + strength * delta, 0.0, 1.0).astype(np.float32)
```

(`lanczos_resize` from `app.enhancement.downsample`.) In `batch._phase2` replace the `u8 = _to_u8(img) ... np.save(_from_u8(u8))` block with `img = apply_ai_delta(img, lambda u8: model.apply(u8, **step.params), scale=2 if name == "realesrgan_upscale" else 1)` and save `img`; `_phase3` no longer converts through `upsample_final` on uint8 but calls `lanczos_resize` on the float image when the size differs from the target (`_parse_target(settings.enhance_target_res, native)`), then `apply_post_ai`. For Review-Focus item 5 add the test in `tests/test_batch.py`: with `enhance_target_res="200%"` and a fake ×2 model, the written master is `(128, 192, 3)`.

- [ ] **Step 4: Run** — PASS; full suite; mypy.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): merge AI results as deltas into the 16-bit float master"`

### Task 19: Super-resolution only when needed

**Files:**
- Modify: `app/enhancement/engine/decision.py` (`native_long_edge: int | None`, `target_scale: float = 1.0`), `app/enhancement/batch.py`, `app/config.py` (`enhance_sr_min_long_edge: int = 3000`)
- Test: `tests/test_engine_decision.py`, `tests/corpus_expectations.py`

- [ ] **Step 1: Failing tests**

```python
def test_sr_skipped_for_large_source_at_native_target() -> None:
    plan = plan_from_report(_baseline(), native_long_edge=6000, target_scale=1.0)
    assert "realesrgan_upscale" not in _step_names(plan)


def test_sr_planned_when_enlarging_or_small_source() -> None:
    assert "realesrgan_upscale" in _step_names(plan_from_report(_baseline(), native_long_edge=6000, target_scale=2.0))
    assert "realesrgan_upscale" in _step_names(plan_from_report(_baseline(), native_long_edge=2400, target_scale=1.0))
```

Update `test_balanced_input_plan_is_minimal` to call with `native_long_edge=6000` and assert SR absent. Add `EXPECT` entries `absent realesrgan_upscale` for `IMG_1177`.

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — planner: `needs_sr = target_scale > 1.0 or (native_long_edge is not None and native_long_edge < sr_min_long_edge)` with `sr_min_long_edge: int = 3000` param; append the SR step only when `needs_sr`. `batch.plan_for` passes `native_long_edge=max(native)`, `target_scale=max(target)/max(native)` from `upsample_final._parse_target(settings.enhance_target_res, native)`, `sr_min_long_edge=settings.enhance_sr_min_long_edge`. Document the setting in `.env.example`/README.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(engine): plan Real-ESRGAN only when enlarging or the source is small"`

### Task 20: Face-restore gating by face quality

**Files:**
- Modify: `app/enhancement/engine/decision.py` (`faces: Sequence[FaceInfo] = ()` replaces `has_faces` as the input; `has_faces` stays as a derived bool), `app/enhancement/engine/plan.py` (`FaceInfo` dataclass), `app/enhancement/batch.py` (compute face crop sharpness), `app/config.py` (`enhance_face_restore_max_px: int = 300`)
- Test: `tests/test_engine_decision.py`

**Interfaces:**
- `FaceInfo(box: tuple[int,int,int,int], lap_var: float)` frozen; `plan_from_report(report, *, faces: Sequence[FaceInfo] = (), ...)`; a face is *degraded* when `max(w, h) < face_restore_max_px` or `lap_var < 100` or the noise rule fired.

- [ ] **Step 1: Failing tests**

```python
def test_sharp_large_faces_are_left_alone() -> None:
    faces = [FaceInfo((100, 100, 800, 800), lap_var=400.0)]
    assert "codeformer_restore" not in _step_names(plan_from_report(_baseline(), faces=faces))


def test_small_or_soft_faces_get_restored() -> None:
    small = [FaceInfo((0, 0, 120, 120), lap_var=400.0)]
    soft = [FaceInfo((0, 0, 900, 900), lap_var=40.0)]
    assert "codeformer_restore" in _step_names(plan_from_report(_baseline(), faces=small))
    assert "codeformer_restore" in _step_names(plan_from_report(_baseline(), faces=soft))
```

Update the two existing face tests (`test_faces_trigger_codeformer`, `test_no_faces_no_codeformer`, `test_face_restore_switch_off...`) to pass `faces=[FaceInfo((0,0,120,120), 40.0)]` where a trigger is expected.

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — add `FaceInfo` to `plan.py`; in the planner compute `degraded_faces = [f for f in faces if max(f.box[2], f.box[3]) < face_restore_max_px or f.lap_var < 100.0 or noisy]` and plan CodeFormer only when `degraded_faces`; `has_faces = bool(faces)` for the CLAHE/skin logic. In `batch._phase1` compute per-face `lap_var` with `cv2.Laplacian` on the sRGB-encoded luma of the crop (reuse `metrics._luma_u8`). `PhotoCandidate` unchanged; `WorkItem` gains `faces: list[FaceInfo]`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(engine): restore faces only when they are small or soft"`

### Task 21: Identity guard for restored faces

**Files:**
- Modify: `app/enhancement/face_restore.py` (`CodeFormerModel.apply(rgb, *, weight, faces=(), embeddings=(), min_similarity=0.5)`), `app/enhancement/enhance_job.py` (`_candidates` also loads `Face.embedding` bytes → `FaceBox` list becomes `list[tuple[FaceBox, bytes | None]]`), `app/enhancement/batch.py`
- Test: `tests/test_face_identity.py` (create; CPU with a fake embedder)

**Interfaces:**
- `face_restore.identity_ok(before: Array, after: Array, box: Box, reference: Array, embed: Callable[[Array], Array], min_similarity: float) -> bool`; `CodeFormerModel.apply` restores, then for each known face box compares `embed(after_crop)` to the stored ArcFace vector and pastes the *original* crop back when similarity < `min_similarity`. `embed` defaults to an InsightFace embedder loaded lazily inside the model (buffalo_l recognition only).

- [ ] **Step 1: Failing tests**

```python
def test_identity_guard_reverts_a_face_that_drifted() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8); after = before.copy(); after[10:50, 10:50] = 255
    ref = np.array([1.0, 0.0], dtype=np.float32)
    drift = lambda crop: np.array([0.0, 1.0], dtype=np.float32)  # orthogonal -> cos 0
    out = guard_faces(before, after, [((10, 10, 40, 40), ref)], embed=drift, min_similarity=0.5)
    assert int(out[20, 20, 0]) == 0  # reverted to the original crop


def test_identity_guard_keeps_a_face_that_matches() -> None:
    from app.enhancement.face_restore import guard_faces

    before = np.zeros((100, 100, 3), dtype=np.uint8); after = before.copy(); after[10:50, 10:50] = 255
    ref = np.array([1.0, 0.0], dtype=np.float32)
    same = lambda crop: ref
    out = guard_faces(before, after, [((10, 10, 40, 40), ref)], embed=same, min_similarity=0.5)
    assert int(out[20, 20, 0]) == 255
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement**

```python
def guard_faces(before: Array, after: Array, faces: Sequence[tuple[Box, Array]], *, embed: Callable[[Array], Array], min_similarity: float) -> Array:
    out = after.copy()
    for (x, y, w, h), ref in faces:
        crop = after[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        v = embed(crop).astype(np.float32); r = ref.astype(np.float32)
        cos = float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-8))
        if cos < min_similarity:
            log.warning("identity drift on face at (%d,%d): cos=%.2f < %.2f; reverting", x, y, cos, min_similarity)
            out[y : y + h, x : x + w] = before[y : y + h, x : x + w]
    return out
```

`CodeFormerModel.apply(..., faces=(), min_similarity=0.5)` calls `guard_faces` when `faces` is non-empty, with `embed=self._embed` that lazily builds `insightface.model_zoo.get_model("<models>/insightface/models/buffalo_l/w600k_r50.onnx")` and returns its normed embedding of the 112×112-aligned crop (use `FaceAnalysis(name="buffalo_l", allowed_modules=["recognition"], root=...)` as in `app/embedding/faces.py`). Stored embeddings are float16 bytes (`Face.embedding`) → `np.frombuffer(..., dtype=np.float16)`. `batch` passes `faces=[(box, emb) ...]` scaled to the current image size (×2 when after SR).

- [ ] **Step 4: Run** — PASS. GPU smoke: run enhance on `IMG_1158` (large sharp face, expect skip) and `IMG_0604` (small faces, expect restore + guard log).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(enhance): ArcFace identity guard reverts restored faces that drift"`

### Task 22: Neutral-pixel white balance

**Files:**
- Modify: `app/enhancement/engine/metrics.py` (`color_metrics` adds `neutral_fraction`, `rg_neutral`, `bg_neutral`), `app/enhancement/engine/plan.py` (`QualityReport` fields with `None` defaults), `app/enhancement/engine/scoring.py` (pass-through), `app/enhancement/engine/decision.py` (WB rule)
- Test: `tests/test_engine_metrics.py`, `tests/test_engine_decision.py`, `tests/corpus_expectations.py`

**Interfaces:**
- Near-neutral pixel: linear RGB with `max(c) − min(c) < 0.08 × max(c)` and luma in `[0.05, 0.9]`. `neutral_fraction` = share of such pixels; `rg_neutral`/`bg_neutral` = channel ratios over them (None when `neutral_fraction < 0.02`).
- WB rule: trigger when `neutral_fraction ≥ 0.02` and `cast_neutral > 0.12`; `strength = min(0.5, (cast_neutral − 0.12) × 2.5)`; targets 1.0.

- [ ] **Step 1: Failing tests**

```python
def test_neutral_pixels_estimate_ignores_a_blue_sky() -> None:
    img = np.zeros((20, 20, 3), dtype=np.float32)
    img[:10] = (0.2, 0.4, 0.9)     # sky: strongly chromatic, excluded
    img[10:] = (0.5, 0.5, 0.5)     # grey wall: neutral
    m = color_metrics(img)
    assert m["neutral_fraction"] == 0.5 and abs(m["rg_neutral"] - 1.0) < 1e-6


def test_blue_hour_scene_without_neutrals_gets_no_white_balance() -> None:
    plan = plan_from_report(_baseline(rg_ratio=0.8, bg_ratio=1.3, neutral_fraction=0.005, rg_neutral=None, bg_neutral=None))
    assert "white_balance" not in _step_names(plan)


def test_tungsten_cast_on_neutral_surfaces_is_half_corrected_at_most() -> None:
    plan = plan_from_report(_baseline(neutral_fraction=0.3, rg_neutral=1.4, bg_neutral=0.7))
    wb = next(s for s in plan.steps if s.name == "white_balance")
    assert wb.params["strength"] <= 0.5
```

`EXPECT`: `"IMG_0958": {"absent": {"white_balance"}}`, `"IMG_1176": {"absent": {"white_balance"}}`.

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — metrics:

```python
    mx = rgb_f01.max(axis=-1); mn = rgb_f01.min(axis=-1)
    lum = luma(rgb_f01)
    neutral = ((mx - mn) < 0.08 * np.maximum(mx, 1e-6)) & (lum > 0.05) & (lum < 0.9)
    neutral_fraction = float(neutral.mean())
    if neutral_fraction >= 0.02:
        nv = rgb_f01[neutral].mean(axis=0); g = max(float(nv[1]), 1e-6)
        rg_neutral, bg_neutral = float(nv[0] / g), float(nv[2] / g)
    else:
        rg_neutral = bg_neutral = None
```

`QualityReport` gains `neutral_fraction: float = 0.0`, `rg_neutral: float | None = None`, `bg_neutral: float | None = None` (at the end so positional construction in tests still works); `score_report` copies them. Planner WB rule replaces the gray-world block:

```python
    if report.rg_neutral is not None and report.bg_neutral is not None:
        cast = max(abs(report.rg_neutral - 1.0), abs(report.bg_neutral - 1.0))
        if cast > 0.12:
            steps.append(StepSpec("white_balance", {"target_rg": 1.0, "target_bg": 1.0,
                                  "strength": min(0.5, (cast - 0.12) * 2.5)},
                                  reason=f"neutral cast rg={report.rg_neutral:.3f} bg={report.bg_neutral:.3f} over {report.neutral_fraction:.0%}"))
```

`white_balance.gray_world` gets an optional `mask` so the correction gains are computed from the same neutral pixels (pass the mask through params as a threshold recomputed inside the step to keep params JSON-serialisable: `params["neutral_only"] = True`).

- [ ] **Step 4: Run** — PASS (unit + corpus).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(engine): white balance from near-neutral pixels, capped and gated"`

### Task 23: CLAHE, chroma and sharpness restraint

**Files:**
- Modify: `app/enhancement/engine/metrics.py` (`mean_chroma` in OKLCh, `lap_var_top`), `app/enhancement/engine/plan.py`, `app/enhancement/engine/scoring.py`, `app/enhancement/engine/decision.py`
- Test: `tests/test_engine_metrics.py`, `tests/test_engine_decision.py`, `tests/corpus_expectations.py`

**Interfaces:**
- `mean_chroma`: mean OKLCh chroma of the linear image (OKLab from linear Rec.2020 via the sRGB matrix then the OKLab matrices); monochrome when `< 0.01`.
- `lap_var_top`: Laplacian variance averaged over the sharpest 25% of 16×16 blocks.
- Rules: CLAHE `clip_limit ≤ 2.0`, skipped when faces are present; saturation boost skipped when monochrome; unsharp triggers on `lap_var_top < 150` instead of the global value.

- [ ] **Step 1: Failing tests**

```python
def test_monochrome_image_gets_no_saturation_boost() -> None:
    plan = plan_from_report(_baseline(avg_saturation=0.05, mean_chroma=0.002))
    assert "saturation_adjust" not in _step_names(plan)


def test_clahe_is_capped_and_skipped_for_portraits() -> None:
    flat = _baseline(dr_p95_p5=100.0, local_dr_mean=40.0)
    clip = next(s for s in plan_from_report(flat).steps if s.name == "clahe_local_contrast").params["clip_limit"]
    assert clip <= 2.0
    assert "clahe_local_contrast" not in _step_names(plan_from_report(flat, faces=[FaceInfo((0, 0, 900, 900), 400.0)]))


def test_bokeh_portrait_is_not_sharpened_globally() -> None:
    # whole-frame variance low (creamy background) but the sharpest blocks are crisp
    plan = plan_from_report(_baseline(lap_var=60.0, lap_var_top=420.0))
    assert "unsharp_mask" not in _step_names(plan)


def test_lap_var_top_exceeds_global_on_a_half_sharp_frame() -> None:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[:, 32:] = np.random.default_rng(6).random((64, 32, 3), dtype=np.float32)
    m = sharpness_metrics(img)
    assert m["lap_var_top"] > m["lap_var"]
```

`EXPECT`: `"IMG_1176": {"clahe_max": 2.0}`, `"IMG_1158": {"absent": {"clahe_local_contrast"}}` (portrait; requires faces from the DB → the corpus test reads InsightFace boxes if present, else skips that entry).

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** — metrics: OKLab per Ottosson (linear sRGB → LMS matrix, cube root, LMS' → Lab matrix), `mean_chroma = mean(sqrt(a²+b²))`; `lap_var_top`: per-block variance of the Laplacian, mean of the top quartile. Planner: `factor` branch guarded by `report.mean_chroma is None or report.mean_chroma >= 0.01`; CLAHE `clip = min(2.0, ...)` and `if not faces`; unsharp uses `report.lap_var_top if report.lap_var_top is not None else report.lap_var`.
- [ ] **Step 4: Run** — PASS. Render `sheets/step3`.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(engine): restrained local contrast, monochrome-aware colour, subject-based sharpness"`

### Task 24: Docs and the Step 3 gate

- [ ] Update `CLAUDE.md` phase 7 to describe the linear pipeline, the AI-as-delta merge, gating rules, verification and step-major execution; update README's enhance row and config table (`DARKTABLE_WORKFLOW`, `DARKTABLE_BASELINE`, `ENHANCE_SR_MIN_LONG_EDGE`, `ENHANCE_FACE_RESTORE_MAX_PX`).
- [ ] Owner reviews `sheets/step3`; Step 3 PR opens after that.
- [ ] Commit: `git commit -am "docs: enhancement pipeline after the correctness pass"`

---

## Project B roadmap (milestones; task-level plan follows its own spec after Step 3)

These are not executable tasks yet. They exist so the traceability matrix below is complete and so the owner can see the shape of the remaining work.

- **M4 Scene analysis.** Zero-shot CLIP classes (portrait, group, landscape, cityscape, night, sunset/golden hour, blue hour, snow, forest, food, macro, wildlife/pet, architecture, interior, monochrome) using the stored CLIP image embeddings and a cached prompt bank; histogram intent flags (low-key, high-key, silhouette, backlit); EXIF priors (ISO, exposure time, focal length, flash, time of day). Output: `scene_profile` JSON per photo, shown in the detail panel.
- **M5 Policy layer.** A declarative per-scene preset table (allowed steps, targets, caps) replacing the fixed rule chain; one solved parametric tone curve from percentile targets (black point, shadow, mid, highlight, white) with subject-region exposure targets for portraits; vibrance in OKLCh with skin and sky protection; per-image skin sampling from face boxes.
- **M6 Masks.** A light segmenter (sky, skin, foliage, water, person) that fits the VRAM rotation; masked sky recovery, clarity excluding skin and sky, focus-aware sharpening from a block sharpness map; motion-blur detection that flags instead of sharpening.
- **M7 Feedback loop.** Before/after in the UI with the recipe, thumbs up/down stored per photo, threshold tuning from accumulated verdicts, and the corpus expectations grown from that feedback.

## Traceability matrix

| Raised by | Finding | Closed by |
|---|---|---|
| Engineer | sRGB mistaken for linear; double gamma | Tasks 1–3 |
| Retoucher | Two tone mappers | Task 4 |
| Retoucher | 8-bit AI replacement → banding | Task 18 |
| Engineer | Face boxes in preview coordinates | Task 7 |
| Retoucher | No ICC / no EXIF on masters; JPEGs lose metadata | Tasks 5, 6 |
| Retoucher | No lens/CA/profiled denoise | Tasks 15, 16 |
| Engineer | Models reloaded per photo | Tasks 11, 12 |
| Engineer | Plan invisible, unverified; crude deletion guard | Tasks 9, 10 |
| Engineer | XMP stem collisions | Task 8 |
| Photographer | Gray-world WB kills sunsets, tungsten, blue hour | Task 22 (interim), M5 |
| Photographer | Mean-luma exposure flattens night/low-key/high-key | M4, M5 |
| Photographer | Saturation rules hurt foliage/food and fog/mono | Task 23 (mono skip), M5 |
| Photographer | Real-ESRGAN on everything | Task 19 |
| Retoucher | CodeFormer rebuilds healthy faces | Tasks 20, 21 |
| Retoucher | CLAHE on skin and skies | Task 23 (cap/skip), M6 |
| Retoucher | Global sharpness vs bokeh; motion blur | Task 23 (top-quartile), M6 |
| Photographer | Stacked tone rules tuned in isolation | M5 |
| Retoucher | Fixed YCbCr skin range | M5 |
| Photographer | No scene/subject awareness | M4, M5, M6 |
| Engineer | Denoise ignores ISO / texture | Task 17, M6 |
| Engineer | No evaluation harness; taste not reviewable | Task 13, M7 |

## Self-review notes

- Spec coverage: every §5 item maps to a task above; §5.2's spike is folded into Tasks 15–16 because the probe already showed darktable-cli exposes no auto-apply keys for lens/denoise, leaving the sidecar route as the only scriptable one.
- Type consistency: `PhotoCandidate(hash, source_path, file_kind, action, preview_path=None, iso=None)`; `FaceBox = tuple[int,int,int,int]`; `FaceInfo(box, lap_var)`; `Verdict(degraded, reasons, q_before, q_after)`; `EnhanceSummary(enhanced, skipped, failed)`; `ModelLike.apply(rgb, **params)`.
- Review Focus items 1–5 are pinned in Tasks 12, 5/6, 7, 10, 18 respectively.
