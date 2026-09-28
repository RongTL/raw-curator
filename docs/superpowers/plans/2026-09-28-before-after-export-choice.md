# Before/After Export Choice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorder the pipeline so enhancement runs before review, and let the curator pick per photo — after seeing a before/after slider — whether the exported JPEG is the original or the enhanced version, with an independent keep-RAW toggle.

**Architecture:** `enhance` moves into auto-run leg 1 and runs on every RAW, writing four cached JPEGs per photo (review-res + full-res, before + after) instead of a 16-bit TIFF, and deleting/moving nothing. A single post-enhance review sets `Decision.export_choice` (`undecided`/`discard`/`original`/`enhanced`) and `Decision.keep_raw`. A new `export` stage (replacing `submit` + `export-jpeg`) copies the chosen full-res JPEG into `photos/jpeg/` and applies RAW retention. The old binary `yes`/`no` routing (`rules.py`, `decide_job.py`) is deleted.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 + Alembic, FastAPI, Typer, pydantic-settings, numpy/PIL/tifffile, pytest; frontend is CDN-React + `htm` ES modules (no build step).

**Spec:** `docs/superpowers/specs/2026-09-28-before-after-export-choice-design.md`

## Global Constraints

- Python 3.12; `ruff check` + `ruff format --check` (line-length 100, ignore E501) and `mypy --strict` must all pass (`make lint`, `make typecheck`).
- All settings are `RAWCURATOR_`-prefixed pydantic-settings fields with a reader; never read `os.environ` directly; add a `.env.example` row for each new field.
- All DB paths are container-side absolute under `/data/`; output folders mirror the `incoming/` subfolder layout via `app.paths.relative_subpath`.
- Image arrays are annotated `Array` (from `app/arrays.py`); the enhancement working space is float32 linear Rec.2020 in `[0,1]`; the only colour conversions live in `app/enhancement/colorspace.py`; a uint8 array means display-referred sRGB.
- Per-item stages catch exceptions per item, `log.exception`, and keep going; one bad file never aborts a batch.
- Source-RAW deletion is irreversible and happens ONLY after the destination JPEG exists on disk.
- Job functions live at `app/<phase>/<phase>_job.py`, named `run_<phase>()`, imported lazily by the Typer CLI.
- Tests run in-container (`make test`); the `tmp_db` fixture (tests/conftest.py) yields a real SQLite session with `Base.metadata.create_all`. GPU/`real_raw` tests are skipped by default.

## Review Focus

- **enhance-then-discard leaves orphaned render JPEGs** — a photo enhanced in leg 1 but set to `discard` in review must produce no `jpeg/` output and leave its RAW untouched in `incoming/`; its cached renders are cleaned by `make reset`. (Task 9 tests.)
- **non-RAW inputs have no enhanced version** — a `jpeg`/`heic` source is skipped by `enhance`, so `export_choice="enhanced"` is impossible; `original` must export from the source file itself, and the review UI must not offer "enhanced". (Task 9, Task 17 tests.)
- **keep_raw=false must never delete before the JPEG exists** — export writes the JPEG first, verifies it on disk, only then deletes the RAW; a JPEG-encode failure leaves the RAW in place. (Task 9 tests.)
- **already-applied rows are skipped on a second export** — `applied==1` gates re-processing so re-running `export` neither re-moves a library RAW nor re-deletes a source. (Task 8 tests.)
- **a JPEG-encode failure must not delete the RAW** — retention runs only after `convert_image_to_jpeg` returns; if it raises, the per-item handler counts a failure and the source RAW is left in place. (Task 8 test.)

---

## Phase 1 — Data model & config

### Task 1: Decision gains `export_choice` + `keep_raw` (schema + migration)

**Files:**
- Modify: `app/models.py:139-153` (Decision)
- Create: `db/migrations/versions/0005_export_choice.py`
- Test: `tests/test_schema.py` (add a case)

**Interfaces:**
- Produces: `Decision.export_choice: str` (default `"undecided"`), `Decision.keep_raw: bool` (default `True`, stored as Integer). Legacy columns (`selected`, `score_tier`, `enhance_requested`, `action`) stay in the model, unused.

- [ ] **Step 1: Write the failing test**

In `tests/test_schema.py` add:

```python
def test_decision_has_export_choice_and_keep_raw(tmp_db) -> None:
    from app.models import Decision

    tmp_db.add(Decision(photo_hash="a" * 32))
    tmp_db.flush()
    row = tmp_db.get(Decision, "a" * 32)
    assert row.export_choice == "undecided"
    assert bool(row.keep_raw) is True
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_schema.py::test_decision_has_export_choice_and_keep_raw -q`
Expected: FAIL (`AttributeError: export_choice` or column missing).

- [ ] **Step 3: Add the columns to the model**

In `app/models.py`, inside `class Decision`, after the `action` column (line 149) add:

```python
    export_choice: Mapped[str] = mapped_column(String(16), nullable=False, default="undecided")
    keep_raw: Mapped[bool] = mapped_column(Integer, nullable=False, default=1)
```

- [ ] **Step 4: Write the migration**

Create `db/migrations/versions/0005_export_choice.py`:

```python
"""decisions: export_choice + keep_raw for the before/after export flow

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLS = [
    sa.Column("export_choice", sa.String(length=16), nullable=False, server_default="undecided"),
    sa.Column("keep_raw", sa.Integer(), nullable=False, server_default="1"),
]


def upgrade() -> None:
    for col in _COLS:
        op.add_column("decisions", col)


def downgrade() -> None:
    with op.batch_alter_table("decisions") as batch:
        for col in reversed(_COLS):
            batch.drop_column(col.name)
```

- [ ] **Step 5: Run the test + full schema test, expect pass**

Run: `pytest tests/test_schema.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/models.py db/migrations/versions/0005_export_choice.py tests/test_schema.py
git commit -m "feat(model): add Decision.export_choice + keep_raw (migration 0005)"
```

---

### Task 2: Config — `review_long_edge`, `keep_raw_default`, `enhanced_dir`

**Files:**
- Modify: `app/config.py:34-95`
- Modify: `.env.example`
- Test: `tests/test_schema.py` or a small new `tests/test_config.py`

**Interfaces:**
- Produces: `settings.review_long_edge: int` (3000), `settings.keep_raw_default: bool` (True), `settings.enhanced_dir: Path` (== `cache / "enhanced"`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from __future__ import annotations

from app.config import settings


def test_enhanced_dir_under_cache() -> None:
    assert settings.enhanced_dir == settings.cache / "enhanced"


def test_review_and_keep_raw_defaults() -> None:
    assert settings.review_long_edge == 3000
    assert settings.keep_raw_default is True
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_config.py -q` → FAIL (`enhanced_dir` missing).

- [ ] **Step 3: Add the settings + property**

In `app/config.py`, in the review/enhance area (near line 66-68) add fields:

```python
    review_long_edge: int = 3000  # long edge of the before/after review JPEGs shown in the viewer
    keep_raw_default: bool = True  # default keep-RAW for a freshly-created decision row
```

And after `thumbs_dir` (line 94) add:

```python
    @property
    def enhanced_dir(self) -> Path:
        return self.cache / "enhanced"
```

- [ ] **Step 4: Document the settings**

In `.env.example`, add (near the other `RAWCURATOR_` rows):

```
# Before/after review + export
RAWCURATOR_REVIEW_LONG_EDGE=3000
RAWCURATOR_KEEP_RAW_DEFAULT=true
```

- [ ] **Step 5: Run the test, expect pass**

Run: `pytest tests/test_config.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add app/config.py .env.example tests/test_config.py
git commit -m "feat(config): review_long_edge, keep_raw_default, enhanced_dir"
```

---

## Phase 2 — Decision routing retirement & bulk staging

### Task 3: Replace `rules.py` with `export_rules.py`; delete `decide_job.py`

**Files:**
- Create: `app/decision/export_rules.py`
- Delete: `app/decision/rules.py`, `app/decision/decide_job.py`
- Delete: `tests/test_decision_rules.py`
- Create: `tests/test_export_rules.py`

**Interfaces:**
- Produces: `EXPORT_CHOICES: tuple[str, ...] = ("undecided", "discard", "original", "enhanced")`; `KEEP_CHOICES: tuple[str, ...] = ("original", "enhanced")`; `is_keeper(choice: str) -> bool`; `normalize_choice(choice: str) -> str | None` (case-insensitive, returns a member of EXPORT_CHOICES or None).

- [ ] **Step 1: Write the failing test**

Create `tests/test_export_rules.py`:

```python
from __future__ import annotations

from app.decision.export_rules import (
    EXPORT_CHOICES,
    KEEP_CHOICES,
    is_keeper,
    normalize_choice,
)


def test_choices() -> None:
    assert EXPORT_CHOICES == ("undecided", "discard", "original", "enhanced")
    assert KEEP_CHOICES == ("original", "enhanced")


def test_is_keeper() -> None:
    assert is_keeper("original") and is_keeper("enhanced")
    assert not is_keeper("discard")
    assert not is_keeper("undecided")


def test_normalize_choice() -> None:
    assert normalize_choice("Enhanced") == "enhanced"
    assert normalize_choice("bogus") is None
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_export_rules.py -q` → FAIL (module missing).

- [ ] **Step 3: Create the module**

Create `app/decision/export_rules.py`:

```python
"""Post-enhance export choice model.

The curator makes one choice per photo after seeing before/after:
- ``discard``  — no JPEG, RAW left untouched in incoming/.
- ``original`` — JPEG from the developed (un-enhanced) render.
- ``enhanced`` — JPEG from the AI-enhanced render.
``keep_raw`` (a separate flag) decides whether a kept photo's RAW is archived
to library/ or deleted after the JPEG is written. This replaces the old binary
yes/no keep-RAW routing.
"""

from __future__ import annotations

EXPORT_CHOICES: tuple[str, ...] = ("undecided", "discard", "original", "enhanced")
KEEP_CHOICES: tuple[str, ...] = ("original", "enhanced")  # produce a share JPEG


def is_keeper(choice: str) -> bool:
    """True if this choice produces an exported JPEG (original or enhanced)."""
    return choice in KEEP_CHOICES


def normalize_choice(choice: str) -> str | None:
    """Case-insensitive lookup; returns a member of EXPORT_CHOICES or None."""
    low = choice.lower()
    return low if low in EXPORT_CHOICES else None
```

- [ ] **Step 4: Delete the retired modules + test**

```bash
git rm app/decision/rules.py app/decision/decide_job.py tests/test_decision_rules.py
```

(Their importers — `enhance_job.py`, `progress.py`, `cli.py` — are rewired in later tasks; this task's commit will leave those temporarily broken, so run the whole suite only after Phase 5. If executing with a per-task gate, note this cross-task dependency in the task hand-off.)

- [ ] **Step 5: Run the new test, expect pass**

Run: `pytest tests/test_export_rules.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add app/decision/export_rules.py tests/test_export_rules.py
git commit -m "feat(decision): export_rules; remove binary rules + submit job"
```

---

### Task 4: Rewrite `bulk.py` around `export_choice` + `keep_raw`

**Files:**
- Modify: `app/decision/bulk.py`
- Test: `tests/test_bulk_decisions.py` (rewrite)

**Interfaces:**
- Consumes: `EXPORT_CHOICES`, `KEEP_CHOICES` (Task 3); `settings.keep_raw_default` (Task 2).
- Produces:
  - `stage_all(sess, *, export_choice=None, keep_raw=None) -> int` — set the provided field(s) on every non-applied photo's decision (creating rows as needed). Raises `ValueError` on an unknown `export_choice`.
  - `stage_cluster(sess, cluster_id, mode) -> int` — modes `keep_recommended` (recommended→`enhanced`, rest→`discard`), `reject_all` (all→`discard`), `keep_all` (all→`enhanced`). Raises `ValueError` on unknown mode/empty cluster.

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_bulk_decisions.py`:

```python
from __future__ import annotations

import pytest

from app.decision.bulk import stage_all, stage_cluster
from app.models import Cluster, Decision, Photo


def _photo(sess, h, cluster_id=None, recommended=0):
    sess.add(Photo(hash=h, source_path=f"/data/photos/incoming/{h}.CR3",
                   file_kind="raw", cluster_id=cluster_id, is_recommended=recommended))


def test_stage_all_sets_export_choice(tmp_db) -> None:
    _photo(tmp_db, "a" * 32); _photo(tmp_db, "b" * 32); tmp_db.flush()
    assert stage_all(tmp_db, export_choice="enhanced") == 2
    assert tmp_db.get(Decision, "a" * 32).export_choice == "enhanced"


def test_stage_all_sets_keep_raw_only(tmp_db) -> None:
    _photo(tmp_db, "a" * 32); tmp_db.flush()
    stage_all(tmp_db, keep_raw=False)
    row = tmp_db.get(Decision, "a" * 32)
    assert bool(row.keep_raw) is False
    assert row.export_choice == "undecided"  # untouched


def test_stage_all_skips_applied(tmp_db) -> None:
    _photo(tmp_db, "a" * 32); tmp_db.flush()
    tmp_db.add(Decision(photo_hash="a" * 32, export_choice="original", applied=1))
    tmp_db.flush()
    assert stage_all(tmp_db, export_choice="enhanced") == 0
    assert tmp_db.get(Decision, "a" * 32).export_choice == "original"


def test_stage_all_rejects_bad_choice(tmp_db) -> None:
    with pytest.raises(ValueError):
        stage_all(tmp_db, export_choice="bogus")


def test_stage_cluster_keep_recommended(tmp_db) -> None:
    tmp_db.add(Cluster(id=1, kind="burst")); tmp_db.flush()
    _photo(tmp_db, "a" * 32, cluster_id=1, recommended=1)
    _photo(tmp_db, "b" * 32, cluster_id=1, recommended=0)
    tmp_db.flush()
    assert stage_cluster(tmp_db, 1, "keep_recommended") == 2
    assert tmp_db.get(Decision, "a" * 32).export_choice == "enhanced"
    assert tmp_db.get(Decision, "b" * 32).export_choice == "discard"
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_bulk_decisions.py -q` → FAIL.

- [ ] **Step 3: Rewrite `bulk.py`**

Replace `app/decision/bulk.py` contents:

```python
"""Bulk-stage the same choice across a cluster or the whole batch.

Used by the review header controls ("use enhanced for all", "keep no RAW", …)
and the cluster keep-best/reject-all buttons. Operates on a caller-provided
session so the route owns the transaction and tests drive it with a temp session.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.decision.export_rules import EXPORT_CHOICES
from app.models import Decision, Photo

_CLUSTER_MODES = {"keep_recommended", "reject_all", "keep_all"}


def _decision(sess: Session, h: str) -> Decision | None:
    """The row to mutate, or None if it exists and is already applied (skip)."""
    dec = sess.get(Decision, h)
    if dec is None:
        dec = Decision(photo_hash=h, keep_raw=int(settings.keep_raw_default))
        sess.add(dec)
        return dec
    return None if dec.applied else dec


def stage_all(
    sess: Session, *, export_choice: str | None = None, keep_raw: bool | None = None
) -> int:
    """Set the provided field(s) on every non-applied photo's decision.

    Returns the number of rows staged. Raises ValueError for an unknown choice.
    """
    if export_choice is not None and export_choice not in EXPORT_CHOICES:
        raise ValueError(f"export_choice must be one of {EXPORT_CHOICES}, got {export_choice!r}")
    staged = 0
    for h in sess.execute(select(Photo.hash)).scalars().all():
        dec = _decision(sess, h)
        if dec is None:
            continue
        if export_choice is not None:
            dec.export_choice = export_choice
        if keep_raw is not None:
            dec.keep_raw = int(keep_raw)
        staged += 1
    return staged


def stage_cluster(sess: Session, cluster_id: int, mode: str) -> int:
    """keep_recommended: recommended→enhanced, rest→discard; reject_all: all
    discard; keep_all: all enhanced. Skips applied rows. Raises ValueError for
    an unknown mode or empty cluster."""
    if mode not in _CLUSTER_MODES:
        raise ValueError(f"mode must be one of {sorted(_CLUSTER_MODES)}, got {mode!r}")
    rows = sess.execute(
        select(Photo.hash, Photo.is_recommended).where(Photo.cluster_id == cluster_id)
    ).all()
    if not rows:
        raise ValueError(f"no photos in cluster {cluster_id}")
    staged = 0
    for h, is_recommended in rows:
        if mode == "keep_all":
            choice = "enhanced"
        elif mode == "reject_all":
            choice = "discard"
        else:  # keep_recommended
            choice = "enhanced" if is_recommended else "discard"
        dec = _decision(sess, h)
        if dec is None:
            continue
        dec.export_choice = choice
        staged += 1
    return staged
```

- [ ] **Step 4: Run the tests, expect pass**

Run: `pytest tests/test_bulk_decisions.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/decision/bulk.py tests/test_bulk_decisions.py
git commit -m "feat(decision): bulk staging for export_choice + keep_raw"
```

---

## Phase 3 — Enhance produces before/after renders

### Task 5: `render_jpeg.py` — write display JPEGs from linear-Rec.2020 floats

**Files:**
- Create: `app/enhancement/render_jpeg.py`
- Test: `tests/test_render_jpeg.py`

**Interfaces:**
- Consumes: `linear_rec2020_to_srgb_u8` (colorspace), `resize_long_edge`/`write_jpeg` (preview.jpeg_writer), `copy_metadata` (pack_tiff), `settings.enhanced_dir`.
- Produces:
  - `render_paths(photo_hash: str) -> dict[str, Path]` with keys `before`, `after`, `before_full`, `after_full` under `settings.enhanced_dir`, named `<hash>.<key>.jpg` (using `.` separators, e.g. `<hash>.after.full.jpg` for `after_full`).
  - `write_render(linear_rgb: Array, dest: Path, *, long_edge: int, quality: int, source: str | None = None) -> None` — linear Rec.2020 float32 → sRGB uint8 → optional long-edge cap → JPEG; copies EXIF from `source` (baking Orientation=1) when given.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_jpeg.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.enhancement import render_jpeg


def test_render_paths_named_by_hash(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(render_jpeg.settings, "cache", tmp_path)
    p = render_jpeg.render_paths("a" * 32)
    assert p["before"] == tmp_path / "enhanced" / f"{'a' * 32}.before.jpg"
    assert p["after_full"] == tmp_path / "enhanced" / f"{'a' * 32}.after.full.jpg"


def test_write_render_caps_long_edge(tmp_path: Path) -> None:
    lin = np.zeros((100, 400, 3), dtype=np.float32)
    dest = tmp_path / "r.jpg"
    render_jpeg.write_render(lin, dest, long_edge=200, quality=90)
    with Image.open(dest) as im:
        assert max(im.size) == 200


def test_write_render_native_when_long_edge_zero(tmp_path: Path) -> None:
    lin = np.zeros((100, 400, 3), dtype=np.float32)
    dest = tmp_path / "r.jpg"
    render_jpeg.write_render(lin, dest, long_edge=0, quality=90)
    with Image.open(dest) as im:
        assert im.size == (400, 100)
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_render_jpeg.py -q` → FAIL.

- [ ] **Step 3: Create the module**

Create `app/enhancement/render_jpeg.py`:

```python
"""Write display-referred JPEGs from the enhancement engine's linear floats.

The before/after review needs viewable images the moment enhance finishes, and
the final export copies the full-res one. Four JPEGs per photo live under
``settings.enhanced_dir`` (cleaned by ``make reset``): review-res + full-res,
before (developed, un-enhanced) + after (AI-enhanced).
"""

from __future__ import annotations

from pathlib import Path

from app.arrays import Array
from app.config import settings
from app.enhancement.colorspace import linear_rec2020_to_srgb_u8
from app.enhancement.pack_tiff import copy_metadata
from app.preview.jpeg_writer import resize_long_edge, write_jpeg

_CHROMA_420 = 2

_SUFFIXES = {
    "before": "before.jpg",
    "after": "after.jpg",
    "before_full": "before.full.jpg",
    "after_full": "after.full.jpg",
}


def render_paths(photo_hash: str) -> dict[str, Path]:
    d = settings.enhanced_dir
    return {key: d / f"{photo_hash}.{suffix}" for key, suffix in _SUFFIXES.items()}


def write_render(
    linear_rgb: Array, dest: Path, *, long_edge: int, quality: int, source: str | None = None
) -> None:
    """linear Rec.2020 float32 [0,1] -> sRGB JPEG at ``dest`` (parents created).

    ``long_edge<=0`` keeps native size. ``source`` (a path) copies EXIF across
    with Orientation baked to 1 (the float image is already upright).
    """
    u8 = resize_long_edge(linear_rec2020_to_srgb_u8(linear_rgb), long_edge)
    write_jpeg(u8, dest, quality=quality, progressive=settings.jpeg_progressive, subsampling=_CHROMA_420)
    if source is not None:
        copy_metadata(Path(source), dest)
```

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_render_jpeg.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/enhancement/render_jpeg.py tests/test_render_jpeg.py
git commit -m "feat(enhance): render_jpeg writes before/after display JPEGs"
```

---

### Task 6: `enhance_job._candidates` selects all RAW; drop `action`/`may_delete_source`

**Files:**
- Modify: `app/enhancement/enhance_job.py`
- Test: `tests/test_enhance_job.py` (rewrite the candidate + delete-gate cases)

**Interfaces:**
- Produces: `PhotoCandidate` without the `action` field. `_candidates()` returns snapshots for every `Photo.file_kind == "raw"` regardless of any decision (enhance now runs before review). `may_delete_source` is removed.

- [ ] **Step 1: Write the failing test**

In `tests/test_enhance_job.py`, replace the candidate/delete-gate tests with:

```python
def test_candidates_selects_all_raw(tmp_db, monkeypatch) -> None:
    from app.enhancement import enhance_job
    from app.models import Photo

    tmp_db.add(Photo(hash="a" * 32, source_path="/data/photos/incoming/a.CR3", file_kind="raw"))
    tmp_db.add(Photo(hash="b" * 32, source_path="/data/photos/incoming/b.JPG", file_kind="jpeg"))
    tmp_db.commit()

    import contextlib
    @contextlib.contextmanager
    def _scope():
        yield tmp_db
    monkeypatch.setattr(enhance_job, "session_scope", _scope)

    cands = enhance_job._candidates()
    hashes = {c.hash for c, _ in cands}
    assert hashes == {"a" * 32}  # non-RAW excluded; no decision needed


def test_photo_candidate_has_no_action() -> None:
    from dataclasses import fields
    from app.enhancement.enhance_job import PhotoCandidate

    assert "action" not in {f.name for f in fields(PhotoCandidate)}
```

Delete the old `test_may_delete_source*` cases (referencing `enhance_job.may_delete_source`).

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_enhance_job.py -q` → FAIL.

- [ ] **Step 3: Edit `enhance_job.py`**

- Remove the import `from app.decision.rules import ENHANCE_ACTIONS` (line 29).
- In `PhotoCandidate` (39-53), delete the `action: str` field.
- In `_candidates` (56-113): drop `Decision.action` from the `select(...)`, drop the `.join(Decision, ...)` and `.where(Decision.action.in_(ENHANCE_ACTIONS))`, replace with `.where(Photo.file_kind == "raw")`; remove `action` from the unpack and the `PhotoCandidate(...)` construction. Remove the now-unused `Decision` import.
- Delete `may_delete_source` (149-150) and its mention in the module docstring (line 11).

The rewritten `_candidates` select becomes:

```python
        rows = sess.execute(
            select(
                Photo.hash,
                Photo.source_path,
                Photo.file_kind,
                Photo.preview_path,
                Photo.iso,
                Photo.camera_make,
                Photo.camera_body,
                Photo.lens,
                Photo.aperture,
                Photo.focal_length,
            ).where(Photo.file_kind == "raw")
        ).all()
```

with the loop unpack and `PhotoCandidate(...)` losing `action`.

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_enhance_job.py -q` → PASS (batch.py still imports `may_delete_source` — that is fixed in Task 7; if running the whole file fails on import, complete Task 7 before the suite gate).

- [ ] **Step 5: Commit**

```bash
git add app/enhancement/enhance_job.py tests/test_enhance_job.py
git commit -m "feat(enhance): enhance every RAW, drop decision coupling + delete gate"
```

---

### Task 7: `batch.py` writes renders instead of a TIFF; no RAW deletion

**Files:**
- Modify: `app/enhancement/batch.py`
- Test: `tests/test_batch.py` (add a render-writing case; drop TIFF/delete assertions)

**Interfaces:**
- Consumes: `render_paths`, `write_render` (Task 5); `PhotoCandidate` without `action` (Task 6).
- Produces: after a successful batch, for each RAW photo `settings.enhanced_dir/<hash>.{before,after}.jpg` and `.{before,after}.full.jpg` exist. No `photos/exported/*.tif` is written; no source RAW is moved or deleted. `WorkItem.out` points at the `after_full` JPEG (for logging).

- [ ] **Step 1: Write/adjust the failing test**

In `tests/test_batch.py`, add (the existing suite stubs `develop`/models — mirror its style; a representative case):

```python
def test_phase3_writes_renders_not_tiff(monkeypatch, tmp_path):
    """A finished item writes before/after JPEGs under enhanced_dir and no TIFF."""
    from app.enhancement import batch, render_jpeg

    monkeypatch.setattr(render_jpeg.settings, "cache", tmp_path)
    # ... build a WorkItem with a small float32 intermediate + plan/report as the
    # existing batch tests do, run batch._phase3(item, fake_develop), then:
    paths = render_jpeg.render_paths(item.photo.hash)
    assert paths["after"].exists() and paths["after_full"].exists()
    assert not (tmp_path.parent / "photos" / "exported").exists()
```

(Follow `tests/test_batch.py`'s existing WorkItem construction; the assertion set is the deliverable.)

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_batch.py -q` → FAIL.

- [ ] **Step 3: Edit imports in `batch.py`**

- Remove `may_delete_source` from the `enhance_job` import (37-47).
- Remove `from app.enhancement.pack_tiff import copy_metadata, write_tiff16` (50).
- Remove `from app.paths import relative_subpath` (56) if now unused (it is — only phase3's TIFF path used it).
- Add: `from app.enhancement.render_jpeg import render_paths, write_render`.

- [ ] **Step 4: Write before renders in `_phase1`**

In `_phase1`, after `item.faces = _face_infos(img, item.face_boxes)` (line 190) and before `item.report = ...`, insert:

```python
    paths = render_paths(item.photo.hash)
    write_render(img, paths["before"], long_edge=settings.review_long_edge,
                 quality=settings.jpeg_quality_preview)
    write_render(img, paths["before_full"], long_edge=0, quality=settings.jpeg_quality,
                 source=item.photo.source_path)
```

(`img` here is developed + lens-corrected, before `apply_pre_ai` — the true "before enhancement".)

- [ ] **Step 5: Replace the TIFF write + deletion in `_phase3`**

Replace lines 300-313 (from `item.plan = plan` through the `elif ... degraded` block) with:

```python
    item.plan = plan
    persist_all(item, verdict)
    paths = render_paths(item.photo.hash)
    write_render(result, paths["after"], long_edge=settings.review_long_edge,
                 quality=settings.jpeg_quality_preview)
    write_render(result, paths["after_full"], long_edge=0, quality=settings.jpeg_quality,
                 source=item.photo.source_path)
    item.out = paths["after_full"]
```

No RAW is deleted or moved here anymore (that is the `export` stage's job, after the curator chooses).

- [ ] **Step 6: Run tests, expect pass**

Run: `pytest tests/test_batch.py tests/test_enhance_job.py -q` → PASS.

- [ ] **Step 7: Commit**

```bash
git add app/enhancement/batch.py tests/test_batch.py
git commit -m "feat(enhance): write before/after renders, stop writing TIFF + deleting RAW"
```

---

## Phase 4 — Export step

### Task 8: New `export` stage — JPEG from chosen render + RAW retention

**Files:**
- Create: `app/export/export_job.py`
- Delete: `app/export/jpeg_job.py`
- Delete: `tests/test_jpeg_export.py`
- Create: `tests/test_export_job.py`

**Interfaces:**
- Consumes: `KEEP_CHOICES`/`is_keeper` (export_rules), `render_paths` (render_jpeg), `convert_image_to_jpeg` (export.jpeg_writer), `Move`/`apply_moves` (decision.executor), `relative_subpath` (paths).
- Produces: `run_export() -> ExportSummary` (dataclass: `exported`, `skipped`, `failed`). For each `Decision` with `is_keeper(export_choice)` and `applied==0`: write `photos/jpeg/<sub>/X.jpg` from the chosen source, then apply retention (`keep_raw` → move RAW to `library/`; else delete RAW after the JPEG exists), then set `applied=1` and update `Photo.source_path` if moved. `_source_jpeg(choice, photo) -> Path` picks the render (`enhanced`→`after_full`; `original`→`before_full` when it exists (RAW), else the non-RAW source file).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_job.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.export import export_job
from app.models import Decision, Photo


def _jpeg(path: Path, size=(8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(path, "JPEG")


@pytest.fixture
def wired(tmp_db, tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    monkeypatch.setattr(export_job.settings, "photos", photos)
    monkeypatch.setattr(export_job.settings, "cache", tmp_path / "cache")
    monkeypatch.setattr(export_job.settings, "jpeg_subdir", "jpeg")
    monkeypatch.setattr(export_job.settings, "jpeg_quality", 90)
    monkeypatch.setattr(export_job.settings, "jpeg_long_edge", 0)
    import contextlib
    @contextlib.contextmanager
    def _scope():
        yield tmp_db
    monkeypatch.setattr(export_job, "session_scope", _scope)
    return photos


def test_enhanced_keep_raw_moves_to_library(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths
    raw = wired / "incoming" / "IMG.CR3"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=1))
    tmp_db.commit()

    export_job.run_export()

    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert (wired / "library" / "IMG.CR3").exists()  # archived
    assert not raw.exists()  # moved out of incoming
    assert tmp_db.get(Decision, "h" * 32).applied == 1


def test_enhanced_no_keep_deletes_raw_after_jpeg(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths
    raw = wired / "incoming" / "IMG.CR3"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=0))
    tmp_db.commit()

    export_job.run_export()

    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert not raw.exists()  # deleted (after JPEG)
    assert not (wired / "library" / "IMG.CR3").exists()


def test_original_uses_before_render(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths
    raw = wired / "incoming" / "IMG.CR3"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["before_full"], size=(10, 10))
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="original", keep_raw=1))
    tmp_db.commit()

    export_job.run_export()
    assert (wired / "jpeg" / "IMG.jpg").exists()


def test_discard_produces_nothing(wired, tmp_db) -> None:
    raw = wired / "incoming" / "IMG.CR3"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="discard"))
    tmp_db.commit()

    export_job.run_export()
    assert not (wired / "jpeg").exists()
    assert raw.exists()  # untouched
    assert tmp_db.get(Decision, "h" * 32).applied == 0


def test_second_run_skips_applied(wired, tmp_db) -> None:
    from app.enhancement.render_jpeg import render_paths
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(wired / "library" / "IMG.CR3"), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=1, applied=1))
    tmp_db.commit()
    export_job.run_export()  # applied → skipped, no crash on missing incoming RAW
    assert not (wired / "jpeg" / "IMG.jpg").exists()


def test_non_raw_original_uses_source(wired, tmp_db) -> None:
    src = wired / "incoming" / "IMG.JPG"; _jpeg(src)
    tmp_db.add(Photo(hash="h" * 32, source_path=str(src), file_kind="jpeg"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="original", keep_raw=1))
    tmp_db.commit()
    export_job.run_export()
    assert (wired / "jpeg" / "IMG.jpg").exists()
    assert (wired / "library" / "IMG.JPG").exists()  # keep_raw archives the original file


def test_encode_failure_leaves_raw(wired, tmp_db, monkeypatch) -> None:
    from app.enhancement.render_jpeg import render_paths
    raw = wired / "incoming" / "IMG.CR3"; raw.parent.mkdir(parents=True); raw.write_bytes(b"raw")
    _jpeg(render_paths("h" * 32)["after_full"])
    tmp_db.add(Photo(hash="h" * 32, source_path=str(raw), file_kind="raw"))
    tmp_db.add(Decision(photo_hash="h" * 32, export_choice="enhanced", keep_raw=0))
    tmp_db.commit()

    def _boom(*a, **k):
        raise RuntimeError("encode failed")
    monkeypatch.setattr(export_job, "convert_image_to_jpeg", _boom)

    summary = export_job.run_export()
    assert summary.failed == 1
    assert raw.exists()  # NOT deleted — encode failed before retention
    assert tmp_db.get(Decision, "h" * 32).applied == 0
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_export_job.py -q` → FAIL (module missing).

- [ ] **Step 3: Create `export_job.py`**

Create `app/export/export_job.py`:

```python
"""Apply export decisions: chosen JPEG into photos/jpeg/, then RAW retention.

Replaces the old submit + export-jpeg stages. Driven by the DB, not the
filesystem: each kept decision picks exactly one source render, so the old
library-vs-exported jpeg collision is gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.decision.executor import Move, apply_moves
from app.decision.export_rules import is_keeper
from app.enhancement.render_jpeg import render_paths
from app.export.jpeg_writer import convert_image_to_jpeg
from app.models import Decision, Photo
from app.paths import relative_subpath

log = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class ExportSummary:
    exported: int = 0
    skipped: int = 0
    failed: int = 0


def _source_jpeg(choice: str, photo: Photo) -> Path:
    """Which image to encode into the share JPEG for this photo."""
    paths = render_paths(photo.hash)
    if choice == "enhanced":
        return paths["after_full"]
    before = paths["before_full"]
    if before.exists():  # a developed RAW render
        return before
    return Path(photo.source_path)  # non-RAW: the original file itself


def _dest_for(source_path: str) -> Path:
    rel = relative_subpath(Path(source_path), settings.photos).with_suffix(".jpg")
    return settings.photos / settings.jpeg_subdir / rel


def run_export() -> ExportSummary:
    exported = skipped = failed = 0
    with session_scope() as sess:
        rows = sess.execute(
            select(Photo, Decision)
            .join(Decision, Photo.hash == Decision.photo_hash)
            .where(Decision.applied == 0)
        ).all()
        todo = [(p, d) for p, d in rows if is_keeper(d.export_choice)]
        if not todo:
            console.print("[yellow]No photos to export.[/yellow]")
            return ExportSummary()

        with Progress() as progress:
            task = progress.add_task("export", total=len(todo))
            for photo, decision in todo:
                try:
                    src = _source_jpeg(decision.export_choice, photo)
                    if not src.exists():
                        log.warning("skip %s: source render missing %s", photo.hash, src)
                        skipped += 1
                        continue
                    dest = _dest_for(photo.source_path)
                    convert_image_to_jpeg(
                        src, dest, quality=settings.jpeg_quality,
                        long_edge=settings.jpeg_long_edge, progressive=settings.jpeg_progressive,
                    )
                    # Retention runs only after the JPEG exists on disk.
                    raw = Path(photo.source_path)
                    if decision.keep_raw and raw.exists():
                        lib = settings.photos / "library" / relative_subpath(raw, settings.photos)
                        apply_moves([Move(src=raw, dst=lib)])
                        photo.source_path = str(lib)
                    elif not decision.keep_raw and raw.exists():
                        raw.unlink()
                        log.info("deleted source RAW after export: %s", raw)
                    decision.applied = 1
                    exported += 1
                    console.print(f"  -> {dest}")
                except Exception as exc:  # noqa: BLE001 — keep the batch going
                    failed += 1
                    log.exception("export failed for %s", photo.source_path)
                    console.print(f"  [red]x {photo.hash}: {exc}[/red]")
                finally:
                    progress.advance(task)

    colour = "red" if failed else "green"
    console.print(
        f"[{colour}]Export complete:[/{colour}] exported={exported} skipped={skipped} failed={failed}"
    )
    return ExportSummary(exported=exported, skipped=skipped, failed=failed)
```

- [ ] **Step 4: Delete the old job + test**

```bash
git rm app/export/jpeg_job.py tests/test_jpeg_export.py
```

- [ ] **Step 5: Run the tests, expect pass**

Run: `pytest tests/test_export_job.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add app/export/export_job.py tests/test_export_job.py
git commit -m "feat(export): DB-driven export stage (JPEG from choice + RAW retention)"
```

---

## Phase 5 — Orchestration & CLI wiring

### Task 9: Stage registry — enhance→leg 1, export→leg 2, drop submit/export-jpeg

**Files:**
- Modify: `app/orchestrator/stages.py:21-29`
- Modify: `app/orchestrator/autorun.py:1-5` (docstring only)
- Test: `tests/test_orchestrator_stages.py`

**Interfaces:**
- Produces: `STAGES` = `ingest, filter, score, cluster, enhance` (leg 1); `export` (leg 2). `STAGE_BY_NAME["export"].cli_args == ("export",)`.

- [ ] **Step 1: Update the test**

Rewrite the relevant asserts in `tests/test_orchestrator_stages.py`:

```python
def test_leg_membership() -> None:
    from app.orchestrator.stages import leg_stages
    assert [s.name for s in leg_stages(1)] == ["ingest", "filter", "score", "cluster", "enhance"]
    assert [s.name for s in leg_stages(2)] == ["export"]


def test_export_cli_args() -> None:
    from app.orchestrator.stages import STAGE_BY_NAME
    assert STAGE_BY_NAME["export"].cli_args == ("export",)
    assert "submit" not in STAGE_BY_NAME
    assert "export-jpeg" not in STAGE_BY_NAME
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_orchestrator_stages.py -q` → FAIL.

- [ ] **Step 3: Edit `STAGES`**

In `app/orchestrator/stages.py` replace the `STAGES` tuple (21-29):

```python
STAGES: tuple[StageDef, ...] = (
    StageDef("ingest", "Ingest", ("ingest",), 1),
    StageDef("filter", "Filter", ("filter",), 1),
    StageDef("score", "Score", ("score",), 1),
    StageDef("cluster", "Cluster", ("cluster",), 1),
    StageDef("enhance", "Enhance", ("enhance",), 1),
    StageDef("export", "Export", ("export",), 2),
)
```

Update `app/orchestrator/autorun.py` docstring (2-5) to: `Leg 1 = ingest → filter → score → cluster → enhance (then the human reviews). Leg 2 = export (after "Export selected").`

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_orchestrator_stages.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator/stages.py app/orchestrator/autorun.py tests/test_orchestrator_stages.py
git commit -m "feat(orchestrator): enhance in leg 1, single export stage in leg 2"
```

---

### Task 10: Progress handlers for `enhance` (all RAW) + `export`

**Files:**
- Modify: `app/orchestrator/progress.py`
- Test: `tests/test_orchestrator_progress.py`

**Interfaces:**
- Consumes: `KEEP_CHOICES` (export_rules), `settings.enhanced_dir`.
- Produces: `_enhance_progress()` = (count of `<hash>.after.full.jpg` in `enhanced_dir`, count of RAW photos). `_export_progress()` = (applied decisions, keeper decisions). `batch_summary()`'s `decided` counts `export_choice != "undecided"`. `_STAGE_HANDLERS` keys: `ingest, filter, score, cluster, enhance, export`.

- [ ] **Step 1: Update the test**

Rewrite the enhance/export/submit cases in `tests/test_orchestrator_progress.py`:

```python
def test_enhance_counts_after_renders(env, tmp_db) -> None:
    from app.models import Photo
    tmp_db.add(Photo(hash="a" * 32, source_path="/x/a.CR3", file_kind="raw"))
    tmp_db.add(Photo(hash="b" * 32, source_path="/x/b.CR3", file_kind="raw"))
    tmp_db.commit()
    (progress.settings.enhanced_dir).mkdir(parents=True, exist_ok=True)
    (progress.settings.enhanced_dir / f"{'a' * 32}.after.full.jpg").write_bytes(b"x")
    assert progress.stage_progress("enhance") == (1, 2)


def test_export_counts_applied_over_keepers(env, tmp_db) -> None:
    from app.models import Decision, Photo
    for h, ch, ap in [("a" * 32, "enhanced", 1), ("b" * 32, "original", 0), ("c" * 32, "discard", 0)]:
        tmp_db.add(Photo(hash=h, source_path=f"/x/{h}.CR3", file_kind="raw"))
        tmp_db.add(Decision(photo_hash=h, export_choice=ch, applied=ap))
    tmp_db.commit()
    assert progress.stage_progress("export") == (1, 2)  # 1 applied of 2 keepers
```

(Match the existing `env`/`tmp_db` wiring in the file — it monkeypatches `progress.settings` + `session_scope`.)

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_orchestrator_progress.py -q` → FAIL.

- [ ] **Step 3: Edit `progress.py`**

- Replace the import `from app.decision.rules import ENHANCE_ACTIONS` (20) with `from app.decision.export_rules import KEEP_CHOICES`.
- Delete `_submit_progress` (95-104) and `_export_jpeg_progress` (121-135).
- Replace `_enhance_progress` (107-118):

```python
def _enhance_progress() -> tuple[int, int]:
    d = settings.enhanced_dir
    done = sum(1 for _ in d.glob("*.after.full.jpg")) if d.exists() else 0
    with session_scope() as sess:
        total = _count(
            sess, select(func.count()).select_from(Photo).where(Photo.file_kind == "raw")
        )
    return min(done, total), total


def _export_progress() -> tuple[int, int]:
    with session_scope() as sess:
        applied = _count(
            sess, select(func.count()).select_from(Decision).where(Decision.applied == 1)
        )
        keepers = _count(
            sess,
            select(func.count()).select_from(Decision).where(Decision.export_choice.in_(KEEP_CHOICES)),
        )
    return applied, keepers
```

- Update `_STAGE_HANDLERS` (138-146): remove `submit`/`export-jpeg`, add `"export": _export_progress` (enhance stays mapped to `_enhance_progress`).
- In `batch_summary` (161-176) and `_export_progress`, `decided` filter changes from `Decision.selected != "undecided"` to `Decision.export_choice != "undecided"`.
- Remove now-unused imports (`relative_subpath`, `TIFF_EXTS`, `JPEG_EXTS`) if no longer referenced.

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_orchestrator_progress.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator/progress.py tests/test_orchestrator_progress.py
git commit -m "feat(orchestrator): progress for all-RAW enhance + export"
```

---

### Task 11: CLI + Makefile — drop `submit`, replace `export-jpeg` with `export`

**Files:**
- Modify: `app/cli.py:54-87`
- Modify: `Makefile:8-9, 22-23, 61-68, help block`
- Test: none automated (Typer wiring); manual `raw-curator --help` check.

**Interfaces:**
- Produces: `raw-curator export` command → `app.export.export_job.run_export()`. `submit` and `export-jpeg` commands removed. `make export` target; `make submit`/`make export-jpeg` removed.

- [ ] **Step 1: Edit `cli.py`**

- Delete the `submit` command (54-59).
- Replace the `export-jpeg` command (70-87) with:

```python
@app.command()
def export() -> None:
    """Apply export decisions: chosen JPEG into photos/jpeg/, then RAW retention."""
    from app.export.export_job import run_export

    run_export()
```

- Update the `enhance` docstring (63) to: `"""RAW -> AI chain -> before/after render JPEGs for every RAW photo."""`.

- [ ] **Step 2: Edit `Makefile`**

- `.PHONY` line (8-9): replace `submit enhance export-jpeg` with `enhance export`.
- help block (22-23): drop the `submit` line; change `export-jpeg` help to `export  Apply export choices -> share JPEGs (+ RAW retention)`.
- Targets (61-68): delete the `submit:` target; rename `export-jpeg:` → `export:` with body `$(RUN) export`.
- `reset` help (15): unchanged wording is fine.

- [ ] **Step 3: Verify wiring**

Run: `podman-compose run --rm app raw-curator --help` (or `make shell` then `raw-curator --help`).
Expected: `export` listed; `submit`/`export-jpeg` absent.

- [ ] **Step 4: Commit**

```bash
git add app/cli.py Makefile
git commit -m "feat(cli): raw-curator export replaces submit + export-jpeg"
```

---

## Phase 6 — API surface

### Task 12: `decide` route + `_urls` — accept `export_choice`/`keep_raw`, derive render URLs

**Files:**
- Modify: `app/api/routes/decide.py`
- Modify: `app/api/routes/_urls.py`
- Test: `tests/test_review_routes.py` (add cases)

**Interfaces:**
- Consumes: `stage_all`/`stage_cluster` (Task 4), `settings.keep_raw_default`.
- Produces:
  - `DecisionIn` gains `export_choice: str | None`, `keep_raw: bool | None` (partial patch).
  - `POST /api/decide/all` body `{export_choice?, keep_raw?}` → `stage_all(sess, export_choice=..., keep_raw=...)`.
  - `GET /api/decide/pending` returns rows with `applied==0` AND `export_choice IN KEEP_CHOICES`, each `{photo_hash, export_choice, keep_raw, stars, favorite, note}`.
  - `_urls.render_url(photo_hash: str, which: str) -> str` → `/cache/enhanced/<hash>.<which>.jpg` (`which` ∈ `before`/`after`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_review_routes.py` add (using the app's TestClient fixture already in the file):

```python
def test_decide_sets_export_choice_and_keep_raw(client, seeded_photo) -> None:
    r = client.post("/api/decide/", json={"photo_hash": seeded_photo, "export_choice": "original", "keep_raw": False})
    assert r.status_code == 200
    detail = client.get(f"/api/photo/{seeded_photo}").json()
    assert detail["decision"]["export_choice"] == "original"
    assert detail["decision"]["keep_raw"] is False


def test_decide_all_enhanced(client, seeded_photo) -> None:
    r = client.post("/api/decide/all", json={"export_choice": "enhanced"})
    assert r.json()["staged"] >= 1
```

(Reuse the file's existing client/seed fixtures; adapt names.)

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_review_routes.py -q` → FAIL.

- [ ] **Step 3: Edit `decide.py`**

- `DecisionIn`: add `export_choice: str | None = None` and `keep_raw: bool | None = None`.
- In `stage_decision`, when creating a new row set the keep-RAW default and apply the new fields:

```python
        if existing is None:
            existing = Decision(photo_hash=d.photo_hash, keep_raw=int(settings.keep_raw_default))
            sess.add(existing)
        if d.export_choice is not None:
            existing.export_choice = d.export_choice
        if d.keep_raw is not None:
            existing.keep_raw = int(d.keep_raw)
        if d.stars is not None:
            ...
```

Remove the `selected` branch. Add `from app.config import settings`.
- `BulkDecisionIn`: replace with `export_choice: str | None = None` and `keep_raw: bool | None = None`; `stage_all_decisions` calls `stage_all(sess, export_choice=d.export_choice, keep_raw=d.keep_raw)`.
- `list_pending`: change the filter to `Decision.applied == 0, Decision.export_choice.in_(KEEP_CHOICES)` (import `KEEP_CHOICES`), and return `export_choice`/`keep_raw` instead of `selected`.

- [ ] **Step 4: Add `render_url` to `_urls.py`**

```python
def render_url(photo_hash: str, which: str) -> str:
    """Browser URL for a before/after review JPEG (may 404 until enhance runs)."""
    return f"/cache/enhanced/{photo_hash}.{which}.jpg"
```

- [ ] **Step 5: Run the tests, expect pass**

Run: `pytest tests/test_review_routes.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/routes/decide.py app/api/routes/_urls.py tests/test_review_routes.py
git commit -m "feat(api): decide accepts export_choice/keep_raw; render URLs"
```

---

### Task 13: Serializers — emit `export_choice`, `keep_raw`, `before_url`, `after_url`

**Files:**
- Modify: `app/api/serializers.py`
- Test: `tests/test_api_serializers.py`

**Interfaces:**
- Consumes: `render_url` (Task 12).
- Produces: `decision_payload` returns `{export_choice, keep_raw, stars, favorite, applied, note}`. `photo_summary` adds `before_url` (`render_url(hash, "before")`) and `after_url` (`render_url(hash, "after")`); keeps `preview_url` (UI falls back to it when a render 404s). `enhanced_url` is removed.

- [ ] **Step 1: Update the test**

In `tests/test_api_serializers.py`, adjust the decision/summary assertions:

```python
def test_decision_payload_shape() -> None:
    from app.api.serializers import decision_payload
    from app.models import Decision
    d = Decision(photo_hash="a" * 32, export_choice="enhanced", keep_raw=0, applied=1)
    p = decision_payload(d)
    assert p == {"export_choice": "enhanced", "keep_raw": False, "stars": 0,
                 "favorite": False, "applied": True, "note": None}


def test_photo_summary_has_before_after_urls() -> None:
    from app.api.serializers import photo_summary
    from app.models import Photo
    p = Photo(hash="a" * 32, source_path="/data/photos/incoming/x.CR3", file_kind="raw")
    s = photo_summary(p, None)
    assert s["before_url"] == f"/cache/enhanced/{'a' * 32}.before.jpg"
    assert s["after_url"] == f"/cache/enhanced/{'a' * 32}.after.jpg"
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_api_serializers.py -q` → FAIL.

- [ ] **Step 3: Edit `serializers.py`**

- Import: `from app.api.routes._urls import cache_url, render_url` (drop `jpeg_url`).
- `decision_payload` returns:

```python
    return {
        "export_choice": d.export_choice,
        "keep_raw": bool(d.keep_raw),
        "stars": d.stars,
        "favorite": bool(d.favorite),
        "applied": bool(d.applied),
        "note": d.note,
    }
```

- In `photo_summary`, replace the `"enhanced_url": jpeg_url(p.source_path),` line with:

```python
        "before_url": render_url(p.hash, "before"),
        "after_url": render_url(p.hash, "after"),
```

- [ ] **Step 4: Run the tests, expect pass**

Run: `pytest tests/test_api_serializers.py tests/test_review_routes.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/serializers.py tests/test_api_serializers.py
git commit -m "feat(api): serialize export_choice/keep_raw + before/after URLs"
```

---

### Task 14: Reset wipes the `enhanced` cache tier

**Files:**
- Modify: `app/orchestrator/reset.py:20`
- Test: `tests/test_session_reset.py` (add a case)

**Interfaces:**
- Produces: `end_session` empties+recreates `cache/enhanced` alongside `previews`/`thumbs`.

- [ ] **Step 1: Write the failing test**

In `tests/test_session_reset.py` add:

```python
def test_reset_wipes_enhanced_tier(tmp_env) -> None:
    from app.config import settings
    from app.orchestrator.reset import end_session
    settings.enhanced_dir.mkdir(parents=True, exist_ok=True)
    (settings.enhanced_dir / "junk.after.full.jpg").write_bytes(b"x")
    end_session(force=True)
    assert settings.enhanced_dir.exists()
    assert not any(settings.enhanced_dir.iterdir())
```

(Match the file's existing env fixture that points `settings` at a temp tree and stubs `alembic`.)

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/test_session_reset.py -q` → FAIL.

- [ ] **Step 3: Edit `reset.py`**

Change line 20:

```python
_CACHE_TIERS = ("previews", "thumbs", "enhanced")
```

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_session_reset.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add app/orchestrator/reset.py tests/test_session_reset.py
git commit -m "feat(reset): wipe the cache/enhanced render tier"
```

---

## Phase 7 — Frontend (no build step; manual verification)

> These tasks touch `app/api/static/js/*` — CDN-React + `htm`. There is no JS test harness in the repo, so each task's verification is running `make serve` and driving the browser. Keep the existing code style (tagged-`html` templates, Tailwind classes).

### Task 15: API client — export_choice/keep_raw and object-form bulk

**Files:**
- Modify: `app/api/static/js/api.js:23-24`

**Interfaces:**
- Produces: `api.decide(body)` (body may carry `export_choice`, `keep_raw`); `api.decideAll(body)` now takes an object (`{export_choice}` or `{keep_raw}`).

- [ ] **Step 1: Edit `api.js`**

```javascript
  decide: (body) => post(`/api/decide/`, body),
  decideAll: (body) => post(`/api/decide/all`, body),
```

(`decideCluster`, `pending` unchanged.)

- [ ] **Step 2: Verify no syntax error**

Run: `make serve`, load `http://localhost:8080`, confirm the console has no module-load errors (Chrome devtools). Leave running for Tasks 16-19.

- [ ] **Step 3: Commit**

```bash
git add app/api/static/js/api.js
git commit -m "feat(ui): api client for export_choice/keep_raw + object bulk"
```

---

### Task 16: `BeforeAfterSlider` component

**Files:**
- Create: `app/api/static/js/slider.js`

**Interfaces:**
- Produces: `export function BeforeAfterSlider({ beforeSrc, afterSrc, fallbackSrc, dw, dh })` — a wipe comparison: `before` under `after`, `after` clipped by a draggable vertical handle (0-100%). On `after` load error it shows `before` only; on `before` load error it swaps in `fallbackSrc` (the ingest preview, for non-RAW).

- [ ] **Step 1: Create `slider.js`**

```javascript
// Before/after wipe comparison. Two stacked <img>; the "after" is clipped by a
// draggable handle. Pure CSS clip-path, pointer-driven; no external deps.
import { html, useRef, useState } from "./ui.js";

export function BeforeAfterSlider({ beforeSrc, afterSrc, fallbackSrc, dw, dh }) {
  const [pos, setPos] = useState(50);        // handle position, % from left
  const [afterOk, setAfterOk] = useState(true);
  const [beforeSrcState, setBeforeSrcState] = useState(beforeSrc);
  const box = useRef(null);

  const onMove = (clientX) => {
    const el = box.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };
  const onDown = (e) => {
    onMove(e.clientX);
    const move = (ev) => onMove(ev.clientX);
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const ratio = dw && dh ? `${dw} / ${dh}` : "3 / 2";
  return html`
    <div ref=${box} class="relative select-none touch-none"
         style=${{ aspectRatio: ratio, width: `min(100cqw, calc(100cqh * ${(dw && dh) ? dw / dh : 1.5}))` }}
         onPointerDown=${onDown}>
      <img src=${beforeSrcState} alt="before"
           class="absolute inset-0 w-full h-full object-contain pointer-events-none"
           onError=${() => fallbackSrc && setBeforeSrcState(fallbackSrc)} />
      ${afterOk && afterSrc && html`
        <img src=${afterSrc} alt="after"
             class="absolute inset-0 w-full h-full object-contain pointer-events-none"
             style=${{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
             onError=${() => setAfterOk(false)} />`}
      ${afterOk && afterSrc && html`
        <div class="absolute inset-y-0 w-0.5 bg-white/80 pointer-events-none" style=${{ left: `${pos}%` }}>
          <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-0 w-6 h-6 rounded-full bg-white/90 shadow flex items-center justify-center text-black text-xs">↔</div>
        </div>`}
      <div class="absolute bottom-1 left-1 text-[10px] px-1 rounded bg-black/60 text-zinc-200">before</div>
      ${afterOk && afterSrc && html`<div class="absolute bottom-1 right-1 text-[10px] px-1 rounded bg-black/60 text-zinc-200">after</div>`}
    </div>`;
}
```

- [ ] **Step 2: Verify import resolves**

With `make serve` running, no action yet (wired in Task 17). Confirm the file parses (no console error after Task 17 import).

- [ ] **Step 3: Commit**

```bash
git add app/api/static/js/slider.js
git commit -m "feat(ui): BeforeAfterSlider wipe component"
```

---

### Task 17: DetailModal — slider view + original/enhanced/discard + keep-RAW

**Files:**
- Modify: `app/api/static/js/review.js` (DetailModal 236-466; DecisionBadge 26-31; PhotoTile badge 62; filterCounts 564-578)

**Interfaces:**
- Consumes: `BeforeAfterSlider` (Task 16); `data.before_url`, `data.after_url`, `data.preview_url`, `data.decision.export_choice`, `data.decision.keep_raw`.
- Produces: viewer shows the slider (before vs after) with a mode toggle (slider / before / after); decision controls set `export_choice` (`o`/`e`/`x`) and toggle `keep_raw` (`r`); grid badge shows the chosen version.

- [ ] **Step 1: Import + view state**

At the top of `review.js` add `import { BeforeAfterSlider } from "./slider.js";`. In `DetailModal`, replace the `showAfter`/`afterError` state with a `viewMode` state:

```javascript
  const [viewMode, setViewMode] = useState("slider"); // "slider" | "before" | "after"
```

Reset it per photo: `useEffect(() => { setViewMode("slider"); setNat(null); }, [hash]);`

- [ ] **Step 2: Replace decideAndAdvance + shortcuts**

```javascript
  const chooseAndAdvance = useCallback(async (choice) => {
    await api.decide({ photo_hash: hash, export_choice: choice });
    onMutated?.();
    onNext?.();
  }, [hash, onMutated, onNext]);
```

In `useKeyboardShortcuts`, replace the `y`/`n`/`u` entries with:

```javascript
    "o": () => chooseAndAdvance("original"),
    "e": () => chooseAndAdvance("enhanced"),
    "x": () => chooseAndAdvance("discard"),
    "r": () => mutate({ keep_raw: !data?.decision?.keep_raw }),
    "b": () => setViewMode((m) => (m === "slider" ? "before" : m === "before" ? "after" : "slider")),
```

- [ ] **Step 3: Swap the image area for the slider**

Replace the header before/after toggle (353-360) with a three-way mode toggle (`slider`/`before`/`after`), and the image container body (369-399) with:

```javascript
        ${viewMode === "slider"
          ? html`<${BeforeAfterSlider} beforeSrc=${data.before_url} afterSrc=${data.after_url}
                    fallbackSrc=${data.preview_url} dw=${dw} dh=${dh} />`
          : html`<img src=${viewMode === "after" ? data.after_url : data.before_url}
                    onError=${(e) => { if (viewMode === "before") e.target.src = data.preview_url; }}
                    alt=${data.filename ?? data.hash}
                    class="max-h-full max-w-full object-contain"
                    onLoad=${(e) => setNat({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />`}
```

(Faces overlay can stay only in `before` single mode; keep the existing `nat` sizing when `ratioReady`.)

- [ ] **Step 4: Replace the decision block (426-454)**

```javascript
        <div>
          <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">export choice</div>
          <div class="flex gap-1.5 mb-2">
            <button class="px-2 py-1 rounded ${d?.export_choice === "original" ? "bg-sky-700" : "bg-zinc-800"}"
                    onClick=${() => chooseAndAdvance("original")}>original <span class="kbd">o</span></button>
            <button class="px-2 py-1 rounded ${d?.export_choice === "enhanced" ? "bg-emerald-700" : "bg-zinc-800"}"
                    onClick=${() => chooseAndAdvance("enhanced")}>enhanced <span class="kbd">e</span></button>
            <button class="px-2 py-1 rounded ${d?.export_choice === "discard" ? "bg-rose-800" : "bg-zinc-800"}"
                    onClick=${() => chooseAndAdvance("discard")}>discard <span class="kbd">x</span></button>
          </div>
          <label class="flex items-center gap-2 text-xs mb-2">
            <input type="checkbox" checked=${!!d?.keep_raw} onChange=${() => mutate({ keep_raw: !d?.keep_raw })} />
            keep RAW <span class="kbd">r</span>
          </label>
          <div class="flex gap-1.5 mb-2">${/* stars unchanged */""}</div>
          <div class="mt-2 text-xs text-zinc-500">
            Exports the chosen version as a JPEG. Keep RAW → archived to
            <span class="font-mono">library/</span>; off → RAW deleted after export.
          </div>
        </div>
```

(Keep the stars row and the favorite/note controls above/below as they were.)

- [ ] **Step 5: DecisionBadge + PhotoTile + filterCounts**

- `DecisionBadge` (26-31): map `export_choice` → label/colour (`original`→sky, `enhanced`→emerald, `discard`→rose); return null for `undecided`.
- `filterCounts` (564-578): count `p.decision?.export_choice` buckets (`original`/`enhanced`/`discard`/`undecided`) instead of `yes`/`no`.
- `nextUndecided` (624-633): treat `export_choice` undecided/absent as undecided.

- [ ] **Step 6: Verify in browser**

With `make serve` + an enhanced batch (or a stubbed DB), open a photo: the slider wipes before↔after; `o`/`e`/`x` set the badge and advance; `r` toggles keep-RAW; grid badges update.

- [ ] **Step 7: Commit**

```bash
git add app/api/static/js/review.js
git commit -m "feat(ui): before/after slider + original/enhanced/discard + keep-RAW"
```

---

### Task 18: Toolbar/header bulk controls + Export button; app.js legs; timeline

**Files:**
- Modify: `app/api/static/js/review.js` (Toolbar 468-507; ReviewPanel handlers 675-711)
- Modify: `app/api/static/js/app.js` (DESTRUCTIVE_STAGES 19-29; autoLabel 154-156; onSubmitAndContinue 185)
- Modify: `app/api/static/js/timeline.js:60` (Review chip after `enhance`)
- Modify: `app/api/static/js/filters.js` (decision filter values)

**Interfaces:**
- Produces: header controls "use enhanced for all"/"use original for all"/"discard all" (`api.decideAll({export_choice})`) and "keep all RAWs"/"keep no RAW" (`api.decideAll({keep_raw})`); the green button becomes "Export selected (N)" → `onSubmitAndContinue` (still `api.autoRun(2)`, now = export). Only `export` is destructive in `DESTRUCTIVE_STAGES`.

- [ ] **Step 1: Toolbar + handlers (review.js)**

- Replace `onMarkAllNo` usage with the new bulk buttons; rewrite `onMarkAllNo` → e.g. `onBulk(field, value)` calling `api.decideAll({ [field]: value })`. Update the confirm text (no more "delete RAW after submit"; discard is non-destructive, keep-RAW off deletes at export).
- `submitAndContinue` confirm text → "Export N photo(s)? Runs the Export stage: writes JPEGs and applies your keep-RAW choices (RAWs with keep-RAW off are deleted after their JPEG is written)."
- Button label → `✔ Export selected (${pendingCount})`.
- The `pending` count now reflects keeper decisions (Task 12), so `pendingCount` still works.

- [ ] **Step 2: app.js**

- `DESTRUCTIVE_STAGES` (19-29): remove `submit` and `enhance` (enhance no longer deletes anything); keep only:

```javascript
const DESTRUCTIVE_STAGES = {
  export:
    "Export writes share JPEGs and applies keep-RAW: photos with keep-RAW off " +
    "have their source RAW deleted after the JPEG is written. This cannot be undone.",
};
```

- `autoLabel` (154-156): `"▶ Auto-run (ingest → enhance)"`.
- The review `onSubmitAndContinue` (185) still calls `api.autoRun(2)` (now the export leg) — no change beyond wording.

- [ ] **Step 3: timeline.js**

Line 60: change `if (s.name === "cluster")` to `if (s.name === "enhance")` so the Review pseudo-stage chip sits after Enhance.

- [ ] **Step 4: filters.js**

Update the decision filter option set from `undecided/yes/no` to `undecided/original/enhanced/discard`, and `matchesFilter` to read `p.decision?.export_choice`.

- [ ] **Step 5: Verify in browser**

`make serve`: timeline shows `… Cluster · Enhance · Review · Export`; header bulk buttons stage choices; "Export selected" triggers leg 2; only Export prompts the destructive confirm.

- [ ] **Step 6: Commit**

```bash
git add app/api/static/js/review.js app/api/static/js/app.js app/api/static/js/timeline.js app/api/static/js/filters.js
git commit -m "feat(ui): export-choice bulk controls, export leg, review-after-enhance timeline"
```

---

### Task 19: CompareModal — frame-vs-frame maps to export choices

**Files:**
- Modify: `app/api/static/js/compare.js`

**Interfaces:**
- Produces: per-card buttons set `export_choice` (`original`/`enhanced`/`discard`); "keep only this" sets the clicked frame to `enhanced` and the rest of the cluster to `discard`.

- [ ] **Step 1: Edit `compare.js`**

- `onDecide(photoHash, choice)` now posts `{ export_choice: choice }` (the parent handler in review.js `onCompareDecide` already just forwards to `api.decide`; update it to pass `export_choice`).
- Replace the card's yes/no buttons with enhanced/original/discard; "keep only this" → `onKeepOnly(hash)` which in review.js maps `keepHash → "enhanced"`, others → `"discard"`.

Update `review.js` `onCompareDecide` (653-656) and `onKeepOnly` (658-673) to use `export_choice` values accordingly.

- [ ] **Step 2: Verify in browser**

Open a multi-frame cluster → compare → picking a frame stages the choices.

- [ ] **Step 3: Commit**

```bash
git add app/api/static/js/compare.js app/api/static/js/review.js
git commit -m "feat(ui): compare modal stages export choices"
```

---

## Phase 8 — Docs, full-suite gate, and cleanup

### Task 20: Update docs (README, USER_GUIDE, CLAUDE.md)

**Files:**
- Modify: `README.md`, `USER_GUIDE.md`, `CLAUDE.md`

- [ ] **Step 1: CLAUDE.md**

- Pipeline phases list: reorder to `ingest → filter → score → cluster → enhance → [review] → export`; rewrite phase 6 (was submit) and 7/8 (enhance/export-jpeg) to describe: enhance runs on all RAW and writes before/after render JPEGs (no TIFF, no RAW moves/deletes); export applies `export_choice` + `keep_raw`.
- Decision-rules section: replace the yes/no table with the `export_choice` (`discard`/`original`/`enhanced`) + `keep_raw` model; note `rules.py`/`decide_job.py` are removed and `app/decision/export_rules.py` is the source of truth.
- Storage: note `cache/enhanced/` holds the four render JPEGs per photo; `photos/exported/` is no longer written.
- Commands: drop `make submit`; `make export-jpeg` → `make export`.
- Config: document `RAWCURATOR_REVIEW_LONG_EDGE`, `RAWCURATOR_KEEP_RAW_DEFAULT`.

- [ ] **Step 2: README.md + USER_GUIDE.md**

Update command tables and the walkthrough to the new order and the before/after review step (slider, original/enhanced/discard, keep-RAW). Remove the "run export-jpeg to see after" instruction (after is visible immediately post-enhance).

- [ ] **Step 3: Commit**

```bash
git add README.md USER_GUIDE.md CLAUDE.md
git commit -m "docs: before/after export workflow"
```

---

### Task 21: Full-suite gate — lint, types, tests

**Files:** none (verification only; fix fallout inline).

- [ ] **Step 1: Grep for stragglers**

Run: `grep -rn "export-jpeg\|export_jpeg\|run_jpeg_export\|may_delete_source\|decision.rules\|apply_decisions\|ENHANCE_ACTIONS\|\.selected\b\|enhanced_url\|jpeg_url" app/ tests/`
Expected: only intended references remain (e.g. `_urls.jpeg_url` fully removed; `Decision.selected` only as a dead ORM column, not read in app code). Fix any live references (likely: `app/api/main.py:59-65` `/jpeg` static mount comment/existence — keep the mount, update the comment; `tests/test_pipeline_routes.py`, `tests/test_orchestrator_autorun.py`, `tests/test_static_headers.py` may reference stage names).

- [ ] **Step 2: Run lint + types + tests**

Run:
```bash
make lint
make typecheck
make test
```
Expected: all pass. Fix fallout in the owning module and re-run.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint/type/test fixups for export-choice workflow"
```

---

## Notes for the implementer

- **Cross-task breakage window:** Tasks 3-11 rewire importers of the deleted `rules.py`/`decide_job.py` and the renamed stages. The full suite (`make test`) only goes green after Task 11; earlier per-task gates should run just the task's own test file. Task 21 is the whole-suite gate.
- **`/jpeg` static mount** (`app/api/main.py:57-65`) stays — it serves the final `photos/jpeg/` outputs. Only its docstring (which mentions export-jpeg) needs a wording touch (Task 21).
- **`test_pipeline_routes.py` / `test_orchestrator_autorun.py`** assert stage names/leg wiring; update them to the new `STAGES` when Task 9/21 surfaces the failures.
- **Disk:** the four render JPEGs per photo are far smaller than the old 16-bit `exported/*.tif` masters, so overall disk use drops; they are cleaned by `make reset` (Task 14).
- **`scripts/contact_sheet.py`** reads the enhanced master from `photos/exported/*.tif`, which is no longer written. It is a manual owner-only look-review tool (not in CI). In Task 20, either point it at `cache/enhanced/<hash>.after.full.jpg` (the new enhanced render) or add a one-line note that it is stale pending a rework; do not block the feature on it.
- **Task 21 grep scope:** run the straggler grep across the repo root too (`Makefile`, `compose.yaml`, `scripts/`, `db/`), not just `app/`/`tests/`, to catch any lingering `submit`/`export-jpeg` references. `compose.yaml` was checked during planning and needs no change.
