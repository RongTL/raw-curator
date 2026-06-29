# Preserve Recursive Folder Structure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror each photo's subfolder path under `incoming/` into the `library/`, `exported/`, and `jpeg/` output folders instead of flattening to the basename.

**Architecture:** One pure helper, `relative_subpath(src, photos)`, derives a file's path relative to whichever known base root (`incoming`/`library`/`exported`) it lives under. The three flatten sites (submit, enhance, export-jpeg) join that relative path onto their destination root, using `Path.with_suffix()` for format changes. No DB, schema, migration, or config changes.

**Tech Stack:** Python 3.12, `pathlib`, pytest, ruff (line-length 100, ignore E501), mypy --strict. Tests run in-container.

**Spec:** `docs/superpowers/specs/2026-06-29-preserve-folder-structure-design.md`

**Test execution note:** Project deps are not installed on the host; tests run inside the Podman container. Single-file run: `podman-compose run --rm app pytest tests/test_paths.py -v`. Full suite: `make test`. `test_paths.py` is pure `pathlib` (no Torch/CUDA/numpy), so it is fast even in-container.

---

## File Structure

- **Create `app/paths.py`** — the single `relative_subpath` helper + `SOURCE_ROOTS` constant. One responsibility: map an absolute source path to its relative subpath under a known root. No imports from other `app/` modules (leaf utility), so every phase can use it without cycles.
- **Create `tests/test_paths.py`** — unit tests for `relative_subpath` (the logic core).
- **Modify `app/decision/decide_job.py`** — line 60, submit destination.
- **Modify `app/enhancement/enhance_job.py`** — line 184, enhance TIFF destination.
- **Modify `app/export/jpeg_job.py`** — `_dest_for` (line 63), `_list_candidates` scans (lines 48-59), `_convert_one` worker (line 19).
- **Modify `tests/test_jpeg_export.py`** — add two new tests for subfolder behavior (leave existing exact-match tests untouched; they use out-of-root paths / flat dirs and stay green via the fallback / identical rglob results).

---

## Task 1: `relative_subpath` helper (the logic core)

**Files:**
- Create: `app/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paths.py`:

```python
"""Unit tests for relative_subpath path-mirroring helper."""

from __future__ import annotations

import logging
from pathlib import Path

from app.paths import relative_subpath


def test_nested_under_incoming() -> None:
    photos = Path("/data/photos")
    src = photos / "incoming" / "2025" / "wedding" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("2025/wedding/IMG_1234.CR3")


def test_nested_under_library() -> None:
    photos = Path("/data/photos")
    src = photos / "library" / "2025" / "wedding" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("2025/wedding/IMG_1234.CR3")


def test_nested_under_exported() -> None:
    photos = Path("/data/photos")
    src = photos / "exported" / "a" / "b" / "IMG.tif"
    assert relative_subpath(src, photos) == Path("a/b/IMG.tif")


def test_file_directly_at_root_has_no_subfolder() -> None:
    photos = Path("/data/photos")
    src = photos / "incoming" / "IMG_1234.CR3"
    assert relative_subpath(src, photos) == Path("IMG_1234.CR3")


def test_unknown_root_falls_back_to_basename_and_warns(caplog) -> None:
    photos = Path("/data/photos")
    src = Path("/somewhere/else/IMG_9999.CR3")
    with caplog.at_level(logging.WARNING):
        result = relative_subpath(src, photos)
    assert result == Path("IMG_9999.CR3")
    assert any("IMG_9999.CR3" in r.getMessage() for r in caplog.records)


def test_suffix_swap_builds_expected_output_path() -> None:
    photos = Path("/data/photos")
    src = photos / "library" / "2025" / "trip" / "IMG.CR3"
    rel = relative_subpath(src, photos)
    assert (photos / "exported" / rel.with_suffix(".tif")) == (
        photos / "exported" / "2025" / "trip" / "IMG.tif"
    )
    assert (photos / "jpeg" / rel.with_suffix(".jpg")) == (
        photos / "jpeg" / "2025" / "trip" / "IMG.jpg"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `podman-compose run --rm app pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.paths'`.

- [ ] **Step 3: Write the minimal implementation**

Create `app/paths.py`:

```python
"""Map an absolute source path to its path relative to a known photos root.

Used by submit / enhance / export to mirror the subfolder layout found under
photos/incoming/ into the library/, exported/, and jpeg/ output trees.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE_ROOTS: tuple[str, ...] = ("incoming", "library", "exported")


def relative_subpath(
    src: Path, photos: Path, roots: tuple[str, ...] = SOURCE_ROOTS
) -> Path:
    """Return ``src`` relative to whichever known root it lives under.

    The returned path includes the filename, e.g.::

        photos/incoming/2025/wedding/IMG.CR3 -> 2025/wedding/IMG.CR3

    Falls back to ``Path(src.name)`` (and logs a warning) when ``src`` is under
    none of the known roots, so callers always get a usable relative path.
    """
    for root in roots:
        base = photos / root
        try:
            return src.relative_to(base)
        except ValueError:
            continue
    log.warning(
        "source %s is under no known root %s; flattening to basename",
        src,
        roots,
    )
    return Path(src.name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `podman-compose run --rm app pytest tests/test_paths.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and type-check the new module**

Run: `podman-compose run --rm app ruff check app/paths.py tests/test_paths.py`
Run: `podman-compose run --rm app mypy app/paths.py`
Expected: both clean (no errors).

- [ ] **Step 6: Commit**

```bash
git add app/paths.py tests/test_paths.py
git commit -m "feat(paths): add relative_subpath helper for folder mirroring"
```

---

## Task 2: Mirror subtree at submit (`decide_job.py`)

**Files:**
- Modify: `app/decision/decide_job.py:60`

Rationale for no new test here: `apply_decisions` has no existing unit harness (it needs a live SQLite session + on-disk files), and the change is a one-line substitution of the already-unit-tested `relative_subpath`. Adding a DB+filesystem integration test would be disproportionate. Coverage comes from `tests/test_paths.py`.

- [ ] **Step 1: Add the import**

In `app/decision/decide_job.py`, the import block currently ends at line 25 with `from app.models import Decision, Photo`. Add directly below it:

```python
from app.paths import relative_subpath
```

- [ ] **Step 2: Replace the flattening line**

Replace line 60:

```python
                dst = settings.photos / rule.library_subdir / src.name
```

with:

```python
                dst = (
                    settings.photos
                    / rule.library_subdir
                    / relative_subpath(src, settings.photos)
                )
```

(`src` is under `incoming/`; the executor already does `dst.parent.mkdir(parents=True, exist_ok=True)` at `app/decision/executor.py:20`, so nested destination dirs are created.)

- [ ] **Step 3: Verify decision tests still pass**

Run: `podman-compose run --rm app pytest tests/test_decision_rules.py -v`
Expected: PASS (existing tests unaffected — they test `resolve`/`tier_from_scores`, not path construction).

- [ ] **Step 4: Lint and type-check**

Run: `podman-compose run --rm app ruff check app/decision/decide_job.py`
Run: `podman-compose run --rm app mypy app/decision/decide_job.py`
Expected: both clean.

- [ ] **Step 5: Commit**

```bash
git add app/decision/decide_job.py
git commit -m "feat(submit): mirror incoming subtree into library/"
```

---

## Task 3: Mirror subtree at enhance (`enhance_job.py`)

**Files:**
- Modify: `app/enhancement/enhance_job.py:184`

Rationale for no new test here: `enhance_job` requires darktable + CUDA and has no host-runnable unit harness; the change is a one-line substitution of the unit-tested `relative_subpath`. Coverage comes from `tests/test_paths.py`.

- [ ] **Step 1: Add the import**

In `app/enhancement/enhance_job.py`, add to the `app` import group (alongside the existing `from app.config import settings`):

```python
from app.paths import relative_subpath
```

- [ ] **Step 2: Replace the flattening line**

Replace line 184:

```python
    out = settings.photos / "exported" / (src.stem + ".tif")
```

with:

```python
    out = (
        settings.photos
        / "exported"
        / relative_subpath(src, settings.photos).with_suffix(".tif")
    )
```

(`src` is under `incoming/` on the enhance_only path or `library/` on the keep_and_enhance path — both known roots. `write_tiff16` already mkdirs the parent at `app/enhancement/pack_tiff.py`.)

- [ ] **Step 3: Lint and type-check**

Run: `podman-compose run --rm app ruff check app/enhancement/enhance_job.py`
Run: `podman-compose run --rm app mypy app/enhancement/enhance_job.py`
Expected: both clean.

- [ ] **Step 4: Commit**

```bash
git add app/enhancement/enhance_job.py
git commit -m "feat(enhance): mirror subtree into exported/ TIFFs"
```

---

## Task 4: Mirror subtree at export-jpeg + recurse the scan (`jpeg_job.py`)

**Files:**
- Modify: `app/export/jpeg_job.py` (`_convert_one` lines 19-37, `_list_candidates` lines 48-59, `_dest_for` lines 63-64)
- Test: `tests/test_jpeg_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_jpeg_export.py`:

```python
def test_dest_for_preserves_subfolders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    monkeypatch.setattr(jpeg_job.settings, "jpeg_subdir", "jpeg")
    raw = tmp_path / "library" / "2025" / "wedding" / "IMG_0001.CR3"
    tif = tmp_path / "exported" / "2025" / "wedding" / "IMG_0001.tif"
    assert jpeg_job._dest_for(raw) == tmp_path / "jpeg" / "2025" / "wedding" / "IMG_0001.jpg"
    assert jpeg_job._dest_for(tif) == tmp_path / "jpeg" / "2025" / "wedding" / "IMG_0001.jpg"


def test_list_candidates_recurses_subfolders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "library" / "2025" / "trip"
    nested.mkdir(parents=True)
    raw = nested / "A.CR3"
    raw.write_bytes(b"\x00")
    monkeypatch.setattr(jpeg_job.settings, "photos", tmp_path)
    assert jpeg_job._list_candidates("library") == [raw]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `podman-compose run --rm app pytest tests/test_jpeg_export.py::test_dest_for_preserves_subfolders tests/test_jpeg_export.py::test_list_candidates_recurses_subfolders -v`
Expected: both FAIL — `_dest_for` still uses `src.stem` (drops subfolders); `_list_candidates` uses `iterdir()` (does not recurse, so the nested file is not found).

- [ ] **Step 3: Add the import**

In `app/export/jpeg_job.py`, add below `from app.export.jpeg_writer import convert_image_to_jpeg, is_convertible` (line 13):

```python
from app.paths import relative_subpath
```

- [ ] **Step 4: Make the candidate scan recursive**

Replace `_list_candidates` body lines 48-59. Change both `sorted(lib.iterdir())` and `sorted(exp.iterdir())` to `sorted(lib.rglob("*"))` / `sorted(exp.rglob("*"))`:

```python
    if source in {"library", "all"}:
        lib = photos / "library"
        if lib.is_dir():
            items.extend(
                p for p in sorted(lib.rglob("*")) if p.is_file() and is_convertible(p)
            )
    if source in {"exported", "all"}:
        exp = photos / "exported"
        if exp.is_dir():
            items.extend(
                p for p in sorted(exp.rglob("*")) if p.is_file() and is_convertible(p)
            )
```

- [ ] **Step 5: Mirror subfolders in `_dest_for`**

Replace lines 63-64:

```python
def _dest_for(src: Path) -> Path:
    return settings.photos / settings.jpeg_subdir / (src.stem + ".jpg")
```

with:

```python
def _dest_for(src: Path) -> Path:
    rel = relative_subpath(src, settings.photos).with_suffix(".jpg")
    return settings.photos / settings.jpeg_subdir / rel
```

- [ ] **Step 6: Create the per-destination subfolder in the worker**

In `_convert_one`, add a `mkdir` just before the `convert_image_to_jpeg` call (the single top-level `out_dir.mkdir` at line 86 no longer covers per-subfolder paths). Replace the `try:` block opening:

```python
    try:
        convert_image_to_jpeg(
```

with:

```python
    try:
        Path(dest_str).parent.mkdir(parents=True, exist_ok=True)
        convert_image_to_jpeg(
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `podman-compose run --rm app pytest tests/test_jpeg_export.py::test_dest_for_preserves_subfolders tests/test_jpeg_export.py::test_list_candidates_recurses_subfolders -v`
Expected: both PASS.

- [ ] **Step 8: Run the whole jpeg-export test file to confirm no regressions**

Run: `podman-compose run --rm app pytest tests/test_jpeg_export.py -v`
Expected: PASS — existing `test_dest_for_uses_configured_subdir` stays green (its inputs are outside all roots, so `relative_subpath` falls back to the basename → same `IMG_0001.jpg`), and `test_list_candidates_partitions_by_source` stays green (flat dirs return the same files under `rglob`).

- [ ] **Step 9: Lint and type-check**

Run: `podman-compose run --rm app ruff check app/export/jpeg_job.py tests/test_jpeg_export.py`
Run: `podman-compose run --rm app mypy app/export/jpeg_job.py`
Expected: both clean.

- [ ] **Step 10: Commit**

```bash
git add app/export/jpeg_job.py tests/test_jpeg_export.py
git commit -m "feat(export): mirror subtree into jpeg/ and recurse candidate scan"
```

---

## Task 5: Full-suite verification

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: PASS (no regressions across the suite). GPU-marked and `real_raw`-marked tests remain skipped unless their env/fixtures are present — that is expected.

- [ ] **Step 2: Lint and type-check the whole touched surface**

Run: `make lint`
Run: `make typecheck`
Expected: both clean.

- [ ] **Step 3: Update CLAUDE.md storage note (docs)**

In `CLAUDE.md`, under **### Storage**, add a short line documenting the new behavior:

```markdown
- Output folders mirror the subfolder layout found under `photos/incoming/`:
  a file at `incoming/<sub>/X.CR3` lands at `library/<sub>/X.CR3`,
  `exported/<sub>/X.tif`, and `jpeg/<sub>/X.jpg`. Path mirroring is centralized
  in `app/paths.py::relative_subpath`; the submit/enhance/export phases all
  derive their destination from it.
```

- [ ] **Step 4: Commit the docs**

```bash
git add CLAUDE.md
git commit -m "docs: note output folder mirroring of incoming/ subtree"
```

---

## Self-Review

- **Spec coverage:**
  - Mirror full subtree everywhere → Tasks 2 (library), 3 (exported), 4 (jpeg). ✓
  - New `app/paths.py` helper with fallback + warning → Task 1. ✓
  - Recursive candidate scan + per-dest mkdir in export → Task 4 steps 4 & 6. ✓
  - "Files at incoming root" / "source under no known root" edge cases → Task 1 tests `test_file_directly_at_root_has_no_subfolder` and `test_unknown_root_falls_back_to_basename_and_warns`. ✓
  - No DB/schema/migration/config change → none of the tasks touch `app/models.py`, `db/migrations/`, or `app/config.py`. ✓
- **Placeholder scan:** no TBD/TODO; every code step shows full code. ✓
- **Type/name consistency:** `relative_subpath(src, photos, roots=SOURCE_ROOTS) -> Path` is defined in Task 1 and called identically in Tasks 2, 3, 4. `.with_suffix(".tif"|".jpg")` applied only where the output format differs (enhance, export), not at submit (RAW stays RAW). ✓
```
