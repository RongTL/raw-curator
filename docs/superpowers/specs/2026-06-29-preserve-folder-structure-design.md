# Design: Preserve recursive folder structure across pipeline outputs

**Date:** 2026-06-29
**Status:** Approved

## Problem

The ingest walker recurses into `photos/incoming/` (`app/ingest/walker.py` uses
`rglob`), so photos may live in arbitrary subfolders such as
`incoming/2025/wedding/IMG_1234.CR3`. But every downstream phase **flattens** to
the basename when it writes output:

| Phase  | Site                       | Current code                                   |
|--------|----------------------------|------------------------------------------------|
| submit | `app/decision/decide_job.py:60` | `settings.photos / rule.library_subdir / src.name` |
| enhance| `app/enhancement/enhance_job.py:184` | `settings.photos / "exported" / (src.stem + ".tif")` |
| export | `app/export/jpeg_job.py:64` | `settings.photos / settings.jpeg_subdir / (src.stem + ".jpg")` |

The export candidate scan is also flat (`lib.iterdir()` / `exp.iterdir()` at
`app/export/jpeg_job.py:52,58`), so it would not even find files placed in
subfolders.

## Goal

Mirror the relative subtree found under `incoming/` into **every** output
folder, identically:

```
incoming/2025/wedding/IMG_1234.CR3
  -> library/2025/wedding/IMG_1234.CR3   (submit, yes)
  -> exported/2025/wedding/IMG_1234.tif  (enhance)
  -> jpeg/2025/wedding/IMG_1234.jpg      (export-jpeg)
```

## Approach (chosen: A — derive from path, no DB change)

A single helper derives the relative subpath of a source file against whichever
known base root it currently lives under. Each flatten site joins that subpath
onto its destination root instead of using `.name` / `.stem`. No new DB column,
no migration, no config change — this fits the project's "avoid migrations"
convention and keeps the export phase filesystem-driven.

The design is self-consistent because **submit** carries the subtree into
`library/`, so the later phases re-derive the same relative path from
`library/` (or from `incoming/` on the `enhance_only` path, or from `exported/`
for the TIFF→JPEG step).

### New module: `app/paths.py`

```python
SOURCE_ROOTS = ("incoming", "library", "exported")

def relative_subpath(src: Path, photos: Path, roots=SOURCE_ROOTS) -> Path:
    """Return src relative to whichever known root it lives under,
    filename included.

        incoming/2025/wedding/IMG.CR3 -> 2025/wedding/IMG.CR3

    Falls back to Path(src.name) (+ a log.warning) when src is under
    none of the known roots."""
```

Destinations that change format use `Path.with_suffix(".tif" | ".jpg")` on the
returned relative path.

### Call-site changes

1. **`app/decision/decide_job.py:60`**
   ```python
   dst = settings.photos / rule.library_subdir / relative_subpath(src, settings.photos)
   ```
   `src` is under `incoming/`; subtree is carried into `library/`. The executor
   already does `dst.parent.mkdir(parents=True, exist_ok=True)`
   (`app/decision/executor.py:20`), so nested dirs are created.

2. **`app/enhancement/enhance_job.py:184`**
   ```python
   out = settings.photos / "exported" / relative_subpath(src, settings.photos).with_suffix(".tif")
   ```
   `src` is under `incoming/` (enhance_only) or `library/` (keep_and_enhance);
   both are known roots. `write_tiff16` already mkdirs its parent.

3. **`app/export/jpeg_job.py:63` (`_dest_for`)**
   ```python
   return settings.photos / settings.jpeg_subdir / relative_subpath(src, settings.photos).with_suffix(".jpg")
   ```
   `src` is under `library/` or `exported/`.

4. **`app/export/jpeg_job.py:40` (`_list_candidates`)** — switch the flat
   `lib.iterdir()` / `exp.iterdir()` scans to recursive `rglob("*")` so files in
   subfolders are discovered. Keep the `is_file()` + `is_convertible()` filters.

5. **`app/export/jpeg_job.py:19` (`_convert_one` worker)** — add
   `Path(dest_str).parent.mkdir(parents=True, exist_ok=True)` before converting.
   The single top-level `out_dir.mkdir` (`jpeg_job.py:86`) no longer covers the
   per-subfolder destinations.

## Edge cases & error handling

- **Source under no known root** (e.g. a manually placed file): fall back to
  flat `Path(src.name)` and emit a `log.warning`, matching the existing
  skip-with-warning convention used for non-RAW sources.
- **Filename collisions**: become *less* likely because subfolders disambiguate.
  The executor's `FileExistsError` guard (`executor.py:21-22`) and export's
  `dest.exists()` skip (`jpeg_job.py:92`) still apply unchanged.
- **Files at the incoming root** (no subfolder): `relative_subpath` returns just
  `IMG.CR3`, so the file lands directly in `library/IMG.CR3` — identical to
  today's behavior.

## Out of scope / non-changes

- No DB schema, model, or Alembic migration changes.
- No config (`app/config.py`) changes.
- No change to ingest, filter, score, or cluster phases.
- Routing rules (`app/decision/rules.py`) are untouched.

## Testing

Pure-function focus — no GPU or RAW fixtures required.

- Unit tests for `relative_subpath`:
  - nested under `incoming/`, `library/`, `exported/`
  - file directly at a root (no subfolder)
  - file under an unknown root → fallback to basename + warning
  - format swap via `.with_suffix()` yields the expected `.tif` / `.jpg` path
- Update any existing `decide` / export tests that assert flat destinations to
  expect the mirrored tree; add at least one nested-input case per phase.

## Files touched

- **New:** `app/paths.py`
- **Edit:** `app/decision/decide_job.py`, `app/enhancement/enhance_job.py`,
  `app/export/jpeg_job.py`
- **Tests:** new `tests/test_paths.py`; update affected decide/export tests.
