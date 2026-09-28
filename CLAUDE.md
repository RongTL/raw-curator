# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

`raw-curator` is an **ephemeral, single-batch** AI photo curation pipeline. The whole thing runs inside a Podman container on a GPU host (Ubuntu + NVIDIA driver + NVIDIA Container Toolkit). The host stays clean; the container holds CUDA, Torch, darktable, exiftool, and all model weights.

There is no long-lived service. Each batch flows through the pipeline once, the user reviews in a web UI, decisions are applied to disk, then `make reset` wipes everything (DB + cache + working dirs) and the next batch starts fresh. `models/` and `xmp/` are the only persistent dirs.

See `README.md` for the user-facing overview and `USER_GUIDE.md` for the end-to-end walkthrough.

## Common commands

Everything is invoked through `make` (which delegates to `podman-compose` → the in-container `raw-curator` Typer CLI). Run from the repo root:

```bash
make image            # build raw-curator:latest from scratch (~12 GB, downloads ~5 GB of wheels)
make image-warm       # rebuild reusing packages from the current image (minutes; use after code/dev-dep edits)
make download-models  # fetch ~17 GB of weights into models/
make reset            # wipe DB + cache + working photo dirs; runs alembic upgrade head

# Per-stage (or `make run` for ingest→filter→score→cluster in one shot)
make ingest filter score cluster
make enhance          # develop before/after render JPEGs into cache/enhanced/ for every RAW
make serve            # FastAPI + UI on :8080 (before/after review + export choices)
make export           # apply export choices: chosen JPEG → photos/jpeg/, then RAW retention

# Dev loop (app/, tests/ and pyproject.toml are bind-mounted from the working
# tree, so these run against your edits without rebuilding the image)
make test             # pytest -q inside the container
make lint             # ruff check + ruff format --check (app/ tests/ scripts/)
make format           # ruff format (rewrites files)
make typecheck        # mypy app/
make shell            # bash inside the app container
```

The web UI can drive the whole pipeline (ingest → export, plus New-batch
reset), so the per-stage make targets are optional; `make image` and
`make download-models` remain host-side prerequisites.

Run a single test (from inside the container or via `podman-compose run --rm app`):

```bash
pytest tests/test_export_rules.py -q
pytest tests/test_export_rules.py::test_is_keeper -q
```

GPU-marked tests are skipped unless `RUN_GPU_TESTS=1`. `real_raw`-marked tests are skipped unless `tests/data/` contains RAW fixtures.

The CLI is also reachable directly inside the container: `raw-curator ingest|filter|score|cluster|enhance|export|serve|run --auto|info|reset`.

## Architecture

### Pipeline phases (each is a `make` target and an `app/<phase>/<phase>_job.py` entrypoint)

1. **ingest** (`app/ingest/`) — walks `photos/incoming/`, computes xxh3 hash, extracts EXIF via `exiftool`, decodes RAW via `rawpy` (and HEIC via `pillow-heif`), writes a 512 px thumb + 3000 px preview JPEG into `cache/`. Records `file_kind` (`raw`/`jpeg`/`tiff`/`heic`/`png`) per photo — downstream stages branch on this.
2. **filter** (`app/filters/`) — CPU-only Laplacian blur variance, pHash/dHash, exposure histogram flags.
3. **score** (`app/scoring/` + `app/embedding/`) — GPU stage. CLIP ViT-L/14 + aesthetic-predictor v2.5 + MUSIQ + MANIQA + InsightFace. Runs **stage-by-stage** (`stage=clip|iqa|faces|all`), freeing CUDA between stages so it fits a 6 GB RTX 2060.
4. **cluster** (`app/clustering/`) — EXIF burst grouping → CLIP HDBSCAN over the not-yet-clustered rest → one recommended photo per cluster (`recommend.rank`). pHash/dHash are computed in filter and exposed in the API but do not drive clustering.
5. **enhance** (`app/enhancement/`, orchestrated step-major over the whole batch in `batch.py::run_batch`) — GPU stage; now runs **before review**, on **every RAW photo** (`enhance_job._candidates` selects `Photo.file_kind == "raw"` and reads no `Decision`), so the before/after comparison exists the moment review starts. Per frame: darktable develops the RAW (sigmoid workflow, `--icc-type LIN_REC2020` → 16-bit linear Rec.2020 TIFF, honouring the user's sidecar via `resolve_xmp`) → per-frame lens distortion + CA correction from EXIF (lensfun via `classical/lens_correct.py`, gated on `enhance_lens_correction`; no-op for lenses lensfun can't resolve) → measure + plan (the rule table in `engine/decision.py` builds one per-photo recipe, persisted to `quality_reports.plan_json`) → classical pre-AI steps in float32 → AI steps run step-major (one model resident at a time across the whole batch), each merged back as a *delta* into the float master (`engine/runner.py::apply_ai_delta`, so the master is never quantised to 8-bit) → resize to `RAWCURATOR_ENHANCE_TARGET_RES` when AI ran → post-AI steps → verify (`verify.py`; a degraded verdict triggers one safe-plan retry, then the better of the two by Q score is kept). **What it writes**: no `photos/exported/*.tif` master anymore — instead four display-referred JPEGs per photo into `cache/enhanced/` via `render_jpeg.py::write_render`: `<hash>.before.jpg` / `<hash>.after.jpg` (review-res, `RAWCURATOR_REVIEW_LONG_EDGE` long edge) and `<hash>.before.full.jpg` / `<hash>.after.full.jpg` (full-res, EXIF copied from source, what `export` copies). `before` is the developed original (darktable + lens/CA, before pre-AI/AI/post-AI); `after` is the enhanced result. Enhance **never moves or deletes a RAW** and no longer reads any decision. **RAW-only**; non-RAW sources are skipped with a warning (original-only in review) because the AI chain expects sensor data, not 8-bit display-referred pixels.
6. **serve** (`app/api/`) — FastAPI app exposing `/api/{queue,photo,cluster,decide}` plus the static SPA from `app/api/static/`. The UI is plain HTML/JS using CDN-hosted React (no Node build step). Review now happens **after enhance**: a single before/after screen with a slider (developed original vs enhanced, `app/api/static/js/slider.js`) and a per-photo version pick — `original` / `enhanced` / `discard` — plus an independent keep-RAW checkbox. Both are staged in the `decisions` table; nothing on disk moves until **Export**. Detail-modal keys: `o`/`e`/`x` (original/enhanced/discard), `r` (toggle keep-RAW), `b` (cycle slider/before/after), `.` (next undecided). The review header exposes bulk controls backed by `POST /api/decide/all` — "use enhanced/original for all", "discard all", "keep all/no RAW" — staging every non-applied row in one transaction via `app/decision/bulk.py::stage_all`; per-cluster keep/reject uses `stage_cluster`.
   `serve` is the primary entry point: the UI (pipeline timeline + context panel) can run every stage itself. Pieces:
   `app/orchestrator/` (subprocess-per-stage `JobRunner` with a single global
   job slot, two-leg `AutoRun`: leg 1 ingest→…→enhance, then leg 2 export after
   the review's "Export selected" button, DB-derived progress in `progress.py`),
   `app/monitor/stats.py` (psutil + NVML sampler, 60 s rolling window, disk
   warnings), and routes `/api/pipeline/*` + `/api/system/stats`. The Review
   pseudo-stage chip sits after `enhance` in the timeline. The frontend
   is ES modules under `app/api/static/js/` (CDN React, no build step).
7. **export** (`app/export/`, `export_job.py::run_export`) — replaces the old `submit` **and** `export-jpeg`. Driven by the DB, not the filesystem: for each `Decision` with `export_choice IN ('original', 'enhanced')` and `applied == 0`, it encodes exactly one share JPEG into `photos/jpeg/` — `enhanced` → the cached `<hash>.after.full.jpg`; `original` → the cached `<hash>.before.full.jpg` (or the source file itself for a non-RAW). Then it applies RAW retention **after** the JPEG exists on disk: `keep_raw` moves the RAW into `photos/library/` (via `executor.apply_moves`, `Photo.source_path` updated), else the RAW is deleted. `discard`/`undecided` produce nothing and touch nothing; `applied` gates re-runs. EXIF copied from source; orientation baked in.

### Decision rules (`app/decision/export_rules.py`)

The curator makes one choice **per photo, after enhancement**, having seen both versions on the slider. The old binary `yes`/`no` keep-RAW routing is retired: `app/decision/rules.py` and `app/decision/decide_job.py` were **deleted**; `export_rules.py` is the source of truth. Two independent, staged fields on `Decision` (`app/models.py`):

- `export_choice ∈ {undecided, discard, original, enhanced}` — the version pick (`EXPORT_CHOICES`); `is_keeper` = `original`/`enhanced`.
- `keep_raw: bool` (default `RAWCURATOR_KEEP_RAW_DEFAULT`, true) — archive the source RAW or not; only meaningful for kept photos.

| `export_choice` | Final JPEG in `photos/jpeg/`                              | keep_raw = true              | keep_raw = false                     |
|-----------------|----------------------------------------------------------|------------------------------|--------------------------------------|
| `enhanced`      | from the cached `<hash>.after.full.jpg`                  | RAW → `photos/library/`      | RAW deleted **after** the JPEG lands |
| `original`      | from the cached `<hash>.before.full.jpg` (or the source, for a non-RAW) | RAW → `photos/library/` | RAW deleted **after** the JPEG lands |
| `discard`       | none                                                     | RAW left untouched (e.g. `incoming/`) | RAW left untouched          |
| `undecided`     | none (skipped, reported)                                 | untouched                    | untouched                            |

`enhance` runs on every RAW regardless of any decision; only `export` reads `export_choice`/`keep_raw` (via `export_rules.is_keeper`). The RAW deletion for `keep_raw = false` is intentional and **irreversible**, but happens only after the output JPEG exists on disk, so a failed export preserves the original; `discard` is non-destructive. Do not re-introduce a pre-enhance cull or score-tier routing without confirming with the user.

The technical/aesthetic blend (`0.6 * technical + 0.4 * normalized_aesthetic`, threshold `0.55`) lives once in `app/scoring/combined.py` (`combined_score`) and still drives in-cluster ranking (`app/clustering/recommend.py`) and the display tier — but clustering is now purely a review aid and gates nothing. `Decision.selected`, `score_tier`, `action`, and `enhance_requested` are dead columns kept in the schema (no migration to drop them) and are no longer read or written by the routing path.

### Storage

- **SQLite + sqlite-vec, WAL mode** at `cache/session.db`. Engine setup in `app/db.py` loads the `sqlite-vec` extension on every connection. Schema in `app/models.py` (SQLAlchemy 2.0 declarative). Migrations via Alembic in `db/migrations/`.
- `photo_embeddings.vec` is a raw `LargeBinary` blob of float16 CLIP features (768-dim by default).
- All paths stored in the DB are container-side absolute paths under `/data/`.
- `cache/enhanced/` (`settings.enhanced_dir`) holds the four render JPEGs `enhance` writes per RAW — `<hash>.{before,after}.jpg` (review-res) + `<hash>.{before,after}.full.jpg` (full-res); it is wiped by `make reset`. `photos/exported/` is **no longer written** (no 16-bit TIFF master anymore, though `reset` still wipes it for safety). The only kept artifacts per photo are the `photos/jpeg/` share JPEG (for kept photos) and, when `keep_raw` is set, the RAW in `photos/library/`.
- Output folders mirror the subfolder layout found under `photos/incoming/`: a file at `incoming/<sub>/X.CR3` lands at `library/<sub>/X.CR3` and `jpeg/<sub>/X.jpg`. Path mirroring is centralized in `app/paths.py::relative_subpath`; the enhance/export phases derive their destination from it (against whichever of `incoming`/`library` the source currently lives under).

### Configuration

All knobs are env vars with prefix `RAWCURATOR_`, loaded via pydantic-settings in `app/config.py`. The container reads them from `.env` (which `compose.yaml` injects via `env_file`, not just `${VAR}` interpolation — see commit `f098c14`). Bind-mount points are fixed: `/data/{photos,cache,models,xmp}`.

Every setting must have a reader: do not add a field to `Settings` (or a row to `.env.example`/README) without code that uses it, and do not read `os.environ` directly — `denoise.py`/`face_restore.py` used to hide knobs that way. Model weight locations are resolved through `app/enhancement/weights.py` (built on `settings.models`); `scripts/download_models.py` shares the same constants. `RAWCURATOR_LOG_LEVEL` sets the root logger level installed by `app/logging_setup.py` from the Typer callback in `app/cli.py`.

Two knobs added for the before/after review + export: `RAWCURATOR_REVIEW_LONG_EDGE` (default `3000`) sizes the review-res `before`/`after` JPEGs `enhance` renders (read in `app/enhancement/batch.py` and passed to `render_jpeg.write_render`); `RAWCURATOR_KEEP_RAW_DEFAULT` (default `true`) is the initial `keep_raw` value stamped on a freshly-created `Decision` row (read in `decide.py`/`bulk.py`). The final share JPEG still obeys `RAWCURATOR_JPEG_QUALITY` / `_LONG_EDGE` / `_PROGRESSIVE`.

VRAM-sensitive defaults are tuned for a 6 GB RTX 2060:
- `RAWCURATOR_ENHANCE_AI_SCALE=1.0` — no pre-AI downscale; AI sees native pixels. Drop to `0.85`/`0.7`/`0.5` if OOM.
- `RAWCURATOR_ENHANCE_TARGET_RES=200%` — keep Real-ESRGAN's x2 output (e.g. 12kx8k for 24 MP source). Set to `native` to downsample back; `200%` roughly quadruples the enhanced-render pixel count (larger `cache/enhanced/` `after.full` JPEGs and, for `enhanced` picks, the final share JPEG). Real-ESRGAN is *planned* only when the target enlarges the source (`200%`/explicit > native) or when the source long edge is below `RAWCURATOR_ENHANCE_SR_MIN_LONG_EDGE` (default `3000`), so at `native` a 24 MP source gets no super-resolution at all.
- `RAWCURATOR_CLIP_BATCH=8` — lower to `4` on OOM during scoring.
- `RAWCURATOR_SCUNET_TILE=512` / `_TILE_PAD=32` and `RAWCURATOR_CODEFORMER_MAX_LONG_EDGE=2048` — tile/long-edge caps for the two steps that otherwise run on the full frame.
- `RAWCURATOR_ENHANCE_DENOISE` / `_FACE_RESTORE` / `_BACKLIT_RECOVERY` — on/off switches read by `plan_from_report`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in `compose.yaml` to reduce fragmentation across the SCUNet → Real-ESRGAN → CodeFormer stages.

### Container topology

`compose.yaml` defines two services backed by the same `raw-curator:latest` image (shared block is the `x-raw-curator` YAML anchor):
- `app` — one-shot CLI commands (`make ingest`, `make score`, etc. each `run --rm app raw-curator <cmd>`).
- `ui` — long-running `raw-curator serve --host 0.0.0.0 --port 8080`.

The image includes `tests/` and the dev dependency group, so `make test`/`make lint`/`make typecheck` run in-container; those targets bind-mount `app/`, `tests/` and `pyproject.toml` from the working tree.

Both use `network_mode: host` to dodge a rootless-podman 5.x netns cleanup bug (see commit `d64fca5`). Both get `nvidia.com/gpu=all` via CDI.

## Conventions worth knowing

- Python 3.12, `ruff` check + format (line-length 100, ignore E501) + `mypy --strict`; all three pass and are expected to stay passing. Pydantic v2 + pydantic-settings.
- Annotate image arrays as `Array` from `app/arrays.py` (dtype-agnostic `NDArray[Any]`); document the dtype contract (uint8 vs float32 in [0, 1]) in the docstring. `app.enhancement.*` has `warn_return_any` off because numpy arithmetic on such arrays is typed `Any`; `_scunet_arch.py` is vendored and ignored by mypy/ruff-N806.
- Luma / YCbCr come from `app/enhancement/colorspace.py`; image decoding (RAW/TIFF/JPEG/HEIC/PNG → RGB8) from `app/ingest/decode.py`; resize + JPEG encoding from `app/preview/jpeg_writer.py`. Don't re-implement these locally.
- The enhancement engine's working space is linear Rec.2020 float32 in `[0, 1]`; the only colour conversions (Rec.2020 ↔ sRGB gamut, sRGB transfer) live in `app/enhancement/colorspace.py`. A uint8 array means display-referred sRGB and exists only at the AI-model boundary and inside the `backlit_recover` step, which round-trips through uint8 sRGB as well — `apply_ai_delta` converts to sRGB8 for the model, then merges its delta back into the float master.
- API JSON for photos/decisions is built only in `app/api/serializers.py`.
- Per-item stages (`ingest`, `filter`, `enhance`, `export`) catch exceptions per item, log with `log.exception`, and keep going; the run summary reports failures. Don't let one bad file abort a batch.
- Session reset lives in `app/orchestrator/reset.py::end_session` (used by `raw-curator reset`, `make reset`, and the UI's New batch). It wipes the DB, `cache/{previews,thumbs,enhanced}`, and `photos/{library,exported,<jpeg_subdir>}`; never `incoming/`.
- Job functions live at `app/<phase>/<phase>_job.py` and are called `run_<phase>()`. The Typer CLI in `app/cli.py` imports them lazily so `--help` doesn't pay the Torch/CUDA import cost.
- The AI models are context-managed classes (`ScunetModel`, `RealEsrganModel`, `CodeFormerModel`); each is opened once per batch step in `batch.py::_phase2`, run over every photo that needs it, then closed — its `close()`/`__exit__` empties the CUDA cache before the next model loads. When adding new GPU work, follow the same pattern — the 6 GB budget assumes only one model is resident at a time.
- File moves from `export` go through `app/decision/executor.py`; image bytes are written only by `app/enhancement/render_jpeg.py` (the before/after render JPEGs, EXIF via `pack_tiff.copy_metadata`) and `app/export/jpeg_writer.py` (the final share JPEG). Don't write image bytes directly from job files.
- New non-RAW input formats: update `app/ingest/decode.py` and audit every enhancement/export step's `file_kind` branching before assuming `rawpy` can open it.
- `tests/test_engine_corpus.py` asserts planner decisions on real frames in `tests/data/corpus/` (`real_raw`); look-changing changes ship with a sheet from `scripts/contact_sheet.py` reviewed by the owner.
