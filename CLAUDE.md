# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project shape

`raw-curator` is an **ephemeral, single-batch** AI photo curation pipeline. The whole thing runs inside a Podman container on a GPU host (Ubuntu + NVIDIA driver + NVIDIA Container Toolkit). The host stays clean; the container holds CUDA, Torch, darktable, exiftool, and all model weights.

There is no long-lived service. Each batch flows through the pipeline once, the user reviews in a web UI, decisions are applied to disk, then `make reset` wipes everything (DB + cache + working dirs) and the next batch starts fresh. `models/` and `xmp/` are the only persistent dirs.

See `README.md` for the user-facing overview and `USER_GUIDE.md` for the end-to-end walkthrough.

## Common commands

Everything is invoked through `make` (which delegates to `podman-compose` → the in-container `raw-curator` Typer CLI). Run from the repo root:

```bash
make image            # build raw-curator:latest (~27 GB, 10–15 min first time)
make download-models  # fetch ~17 GB of weights into models/
make reset            # wipe DB + cache + working photo dirs; runs alembic upgrade head

# Per-stage (or `make run` for ingest→filter→score→cluster in one shot)
make ingest filter score cluster
make serve            # FastAPI + UI on :8080 (review + stage decisions)
make submit           # apply staged decisions (moves files on disk)
make enhance          # AI chain for the enhance_export set (RAW → 16-bit TIFF)
make export-jpeg      # share-ready JPEGs from library RAWs and exported TIFFs

# Dev loop
make test             # pytest -q inside the container
make lint             # ruff check app/ tests/
make typecheck        # mypy app/
make shell            # bash inside the app container
```

Run a single test (from inside the container or via `podman-compose run --rm app`):

```bash
pytest tests/test_decision_rules.py -q
pytest tests/test_decision_rules.py::test_yes_high_keeps_raw -q
```

GPU-marked tests are skipped unless `RUN_GPU_TESTS=1`. `real_raw`-marked tests are skipped unless `tests/data/` contains RAW fixtures.

The CLI is also reachable directly inside the container: `raw-curator ingest|filter|score|cluster|submit|enhance|export-jpeg|serve|run --auto|info`.

## Architecture

### Pipeline phases (each is a `make` target and an `app/<phase>/<phase>_job.py` entrypoint)

1. **ingest** (`app/ingest/`) — walks `photos/incoming/`, computes xxh3 hash, extracts EXIF via `exiftool`, decodes RAW via `rawpy` (and HEIC via `pillow-heif`), writes a 512 px thumb + 3000 px preview JPEG into `cache/`. Records `file_kind` (`raw`/`jpeg`/`tiff`/`heic`/`png`) per photo — downstream stages branch on this.
2. **filter** (`app/filters/`) — CPU-only Laplacian blur variance, pHash/dHash, exposure histogram flags.
3. **score** (`app/scoring/` + `app/embedding/`) — GPU stage. CLIP ViT-L/14 + aesthetic-predictor v2.5 + MUSIQ + MANIQA + InsightFace. Runs **stage-by-stage** (`stage=clip|iqa|faces|all`), freeing CUDA between stages so it fits a 6 GB RTX 2060.
4. **cluster** (`app/clustering/`) — EXIF burst grouping → pHash dedupe within burst → CLIP cosine + HDBSCAN across the batch → one recommended photo per cluster.
5. **serve** (`app/api/` + `app/ui/`) — FastAPI app exposing `/api/{queue,photo,cluster,decide,submit}` plus the static SPA from `app/api/static/`. The UI is plain HTML/JS using CDN-hosted React (no Node build step). Decisions are staged in the `decisions` table; nothing moves on disk until **Submit**.
6. **submit** (`app/decision/`) — applies staged decisions per the binary rule table in `app/decision/rules.py`. `yes` moves the RAW into `photos/library/`; `no` leaves the RAW in place so `make enhance` can still develop it. `executor.py` does the actual moves.
7. **enhance** (`app/enhancement/`) — for every photo with `action IN ('keep_and_enhance', 'enhance_only')`, the chain is: darktable develops RAW → 16-bit linear TIFF → backlit recovery → downscale to fit 6 GB VRAM → SCUNet denoise → Real-ESRGAN x2 → (CodeFormer if faces) → upsample back to native → write 16-bit TIFF to `photos/exported/`. When `action == "enhance_only"` (the `no` path) the source RAW is deleted on disk after the TIFF is successfully written. **RAW-only**; non-RAW sources are skipped with a warning (their originals are left in place) because the AI chain expects sensor data, not 8-bit display-referred pixels.
8. **export-jpeg** (`app/export/`) — optional final step. Develops every kept RAW in `photos/library/` and every enhanced TIFF in `photos/exported/` into share-ready JPEGs in `photos/jpeg/`. EXIF copied from source; orientation baked in. Multi-process via `ProcessPoolExecutor`.

### Decision rules (`app/decision/rules.py`)

Binary: `yes` or `no`. Score tier no longer drives routing.

| Selected | Action             | At submit                          | After enhance                  |
|----------|--------------------|------------------------------------|--------------------------------|
| yes      | `keep_and_enhance` | Move RAW to `photos/library/`      | TIFF written to `exported/`; RAW kept in `library/` |
| no       | `enhance_only`     | RAW stays in place (e.g. `incoming/`) | TIFF written to `exported/`; source RAW deleted from disk |

Every decided photo flows through the enhancement chain. `make enhance` queries `Decision.action IN ('keep_and_enhance', 'enhance_only')`. The source-RAW deletion for `enhance_only` is intentional and **irreversible**; it happens only after the output TIFF exists on disk, so if enhance fails the original is preserved. Do not re-introduce score tiers into routing without confirming with the user.

`tier_from_scores` (`combined = 0.6 * technical + 0.4 * normalized_aesthetic`, threshold `0.55`) is retained as a display-only helper; the `Decision.score_tier` and `Decision.enhance_requested` columns are dead fields kept in the schema to avoid a migration but are no longer read or written by app code.

### Storage

- **SQLite + sqlite-vec, WAL mode** at `cache/session.db`. Engine setup in `app/db.py` loads the `sqlite-vec` extension on every connection. Schema in `app/models.py` (SQLAlchemy 2.0 declarative). Migrations via Alembic in `db/migrations/`.
- `photo_embeddings.vec` is a raw `LargeBinary` blob of float16 CLIP features (768-dim by default).
- All paths stored in the DB are container-side absolute paths under `/data/`.

### Configuration

All knobs are env vars with prefix `RAWCURATOR_`, loaded via pydantic-settings in `app/config.py`. The container reads them from `.env` (which `compose.yaml` injects via `env_file`, not just `${VAR}` interpolation — see commit `f098c14`). Bind-mount points are fixed: `/data/{photos,cache,models,xmp}`.

VRAM-sensitive defaults are tuned for a 6 GB RTX 2060:
- `RAWCURATOR_ENHANCE_AI_SCALE=1.0` — no pre-AI downscale; AI sees native pixels. Drop to `0.85`/`0.7`/`0.5` if OOM.
- `RAWCURATOR_ENHANCE_TARGET_RES=200%` — keep Real-ESRGAN's x2 output (e.g. 12kx8k for 24 MP source). Set to `native` to downsample back; `200%` output TIFFs are ~4x larger on disk.
- `RAWCURATOR_CLIP_BATCH=8` — lower to `4` on OOM during scoring.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in `compose.yaml` to reduce fragmentation across the SCUNet → Real-ESRGAN → CodeFormer stages.

### Container topology

`compose.yaml` defines two services backed by the same `raw-curator:latest` image:
- `app` — one-shot CLI commands (`make ingest`, `make score`, etc. each `run --rm app raw-curator <cmd>`).
- `ui` — long-running `raw-curator serve --host 0.0.0.0 --port 8080`.

Both use `network_mode: host` to dodge a rootless-podman 5.x netns cleanup bug (see commit `d64fca5`). Both get `nvidia.com/gpu=all` via CDI.

## Conventions worth knowing

- Python 3.12, `ruff` (line-length 100, ignore E501) + `mypy --strict`. Pydantic v2 + pydantic-settings.
- Job functions live at `app/<phase>/<phase>_job.py` and are called `run_<phase>()`. The Typer CLI in `app/cli.py` imports them lazily so `--help` doesn't pay the Torch/CUDA import cost.
- Enhancement steps explicitly call `torch.cuda.empty_cache()` between stages (`_free_gpu` in `enhance_job.py`). When adding new GPU work, follow the same pattern — the 6 GB budget assumes only one model is resident at a time.
- File moves from `submit` and outputs from `enhance`/`export-jpeg` all go through helpers in `app/decision/executor.py`, `app/enhancement/pack_tiff.py`, `app/export/jpeg_writer.py`. Don't write image bytes directly from job files.
- New non-RAW input formats: update `app/ingest/decode.py` and audit every enhancement/export step's `file_kind` branching before assuming `rawpy` can open it.
