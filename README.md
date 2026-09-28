# raw-curator

An ephemeral, single-batch AI photo curation pipeline. Drop a batch of
RAW, JPEG, TIFF, HEIC, or PNG files into `photos/incoming/`, then drive
the whole pipeline from the **Control Center** web UI: auto-run the
analysis stages *and* the Auto Enhancement Engine on every RAW, review
each frame **original vs enhanced** side by side, pick a version per
photo (with an independent keep-RAW toggle), export the chosen JPEGs,
then wipe the session and start fresh on the next batch.

Everything runs inside a Podman container. The host only needs the NVIDIA
driver, Podman, `podman-compose`, and the NVIDIA Container Toolkit.

- **End-user walkthrough: [`USER_GUIDE.md`](./USER_GUIDE.md)**
- **Contributor notes: [`CLAUDE.md`](./CLAUDE.md)**

---

## What it does

**Supported input formats:** RAW (Canon CR2/CR3, Nikon NEF/NRW, Sony
ARW, Fuji RAF, Olympus ORF, Panasonic RW2, Pentax PEF, Adobe DNG, and
~20 others), JPEG, TIFF, HEIC/HEIF, PNG. The file kind is recorded
per-photo and surfaced as a chip in the review UI.

For each batch:

1. **Ingest** — walk `photos/incoming/`, compute xxh3 hash, extract EXIF,
   produce 512 px thumb + 3000 px preview JPEGs.
2. **Filter** — Laplacian blur variance, pHash/dHash, exposure histogram.
3. **Score** — CLIP ViT-L/14 + aesthetic-predictor v2.5 + MUSIQ + MANIQA
   + InsightFace, all FP16 on a single GPU worker, stage-by-stage to fit
   6 GB VRAM.
4. **Cluster** — EXIF burst grouping, then CLIP HDBSCAN across the rest of
   the batch; one recommendation per cluster (ranked 0.6·technical +
   0.4·aesthetic).
5. **Enhance** — runs on **every RAW**, *before* review, via the **Auto
   Enhancement Engine**. Darktable develops the RAW with its sigmoid
   workflow into a 16-bit **linear Rec.2020** intermediate; lens
   distortion and chromatic aberration are corrected per-frame from EXIF
   (lensfun); the engine measures the frame across five quality
   dimensions (exposure, dynamic range, color, sharpness, noise) and
   builds a **per-photo recipe** of classical + AI steps tuned to its
   measured deficits — the recipe is visible in the review UI. Each AI
   step (SCUNet denoise, Real-ESRGAN when enlarging or the source is
   small, CodeFormer for faces that are small, soft, or in a noisy frame)
   sees an 8-bit sRGB copy but is merged back as a *delta* into the float
   master, so the master is never quantised; the result passes a
   verification gate. Instead of a TIFF master, enhance writes four
   preview JPEGs per photo into `cache/enhanced/` — a **before**
   (developed, un-enhanced) and an **after** (enhanced) at review
   resolution plus full-res copies — so the before/after comparison is
   ready the moment review starts. Enhance **never moves or deletes a
   RAW**. **Only runs on RAW sources** — already-developed JPEG/TIFF/HEIC
   inputs are skipped (offered original-only in review) since the engine
   expects sensor data, not 8-bit display-referred pixels.
6. **Review** — the Control Center (FastAPI + a single-page React UI,
   CDN, no Node build) shows a stage timeline and a review panel
   (All/Clusters views). Each frame gets a real **before/after slider**
   (developed original vs enhanced) plus a per-photo version pick —
   **original** / **enhanced** / **discard** — and an independent
   **keep-RAW** checkbox, all with keyboard shortcuts. Everything is
   staged; nothing on disk moves until you click **Export selected**.
7. **Export** — replaces the old submit + export-jpeg steps. For each
   kept photo (`original` or `enhanced`) it writes the chosen JPEG into
   `photos/jpeg/` (enhanced → the cached enhanced render; original → the
   developed RAW render, or the source itself for a non-RAW), then
   applies retention: `keep_raw` moves the RAW into `photos/library/`,
   otherwise the RAW is deleted **after** the JPEG exists on disk.
   `discard`/`undecided` photos produce nothing. EXIF copied from the
   source; orientation baked in.
8. **Reset** — `make reset` deletes the SQLite DB, clears cache and
   working dirs; `models/` is left alone.

---

## Quick start

```bash
# One-time on the GPU host (Ubuntu 24.04+, NVIDIA driver already installed):
ssh -t desktop 'sudo bash -s' < scripts/host-bootstrap.sh

# Build the image (~27 GB; takes 10–15 minutes the first time)
make image

# Fetch model weights into models/ (~17 GB; idempotent — skips on re-run)
make download-models

# First time only: initialise the DB schema
# (the UI's "New batch" button does the same wipe+init later on)
make reset

# Drop photos into photos/incoming/, then start the Control Center
make serve
```

Open `http://<host>:8080` and drive the whole batch from the browser:

1. Click **Auto-run** — leg 1 runs ingest → filter → score → cluster →
   **enhance**, then stops for human review.
2. Review in the panel: drag the **before/after slider**, pick
   **original / enhanced / discard** per photo and toggle keep-RAW
   (bulk controls + keyboard shortcuts too), then click **Export
   selected** — leg 2 runs the export unattended.
3. Done: share-ready JPEGs land in `photos/jpeg/` and kept RAWs in
   `photos/library/`.
4. Copy your outputs somewhere safe, then click **New batch** (type
   `RESET` to confirm) to wipe the session — `photos/incoming/` and
   `models/` are left alone.

A docked resource bar shows live CPU/RAM/GPU/VRAM/disk with sparklines
and warns when free space on the photos volume drops below
`RAWCURATOR_MONITOR_DISK_WARN_FREE_GB` (default 50 GB). Stage logs
stream into the UI and are also written under `cache/logs/`.

`make help` lists every target.

### Advanced / CLI path

Every stage is still a standalone make target for headless use:

```bash
# Initialise empty cache + DB (the UI's "New batch" does the same)
make reset

# Drop RAWs into photos/incoming/, then run the analysis pipeline (no UI)
make run            # = make ingest filter score cluster

# Run the Auto Enhancement Engine on every RAW (before review):
make enhance

# Start the UI to review before/after and pick a version per photo
make serve

# After choosing original/enhanced/discard + keep-RAW in the UI, apply them:
# writes the chosen JPEG to photos/jpeg/ and moves or deletes each RAW
make export

# At the end of the session, wipe state:
make reset
```

---

## Make targets

| Target            | What it does                                                    |
|-------------------|-----------------------------------------------------------------|
| `image`           | `podman build -t raw-curator:latest -f Containerfile .` (from scratch; downloads ~5 GB of wheels) |
| `image-warm`      | Same build, but seeds site-packages from the current `raw-curator:latest` so only changed/added packages are fetched. Minutes instead of an hour on a slow link. |
| `download-models` | Fetches CLIP, SigLIP, Real-ESRGAN, SCUNet, CodeFormer, InsightFace into `models/` |
| `reset`           | `raw-curator reset --force`: drops DB, empties `cache/{previews,thumbs,enhanced}/` and `photos/{library,exported,jpeg}/`, runs `alembic upgrade head` |
| `ingest`          | Walk `photos/incoming/` → DB rows + previews + thumbs           |
| `filter`          | Blur / pHash / exposure                                         |
| `score`           | GPU scoring: CLIP, IQA, faces (stage-by-stage)                  |
| `cluster`         | EXIF burst + CLIP HDBSCAN + recommendation                      |
| `run`             | `ingest → filter → score → cluster` in one shot (no UI)         |
| `serve`           | Control Center UI on `http://0.0.0.0:8080` — runs every stage, streams logs, live resource monitor |
| `enhance`         | Auto Enhancement Engine: RAW → classical + AI → before/after render JPEGs in `cache/enhanced/` for every RAW |
| `export`          | Apply export choices: chosen JPEG → `photos/jpeg/`, then RAW retention (keep → `library/`, else delete) |
| `shell`           | Drop into a bash shell inside the container                     |
| `test`            | `pytest -q` inside the container; `app/` and `tests/` are bind-mounted from the working tree, so no rebuild is needed |
| `lint`            | `ruff check` + `ruff format --check` over `app/ tests/ scripts/` |
| `format`          | `ruff format app/ tests/ scripts/` (rewrites files)              |
| `typecheck`       | `mypy app/`                                                     |
| `clean`           | `podman compose down -v` and remove the image                   |

---

## Directory layout

```
photos/
  incoming/      <- drop RAWs here at session start; kept RAWs move to library/ at export, others deleted
  library/       <- RAWs kept via the keep-RAW toggle (moved here at export)
  jpeg/          <- share-ready 8-bit JPEGs from `make export` (the chosen version per photo)

cache/           <- session DB + previews + thumbs + enhanced renders (wiped by `make reset`)
  session.db     <- SQLite + sqlite-vec, WAL mode; includes `quality_reports`
  previews/      <- 3000 px JPEG, used by UI + AI stages
  thumbs/        <- 512 px JPEG, used by grid + pHash
  enhanced/      <- per-RAW before/after render JPEGs (<hash>.{before,after}.jpg + .full.jpg)

models/          <- model weights (~17 GB, persistent across sessions)
  hf/            <- CLIP + SigLIP HF snapshots
  insightface/   <- buffalo_l (det + arcface)
  torch/         <- pyiqa weights, populated on first run
  CodeFormer/    <- codeformer.pth
  RealESRGAN_x2plus.pth
  scunet_color_real_psnr.pth

xmp/             <- darktable sidecars (user-authored, persistent)
```

`photos/`, `cache/`, `models/`, and `xmp/` are bind-mounted into the
container at `/data/{photos,cache,models,xmp}` so you can browse them
natively from the host.

---

## Configuration

All knobs are environment variables, prefix `RAWCURATOR_`. Copy
`.env.example` to `.env` and edit as needed. See
[`app/config.py`](./app/config.py) for the full list.

The most useful overrides:

| Variable                          | Default       | Purpose                                                                 |
|-----------------------------------|---------------|-------------------------------------------------------------------------|
| `RAWCURATOR_DARKTABLE_WORKFLOW`   | `scene-referred (sigmoid)` | darktable pixel workflow for develop; filmic is the 4.6 default look |
| `RAWCURATOR_ENHANCE_LENS_CORRECTION` | `true`     | Per-frame lens distortion + chromatic-aberration correction via lensfun, read from EXIF (bundled DB plus the calibration XML under `app/enhancement/darktable/lensfun/`). Frames whose lens lensfun cannot resolve pass through uncorrected. Set false to skip it. |
| `RAWCURATOR_ENHANCE_AI_SCALE`     | `1.0`         | Pre-AI downscale factor. `1.0` means AI sees the full native source — maximum detail recovery, peaks ~5.5 GB on a 6 GB card (24 MP). Drop to `0.85` / `0.7` / `0.5` progressively if OOM or if other CUDA processes share the GPU. |
| `RAWCURATOR_ENHANCE_DENOISE`      | `true`        | Skip SCUNet if false                                                    |
| `RAWCURATOR_ENHANCE_DENOISE_STRENGTH` | `0.75`    | Blends SCUNet output with the input. `1.0` is full denoise; `<1` retains natural micro-texture so the image doesn't look plastic. |
| `RAWCURATOR_ENHANCE_REALESRGAN_FIDELITY` | `0.7` | Blends Real-ESRGAN output with a Lanczos upscale. `1.0` is full AI sharpening (riskier on skin/sky/foliage); `0.7` keeps most detail recovery while softening AI artifacts; drop to `0.5` for very soft output. |
| `RAWCURATOR_ENHANCE_FACE_RESTORE` | `true`        | Skip CodeFormer if false                                                |
| `RAWCURATOR_ENHANCE_CODEFORMER_W` | `0.85`        | Higher = more faithful to the original skin texture (natural). Lower = stronger restoration (waxy/airbrushed risk). Default leans natural. |
| `RAWCURATOR_ENHANCE_FACE_RESTORE_MAX_PX` | `300` | Faces whose box is at least this many pixels on the long side and sharp are left alone; smaller or soft faces get CodeFormer. |
| `RAWCURATOR_ENHANCE_FACE_MIN_SIMILARITY` | `0.5` | ArcFace cosine between a face before and after CodeFormer; below this the original face is pasted back. |
| `RAWCURATOR_ENHANCE_BACKLIT_RECOVERY` | `true`    | Auto-detects backlit scenes (dense shadows + dense highlights) and lifts the subject while protecting background highlights. Edge-preserving — no HDR halos. |
| `RAWCURATOR_ENHANCE_BACKLIT_SHADOW_LIFT` | `0.4` | `0` disables; `~0.4` is natural; `>0.7` starts looking HDR.            |
| `RAWCURATOR_ENHANCE_BACKLIT_HIGHLIGHT_PROTECT` | `0.15` | How aggressively the lift rolls off above ~65% luminance.        |
| `RAWCURATOR_ENHANCE_TARGET_RES`   | `200%`        | `native` (downsample back to source) \| `200%` (keep Real-ESRGAN's 2x output — 24 MP source becomes ~96 MP, ~4x larger enhanced render JPEGs) \| `WIDTHxHEIGHT` (explicit pixel size). |
| `RAWCURATOR_ENHANCE_SR_MIN_LONG_EDGE` | `3000`    | Real-ESRGAN runs only when enlarging (target > native) or when the source long edge is below this many pixels. |
| `RAWCURATOR_SCUNET_TILE` / `_TILE_PAD` | `512` / `32` | SCUNet tile edge and reflective padding (px). Lower the tile (multiples of 64) on OOM. |
| `RAWCURATOR_CODEFORMER_MAX_LONG_EDGE` | `2048`    | Long-edge cap fed to CodeFormer's face detector, which runs on the full frame. Lower on OOM. |
| `RAWCURATOR_BURST_SECONDS`        | `2`           | EXIF timestamp window for burst grouping                                |
| `RAWCURATOR_CPU_WORKERS`          | `os.cpu_count()` (e.g. `8` on Ryzen 3 3100) | Process pool size for ingest / filter. Set lower to cap memory pressure. |
| `RAWCURATOR_JPEG_QUALITY`         | `92`          | JPEG quality used by `make export`                                      |
| `RAWCURATOR_JPEG_LONG_EDGE`       | `0`           | `0` keeps native resolution; e.g. `4000` caps the long edge for sharing |
| `RAWCURATOR_JPEG_PROGRESSIVE`     | `true`        | Write progressive JPEGs (better for web preview)                         |
| `RAWCURATOR_REVIEW_LONG_EDGE`     | `3000`        | Long edge (px) of the before/after review JPEGs `make enhance` renders for the slider |
| `RAWCURATOR_KEEP_RAW_DEFAULT`     | `true`        | Initial keep-RAW state for a freshly-created decision row (safe default: archive the RAW) |

---

## Verified end-to-end on real RAWs

Validated against 10 Canon EOS R8 CR3 files (24 MP each) on Ryzen 3 3100 +
RTX 2060 6 GB:

| Stage         | Time            | Notes                                                |
|---------------|-----------------|------------------------------------------------------|
| Ingest        | 6.5 s           | 1.5 img/s — passes the plan's 1.1 img/s target       |
| Filter        | 2.8 s           |                                                      |
| Score (first run, with model downloads) | 7 min 16 s | Subsequent runs ~11.5 s/photo steady-state |
| Cluster       | 14 s            |                                                      |
| Enhance       | 47.6 s / photo  | Plan budget was 6 min/photo; well within             |

Export choices exercised end-to-end: `enhanced` → the chosen enhanced
JPEG in `photos/jpeg/`; `original` → the developed-RAW JPEG in
`photos/jpeg/`; `keep_raw` on → RAW archived in `photos/library/`, off →
RAW deleted after the JPEG lands; `discard` → nothing written. The Auto
Enhancement Engine produces a `quality_reports` row per photo and runs
the planned classical + AI steps (real SCUNet + Real-ESRGAN + CodeFormer
altered pixels; the enhanced render differs from the developed
original).

### Known gaps

- **Score throughput** on the 6 GB RTX 2060 is steady-state
  ~11.5 s/photo. Acceptable for batch use; would benefit from
  `torch.compile` re-enabling and stage reordering.
- **CodeFormer face-parse weights** (~81 MB) download into
  `codeformer-pip`'s in-container default cache, not `/data/models/`.
  Each fresh `podman run --rm` re-downloads them. Workaround: set
  `XDG_CACHE_HOME=/data/models/codeformer-cache` in `compose.yaml`.
- **Real-RAW acceptance coverage is plan-level only** —
  `tests/test_engine_corpus.py` asserts the planned steps for each RAW in
  `tests/data/corpus` (skipped via the `real_raw` marker unless the fixtures
  are present); the developed-pixel quality is still validated by eye. Unit
  tests cover schema, filters, clustering, and decision rules.

---

## Troubleshooting

| Symptom                                        | Fix                                                                    |
|------------------------------------------------|------------------------------------------------------------------------|
| `Failed to initialize NVML` inside container   | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`            |
| `make image` is slow                           | Expected — CUDA base + pyiqa + insightface push the image to ~27 GB. Cached on rebuild. |
| `make run` OOM during scoring                  | Lower `RAWCURATOR_CLIP_BATCH=4`; ensure no other CUDA process is resident |
| `make enhance` OOM                             | Lower `RAWCURATOR_ENHANCE_AI_SCALE` (1.0 → 0.85 → 0.7 → 0.5)             |
| UI shows "loading…" forever                    | Check `podman logs <ui-container>`; usually `make reset` was skipped and the DB schema is missing |
| `darktable-cli` error "output file already exists" | Already worked around — if you see this, the workaround in `app/enhancement/develop_full.py` regressed |

---

## License

MIT.
