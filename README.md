# raw-curator

An ephemeral, single-batch AI photo curation pipeline. Drop a batch of
RAW, JPEG, TIFF, HEIC, or PNG files into `photos/incoming/`, run the
pipeline, review in a local web UI, submit decisions, optionally
enhance the keep-and-fix set, then wipe the session and start fresh
on the next batch.

Everything runs inside a Podman container. The host only needs the NVIDIA
driver, Podman, `podman-compose`, and the NVIDIA Container Toolkit.

- Spec: [`../plan.md`](../plan.md)
- Build plan: [`../implementation.md`](../implementation.md)
- **End-user walkthrough: [`USER_GUIDE.md`](./USER_GUIDE.md)**

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
4. **Cluster** — EXIF burst → pHash dedupe → CLIP cosine via HDBSCAN; one
   recommendation per cluster.
5. **Review** — FastAPI + a single-page React UI (CDN, no Node build).
   Stage decisions per photo with keyboard shortcuts; nothing on disk
   moves until you click **Submit**.
6. **Decide** — binary table: `selected` → action. Score tier is no
   longer part of routing.

   | Selected | Action             | At submit                         | After enhance                                             |
   |----------|--------------------|-----------------------------------|-----------------------------------------------------------|
   | yes      | `keep_and_enhance` | Move RAW → `photos/library/`      | Enhanced TIFF in `photos/exported/`; RAW kept in `library/` |
   | no       | `enhance_only`     | RAW stays in place                | Enhanced TIFF in `photos/exported/`; **original RAW deleted from disk** |

7. **Enhance** — runs on every decided photo (yes or no). Darktable
   develops the RAW to a 16-bit linear TIFF, the AI chain
   (SCUNet → Real-ESRGAN x2 → CodeFormer-if-faces) runs on a downscaled
   copy that fits in 6 GB VRAM, then the result is resampled back to
   native resolution and written as a 16-bit TIFF. For `no` photos
   (`action == "enhance_only"`) the source RAW is deleted **after** the
   TIFF is successfully written — if enhance fails, the original is
   preserved. **Only runs on RAW sources** — already-developed
   JPEG/TIFF/HEIC inputs are skipped with a warning since the AI chain
   is designed around sensor data, not 8-bit display-referred pixels.
8. **Export JPEG** *(optional)* — `make export-jpeg` develops every
   kept RAW in `photos/library/` and every enhanced TIFF in
   `photos/exported/` into a share-ready JPEG under `photos/jpeg/`.
   EXIF is copied from the source; orientation is baked in so viewers
   render the image right-side-up without further rotation.
9. **Reset** — `make reset` deletes the SQLite DB, clears cache and
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

# Initialise empty cache + DB
make reset

# Drop RAWs into photos/incoming/, then run the full pipeline (no UI)
make run

# Start the review UI on http://<host>:8080
make serve

# After reviewing in the UI and clicking Submit, the files move on disk.
# For the yes-low subset, run the enhancement pipeline:
make enhance

# Optional final step: produce share-ready JPEGs from library RAWs and exported TIFFs
make export-jpeg

# At the end of the session, wipe state:
make reset
```

`make help` lists every target.

---

## Make targets

| Target            | What it does                                                    |
|-------------------|-----------------------------------------------------------------|
| `image`           | `podman build -t raw-curator:latest -f Containerfile .`          |
| `download-models` | Fetches CLIP, SigLIP, Real-ESRGAN, SCUNet, CodeFormer, InsightFace into `models/` |
| `reset`           | Drops DB + clears `cache/` + clears `photos/{library,archive,quarantine,exported}/`; runs `alembic upgrade head` |
| `ingest`          | Walk `photos/incoming/` → DB rows + previews + thumbs           |
| `filter`          | Blur / pHash / exposure                                         |
| `score`           | GPU scoring: CLIP, IQA, faces (stage-by-stage)                  |
| `cluster`         | EXIF burst + pHash dedupe + CLIP HDBSCAN + recommendation       |
| `run`             | `ingest → filter → score → cluster` in one shot (no UI)         |
| `serve`           | FastAPI + UI on `http://0.0.0.0:8080`                           |
| `submit`          | Apply staged decisions (file moves) outside the UI              |
| `enhance`         | Hybrid RAW → AI → 16-bit TIFF for the yes-low set               |
| `export-jpeg`     | RAWs (`library/`) and TIFFs (`exported/`) → share-ready JPEGs in `photos/jpeg/` |
| `shell`           | Drop into a bash shell inside the container                     |
| `test`            | `pytest -q` inside the container                                |
| `lint`            | `ruff check app/ tests/`                                        |
| `typecheck`       | `mypy app/`                                                     |
| `clean`           | `podman compose down -v` and remove the image                   |

---

## Directory layout

```
photos/
  incoming/      <- drop RAWs here at session start
  library/       <- yes + high score (RAW kept)
  archive/       <- yes + low (RAW set aside after enhance) | no + high
  quarantine/    <- no + low (wiped by `make reset`)
  exported/      <- enhanced 16-bit TIFFs
  jpeg/          <- share-ready 8-bit JPEGs from `make export-jpeg` (optional)

cache/           <- session DB + previews + thumbs (wiped by `make reset`)
  session.db     <- SQLite + sqlite-vec, WAL mode
  previews/      <- 3000 px JPEG, used by UI + AI stages
  thumbs/        <- 512 px JPEG, used by grid + pHash

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
| `RAWCURATOR_ENHANCE_AI_SCALE`     | `0.7`         | Pre-AI downscale factor. `0.7` peaks ~5.5 GB on a 6 GB card (24 MP source). Lower to `0.5` if other CUDA processes share the GPU; raise toward `0.75` only on 8 GB+ cards. |
| `RAWCURATOR_ENHANCE_DENOISE`      | `true`        | Skip SCUNet if false                                                    |
| `RAWCURATOR_ENHANCE_DENOISE_STRENGTH` | `0.75`    | Blends SCUNet output with the input. `1.0` is full denoise; `<1` retains natural micro-texture so the image doesn't look plastic. |
| `RAWCURATOR_ENHANCE_REALESRGAN_FIDELITY` | `0.7` | Blends Real-ESRGAN output with a Lanczos upscale. `1.0` is full AI sharpening (riskier on skin/sky/foliage); `0.7` keeps most detail recovery while softening AI artifacts; drop to `0.5` for very soft output. |
| `RAWCURATOR_ENHANCE_FACE_RESTORE` | `true`        | Skip CodeFormer if false                                                |
| `RAWCURATOR_ENHANCE_CODEFORMER_W` | `0.85`        | Higher = more faithful to the original skin texture (natural). Lower = stronger restoration (waxy/airbrushed risk). Default leans natural. |
| `RAWCURATOR_ENHANCE_BACKLIT_RECOVERY` | `true`    | Auto-detects backlit scenes (dense shadows + dense highlights) and lifts the subject while protecting background highlights. Edge-preserving — no HDR halos. |
| `RAWCURATOR_ENHANCE_BACKLIT_SHADOW_LIFT` | `0.4` | `0` disables; `~0.4` is natural; `>0.7` starts looking HDR.            |
| `RAWCURATOR_ENHANCE_BACKLIT_HIGHLIGHT_PROTECT` | `0.15` | How aggressively the lift rolls off above ~65% luminance.        |
| `RAWCURATOR_ENHANCE_TARGET_RES`   | `native`      | `native` \| `200%` \| `WIDTHxHEIGHT`                                    |
| `RAWCURATOR_BURST_SECONDS`        | `2`           | EXIF timestamp window for burst grouping                                |
| `RAWCURATOR_PHASH_HAMMING_THRESHOLD` | `8`        | Within-burst pHash distance for "duplicate"                             |
| `RAWCURATOR_CLIP_COSINE_THRESHOLD`| `0.92`        | CLIP cosine threshold for cross-batch "duplicate"                       |
| `RAWCURATOR_CPU_WORKERS`          | `os.cpu_count()` (e.g. `8` on Ryzen 3 3100) | Process pool size for ingest / filter / export-jpeg. Set lower to cap memory pressure. |
| `RAWCURATOR_JPEG_QUALITY`         | `92`          | JPEG quality used by `make export-jpeg`                                 |
| `RAWCURATOR_JPEG_LONG_EDGE`       | `0`           | `0` keeps native resolution; e.g. `4000` caps the long edge for sharing |
| `RAWCURATOR_JPEG_PROGRESSIVE`     | `true`        | Write progressive JPEGs (better for web preview)                         |

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

Decisions exercised end-to-end: yes-high → `library/`, no-low →
`quarantine/`, yes-low → `exported/` (real SCUNet + Real-ESRGAN +
CodeFormer altered pixels, output SHA differs from a no-op TIFF).

### Known gaps vs `implementation.md`

- **Score throughput** on the 6 GB RTX 2060 is 5–6× slower than the
  ambitious 1.5 s/photo target in plan §4 (steady-state ~11.5 s/photo
  here). Acceptable for batch use; would benefit from `torch.compile`
  re-enabling and stage reordering.
- **CodeFormer face-parse weights** (~81 MB) download into
  `codeformer-pip`'s in-container default cache, not `/data/models/`.
  Each fresh `podman run --rm` re-downloads them. Workaround: set
  `XDG_CACHE_HOME=/data/models/codeformer-cache` in `compose.yaml`.
- **No formal pytest acceptance suite against real RAW fixtures** —
  validation was hand-run. Unit tests cover schema, filters,
  clustering, and decision rules.

---

## Troubleshooting

| Symptom                                        | Fix                                                                    |
|------------------------------------------------|------------------------------------------------------------------------|
| `Failed to initialize NVML` inside container   | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`            |
| `make image` is slow                           | Expected — CUDA base + pyiqa + insightface push the image to ~27 GB. Cached on rebuild. |
| `make run` OOM during scoring                  | Lower `RAWCURATOR_CLIP_BATCH=4`; ensure no other CUDA process is resident |
| `make enhance` OOM                             | Lower `RAWCURATOR_ENHANCE_AI_SCALE` (0.7 → 0.5 → 0.4)                    |
| UI shows "loading…" forever                    | Check `podman logs <ui-container>`; usually `make reset` was skipped and the DB schema is missing |
| `darktable-cli` error "output file already exists" | Already worked around — if you see this, the workaround in `app/enhancement/develop_full.py` regressed |

---

## License

MIT. See `../plan.md` for full project rationale.
