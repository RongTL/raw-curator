# raw-curator — User Guide

This guide walks you through one complete session, from dropping RAW
files into `photos/incoming/` to wiping state at the end. It assumes
the host has been bootstrapped (see [README](./README.md) §"Quick
start") and the container image already built.

If anything in this guide contradicts the code, the code is right —
file an issue against the guide.

---

## Mental model: ephemeral, single-batch

The system has **no long-term memory**. One "session" = one batch:

```
[1] drop RAWs into photos/incoming/
[2] run the pipeline (ingest → filter → score → cluster)
[3] open the UI, review, stage decisions, submit
[4] (optional) run enhance for the yes-low set
[5] copy outputs out of photos/library/ + photos/exported/
[6] make reset → DB + cache + working dirs are wiped
[7] next batch is a clean slate; models/ is kept
```

There is **no re-curation across sessions**, **no cross-batch search**,
and **no audit log retention** beyond the active session. This is by
design — see [`../plan.md`](../plan.md) §"Session Model".

---

## Where to run commands

Everything runs from the project root on the GPU host:

```bash
ssh desktop
cd ~/projects/raw-curator
```

Every `make` target wraps `podman-compose run --rm app ...`, so you do
not need a local Python install. To open a shell inside the container:

```bash
make shell           # inside: raw-curator --help
```

To check what the container sees:

```bash
make shell
$ raw-curator info
photos:  /data/photos
cache:   /data/cache
models:  /data/models
db_url:  sqlite:////data/cache/session.db
db file present: True
tables (8): ['alembic_version', 'cluster_members', 'clusters',
             'decisions', 'faces', 'photo_embeddings', 'photos',
             'session_meta']
```

---

## Session walkthrough

### Step 0 — Prepare a clean slate

```bash
cd ~/projects/raw-curator
make reset          # confirms before wiping; wipes DB + cache + working dirs
```

After this:
- `cache/session.db` is freshly migrated (empty schema).
- `photos/{library,archive,quarantine,exported}/` are empty.
- `photos/incoming/` is **left alone** — that is your input.
- `models/` and `xmp/` are **left alone**.

### Step 1 — Drop RAWs into `photos/incoming/`

From your laptop, copy a batch in:

```bash
# From a laptop on the same network
rsync -av --progress \
  ~/today-shoot/*.CR3 \
  desktop:~/projects/raw-curator/photos/incoming/
```

Or read from an SD card on the host directly. Supported formats:
`.CR2`, `.CR3`, `.ARW`, `.NEF`, `.DNG`, `.RAF`, `.ORF`. Filenames are
preserved end-to-end.

### Step 2 — Run the analysis pipeline

The fastest way is the autopilot:

```bash
make run            # ingest → filter → score → cluster
```

This is equivalent to running the four phases individually:

```bash
make ingest         # ~1.5 img/s on R3-3100 (10 RAWs = ~7 s)
make filter         # ~200 img/s — CPU only, blur + pHash + exposure
make score          # GPU-bound, ~11.5 s/photo steady-state on RTX 2060
make cluster        # whole batch in seconds
```

Run them separately when debugging or when you want a checkpoint
between stages.

#### What each stage does in practice

- **ingest**: opens the RAW with rawpy, extracts the embedded JPEG
  preview where possible (much faster than re-demosaicing), writes a
  512 px thumb to `cache/thumbs/<hash>.jpg` and a 3000 px preview to
  `cache/previews/<hash>.jpg`. Inserts a row in `photos` keyed by
  xxh3 of the RAW bytes. Re-running is a no-op for files already in
  the DB.

- **filter**: computes Laplacian variance on a 512 px grayscale crop
  of the thumbnail (blur signal), perceptual + difference hashes for
  near-duplicate prefilter, and an exposure histogram.
  Strongly-blurry or blown/crushed photos are flagged but **never
  auto-rejected** — the user still reviews them.

- **score**: runs three GPU stages in sequence, unloading each model
  before loading the next so 6 GB VRAM is enough:
  1. **CLIP ViT-L/14** (`laion2b_s32b_b82k`) embedding +
     **aesthetic-predictor v2.5** head.
  2. **MUSIQ** + **MANIQA** ensemble → `technical_score` (0–1).
  3. **InsightFace buffalo_l** → bounding boxes + 512-dim ArcFace
     embeddings per detected face.

- **cluster**: three funneled passes — EXIF burst window (same
  `camera_body`, capture time ±2 s), within-burst pHash Hamming ≤ 8,
  whole-batch CLIP cosine ≥ 0.92 via HDBSCAN. One photo per cluster
  is marked `is_recommended = True`, ranked by
  `0.6·technical_score + 0.4·aesthetic_score`.

### Step 3 — Review in the UI

```bash
make serve          # foreground; Ctrl-C to stop
```

Open `http://<host>:8080` in any browser on your network.

The UI is a single page: a grid of thumbnails on top, a header with
sort/submit on top, and a full-screen detail modal that opens when
you click or press Enter.

#### Grid view

Each tile shows:
- Thumbnail (lazy-loaded from `cache/thumbs/`).
- Stars (top right) — the per-photo rating you assign.
- `REC` badge (top left) — the cluster recommendation.
- `yes`/`no` chip — your current selection (no chip = undecided).
- Tech/aesthetic scores along the bottom.

Click a tile or press **Enter** to open the first photo in detail
view.

#### Detail view (full-screen modal)

Center: 3000 px preview from `cache/previews/`. Bottom panel:
scores, decision controls, EXIF, cluster info.

Keyboard shortcuts inside the modal:

| Key       | Action                                  |
|-----------|-----------------------------------------|
| `1`–`5`   | Set stars                               |
| `0`       | Clear stars                             |
| `y`       | Select **yes** (keep)                   |
| `n`       | Select **no** (reject)                  |
| `u`       | Set back to **undecided**               |
| `f`       | Toggle **favorite**                     |
| `e`       | Toggle **enhance requested** (yes-low)  |
| `←` / `→` | Previous / next photo in current sort   |
| `space`   | Next photo (one-handed reviewing)       |
| `esc`     | Close detail view                       |

Outside the modal (grid view only):

| Key      | Action                                          |
|----------|-------------------------------------------------|
| `Enter`  | Open first photo in detail                      |
| `s`      | Submit all staged decisions                     |

Sort: `score (technical)` or `captured`. Filter by score is implicit
through sort order.

#### Staging vs submitting

Every decision (stars / yes-no / favorite / enhance) is **staged**
in the `decisions` table. **Nothing on disk moves** until you click
the green "submit batch (N)" button or press `s`.

The header always shows how many photos still have a pending stage.

When you submit:
1. The decision engine maps `(selected, score_tier)` → action using
   the rule table in [`app/decision/rules.py`](./app/decision/rules.py).
2. A single transactional pass moves each RAW from
   `photos/incoming/` to its target subdirectory.
3. Decisions are marked `applied=True`.

If you do not want to use the UI, you can do the equivalent CLI:
write to the DB by hand (`make shell`, then `sqlite3 /data/cache/session.db`)
and run:

```bash
make submit
```

### Step 4 — Enhance the yes-low set (optional)

For photos where you said **yes** but the technical score was **low**,
the rule engine staged action `enhance_export`. Submitting moved the
RAW into a queue (the `Decision.action` column is `enhance_export`),
but didn't actually run the AI pipeline.

To run it:

```bash
make enhance
```

For each photo:

1. **darktable-cli** develops the RAW (using a matching `.xmp` sidecar
   from `xmp/` if present) to a 16-bit TIFF.
2. The TIFF is downscaled to `RAWCURATOR_ENHANCE_AI_SCALE` × native
   size — default `0.4`, which puts a 24 MP image at ~2.4k × 1.6k,
   the sweet spot for SCUNet at 6 GB VRAM.
3. **SCUNet** denoises (FP16). VRAM cleared.
4. **Real-ESRGAN x2** upscales back toward native (FP16, tiled). VRAM
   cleared.
5. If `score` found faces in this photo, **CodeFormer**
   (`w = 0.7` by default) restores them. VRAM cleared.
6. Lanczos resample to `RAWCURATOR_ENHANCE_TARGET_RES` (default
   `native`).
7. Write a 16-bit TIFF to `photos/exported/<name>.tif` with sRGB v2
   ICC profile.

The original RAW stays in `photos/exported/` alongside the new TIFF.

Performance reference: 24 MP CR3 → ~48 s/photo on RTX 2060 6 GB.

If you want to tune fidelity vs. restoration strength for portraits,
override the CodeFormer weight before running:

```bash
RAWCURATOR_ENHANCE_CODEFORMER_W=0.85 make enhance
```

### Step 4b — Export share-ready JPEGs (optional)

RAWs (~25–50 MB each) and 16-bit TIFFs (~150 MB each) are great for
archival and re-editing, but they are unwieldy for everyday viewing,
phones, social media, or email. The `export-jpeg` step produces an
8-bit JPEG sibling for every kept RAW (`photos/library/*`) and every
enhanced TIFF (`photos/exported/*.tif`).

```bash
make export-jpeg
```

Defaults: quality `92`, native resolution, progressive, 4:2:0 chroma
subsampling. EXIF is copied from the source via `exiftool`; the
output's `Orientation` tag is forced to `1` because rawpy and
darktable have already baked the rotation into the pixels — leaving
the source's `Orientation` tag in place would cause viewers to rotate
the image a second time.

Outputs land at `photos/jpeg/<stem>.jpg`. If the destination already
exists the file is skipped, so re-running is cheap. Pass `--overwrite`
to force a re-encode.

Common variations (run inside the container — `make shell` first, or
prepend env vars to the `make` invocation):

```bash
# Quality 95, cap the long edge at 4000 px for web sharing
RAWCURATOR_JPEG_QUALITY=95 RAWCURATOR_JPEG_LONG_EDGE=4000 make export-jpeg

# Only the enhanced set
raw-curator export-jpeg --source exported

# Only the kept RAWs (no TIFFs)
raw-curator export-jpeg --source library

# Re-encode everything, ignoring existing outputs
raw-curator export-jpeg --overwrite
```

How each source is processed:

- **RAW → JPEG**: `rawpy.postprocess` with `use_camera_wb=True`,
  sRGB output, BT.709 gamma `(2.222, 4.5)` — the same recipe as the
  3000 px previews used in the UI, just at native resolution.
- **TIFF → JPEG**: `tifffile.imread` → drop alpha if present →
  `uint16 >> 8` to 8-bit → Pillow JPEG encode. The enhanced TIFFs are
  already sRGB display-referred so no colour transform is needed.

This step is intentionally last and intentionally optional. It does
**not** touch the RAW/TIFF sources, and it is the only stage whose
output is meant to leave the box as-is.

### Step 5 — Collect your outputs

After submit (and optionally enhance), the working tree looks like:

```
photos/
  incoming/      <- now empty (originals were moved)
  library/       <- yes + high (kept RAWs, untouched)
  archive/       <- yes + low originals (after enhance) + no + high RAWs
  quarantine/    <- no + low (will be wiped by `make reset`)
  exported/      <- enhanced 16-bit TIFFs
  jpeg/          <- share-ready JPEGs (if you ran `make export-jpeg`)
```

**Copy `library/`, `exported/`, and `jpeg/` somewhere safe before resetting.** The
system intentionally has no backup story — that is your job. Example:

```bash
DEST=~/photos/2026-05-shoot
mkdir -p "$DEST"
rsync -a photos/library/ "$DEST/library/"
rsync -a photos/exported/ "$DEST/exported/"
rsync -a photos/jpeg/     "$DEST/jpeg/"
# If you want the no-but-good ones too:
rsync -a photos/archive/  "$DEST/archive/"
```

### Step 6 — Reset for the next session

```bash
make reset          # asks "have you saved anything you need to keep?"
```

Confirms then:
- Deletes `cache/session.db` (and `-wal`/`-shm`).
- Empties `cache/previews/` and `cache/thumbs/`.
- Empties `photos/library/`, `photos/archive/`, `photos/quarantine/`,
  `photos/exported/`, `photos/jpeg/`.
- Runs `alembic upgrade head` to give you a fresh empty schema.
- Leaves `photos/incoming/`, `models/`, and `xmp/` alone.

Skip the confirmation in scripts with `make reset` → `raw-curator reset --force`.

---

## Tuning notes

The score tier (`high` / `low`) is derived in
[`app/decision/rules.py`](./app/decision/rules.py):

```python
combined = 0.6 * technical_score + 0.4 * normalized_aesthetic
tier = "high" if combined >= 0.55 else "low"
```

If too many photos land in `low`, lower the `0.55` threshold (or
rebuild with a different weighting). For portrait-heavy batches, you
likely want to raise the aesthetic weight from `0.4` to `0.5`.

The cluster recommendation uses the same combined score. Override by
manually marking a different cluster member as your `yes` in the UI —
the decision engine respects the UI's choice, not the recommendation.

---

## Common operations

### Re-score after a model change

Models are not session state — they survive `make reset`. To re-score
the same batch with different weights:

```bash
# Edit RAWCURATOR_* env vars or swap weights under models/
make reset          # this wipes the DB, so:
# Re-drop the same RAWs into photos/incoming/ (or rsync them back)
make run
```

### Resume a crashed run

The pipeline is idempotent within a session. Ingest skips files
already in the `photos` table by hash. Filter and score skip rows
that already have the relevant columns populated. Just re-run the
same `make ingest|filter|score|cluster` target.

### Run pipeline against a specific subdirectory

The walker only looks at `photos/incoming/`. Symlink or rsync
subdirectories in:

```bash
ln -s /mnt/nas/wedding-batch-3 photos/incoming/batch-3
make run
```

### Debug a single photo

```bash
make shell
$ python -c "
from sqlalchemy import select
from app.db import session_scope
from app.models import Photo
with session_scope() as s:
    p = s.execute(select(Photo).limit(1)).scalar_one()
    print(p.hash, p.technical_score, p.aesthetic_score, p.cluster_id)
"
```

### Inspect the DB directly

```bash
make shell
$ sqlite3 /data/cache/session.db
sqlite> .tables
sqlite> SELECT hash, technical_score, aesthetic_score FROM photos ORDER BY technical_score DESC LIMIT 10;
```

---

## When things go wrong

| Symptom                                    | What to check                                                         |
|--------------------------------------------|-----------------------------------------------------------------------|
| `make image` hangs on pyiqa install        | Network egress to pytorch CDN; rerun with `--no-cache` if it stays stuck |
| `nvidia-smi` works on host but not in container | Re-run `host-bootstrap.sh`; verify `/etc/cdi/nvidia.yaml` exists      |
| `make score` reports CUDA OOM              | Lower `RAWCURATOR_CLIP_BATCH` (default 8) → 4                          |
| `make enhance` reports CUDA OOM mid-photo  | Lower `RAWCURATOR_ENHANCE_AI_SCALE` (default 0.4) → 0.35              |
| UI thumbnails 404                          | Cache dir not writable — `chmod -R u+rw cache/` on the host           |
| Submit fails partway                       | DB is in WAL mode and transactional; rerun `make submit`; check `decisions.applied` |
| Enhance output looks oversharpened         | Lower `RAWCURATOR_ENHANCE_CODEFORMER_W` (the higher the w, the more identity-preserving but less restoration) |
| RAW files not detected                     | Check the file extension is one of `.CR2 .CR3 .ARW .NEF .DNG .RAF .ORF` (case-insensitive) |

For anything not on the table: `make shell` + `raw-curator info` and
then walk through `app/cli.py` — every command is a thin wrapper around
a job module in `app/<phase>/`.
