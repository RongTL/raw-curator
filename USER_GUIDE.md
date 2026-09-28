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
[2] make serve → open the Control Center at http://<host>:8080
[3] Auto-run — leg 1 (ingest → filter → score → cluster → enhance),
    then it stops for human review
[4] review before/after, pick original/enhanced/discard + keep-RAW
    per photo, click Export selected — leg 2 (export) runs unattended
[5] copy outputs out of photos/jpeg/ (+ photos/library/ for kept RAWs)
[6] New batch (type RESET) → DB + cache + working dirs are wiped
[7] next batch is a clean slate; models/ is kept
```

There is **no re-curation across sessions**, **no cross-batch search**,
and **no audit log retention** beyond the active session.

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
tables (9): ['alembic_version', 'cluster_members', 'clusters',
             'decisions', 'faces', 'photo_embeddings', 'photos',
             'quality_reports', 'session_meta']
```

---

## Session walkthrough

### Step 0 — Prepare a clean slate

```bash
cd ~/projects/raw-curator
make reset          # confirms before wiping; wipes DB + cache + working dirs
```

If the Control Center is already running, the header's **New batch**
button does the same wipe from the browser (you must type `RESET` to
confirm).

After this:
- `cache/session.db` is freshly migrated (empty schema, including the
  `quality_reports` table the Auto Enhancement Engine writes into).
- `photos/{library,exported,jpeg}/` are empty.
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

- **RAW**: `.CR2 .CR3 .CRW .NEF .NRW .ARW .SRW .SR2 .SRF .RAF .ORF
  .RW2 .DNG .PEF .RAW .X3F .RWL .3FR .IIQ .MEF .MOS .MRW` (developed
  via rawpy with camera white balance, sRGB output).
- **Already-developed**: `.JPG .JPEG .TIF .TIFF .HEIC .HEIF .PNG`
  (decoded via Pillow/tifffile with EXIF auto-rotation; HEIC requires
  `pillow-heif`, included in the container image).

Each photo's file kind is recorded in the DB (`Photo.file_kind`) and
shown as a chip on the tile + detail header. Filenames are preserved
end-to-end. Mixed batches are fine.

Note: `make enhance` only runs the Auto Enhancement Engine on RAW
sources, and it runs on **every** RAW before you review (no decision is
needed first). Any JPEG/TIFF/HEIC photo is skipped with a warning (the
engine expects sensor data; running it on 8-bit display-referred pixels
gives worse results than just leaving the file as-is). A skipped non-RAW
still shows up in review with an original-only choice (no enhanced
version to compare against), and its original file is never modified by
enhance.

### Step 2 — Start the Control Center and Auto-run

```bash
make serve          # foreground; Ctrl-C to stop
```

Open `http://<host>:8080` in any browser on your network. The header
shows a horizontal stage timeline:

```
Ingest → Filter → Score → Cluster → Enhance → [Review] → Export
```

- Click **Auto-run** to run leg 1 (ingest → filter → score → cluster →
  enhance) in one shot. It stops automatically at **Review** and waits
  for you — enhancement has already produced a before/after for every
  RAW, but nothing is exported and no RAW is moved or deleted without a
  human decision.
- Or click any stage in the timeline to run just that stage — useful
  for debugging or re-running one phase. Only one stage runs at a
  time.
- **Stop** in the header cancels the running stage (and the rest of
  an in-flight auto-run).
- Each stage runs as an isolated subprocess: its output streams live
  into the stage panel and is written to `cache/logs/<stage>-NNN.log`.
  If a stage crashes (e.g. GPU OOM), it turns red in the timeline
  with the exit code and log tail — the server survives, so fix the
  cause (see "When things go wrong") and re-run that stage.
- The docked resource bar at the bottom shows live CPU / RAM / GPU /
  VRAM / disk with sparklines; expand it for 60-second charts. It
  warns when free space on the photos volume drops below
  `RAWCURATOR_MONITOR_DISK_WARN_FREE_GB` (default 50 GB).

#### Advanced: headless CLI

The analysis stages are available without the UI:

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

To finish leg 1 headless, run enhancement over every RAW:

```bash
make enhance        # ~48 s/photo on RTX 2060; writes before/after renders to cache/enhanced/
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

- **enhance** (the last stage of leg 1): runs the Auto Enhancement
  Engine on **every RAW**, before you review, so both versions exist to
  compare the moment review starts. Non-RAW inputs are skipped
  (original-only in review). It reads no decision and never moves or
  deletes a RAW. For each RAW:

  1. **darktable-cli** develops the RAW with its **sigmoid** workflow
     (using a matching `.xmp` sidecar from `xmp/` if present) to a 16-bit
     **linear Rec.2020** intermediate, loaded as float32 RGB in `[0, 1]`.
     Lens distortion and chromatic aberration are corrected per-frame
     from EXIF via lensfun (`RAWCURATOR_ENHANCE_LENS_CORRECTION`,
     default on); frames whose lens lensfun cannot resolve pass through
     uncorrected. This developed, un-enhanced frame is the **before**.
  2. The engine **measures quality** across five dimensions — exposure,
     dynamic range, color, sharpness, noise — and writes a row into the
     `quality_reports` table (composite Q score + per-dimension
     sub-scores + raw metrics).
  3. The engine **builds an ordered plan** of classical + AI steps from
     those measurements (`app/enhancement/engine/decision.py`). Each
     step is included only when its indicator metric crosses a spec
     threshold, and step strength scales with the measured deficit, so
     already-clean inputs are not over-processed.
  4. The runner **executes the plan** in three passes:
     - Pre-AI classical steps at full native resolution in float32
       (e.g. exposure gamma, shadow lift, highlight recover, backlit
       recovery, gray-world WB, saturation, global tone compression).
     - AI steps at native resolution by default
       (`RAWCURATOR_ENHANCE_AI_SCALE`, default `1.0` — a 24 MP image
       stays ~6k × 4k and peaks around 5.5 GB VRAM during SCUNet on a
       6 GB card; drop to `0.85`/`0.7` if OOM). Each model runs once
       over the batch and sees only an 8-bit sRGB copy; the engine then
       merges just that model's **change (delta)** back into the float
       master, so the master is never quantised. SCUNet denoise
       (blended at `RAWCURATOR_ENHANCE_DENOISE_STRENGTH`, default
       `0.75`) runs when the frame is noisy; Real-ESRGAN x2 (blended at
       `RAWCURATOR_ENHANCE_REALESRGAN_FIDELITY`, default `0.7`) runs
       **only when the target enlarges the source or the source is
       small** (long edge < `RAWCURATOR_ENHANCE_SR_MIN_LONG_EDGE`);
       CodeFormer (weight `RAWCURATOR_ENHANCE_CODEFORMER_W`, default
       `0.85`) runs **only on faces that are small, soft, or in a noisy
       frame**, with an ArcFace identity guard that pastes the original
       face back if the restored one drifts too far
       (`RAWCURATOR_ENHANCE_FACE_MIN_SIMILARITY`). Only one model is
       resident at a time (VRAM is cleared between them), and after the
       AI pass the frame is resized to `RAWCURATOR_ENHANCE_TARGET_RES`
       (default `200%`, keeping Real-ESRGAN's 2× output). The result is
       the **after**.
     - Post-AI classical steps at native (e.g. unsharp mask, CLAHE
       local contrast). There is **no** final tone-mapping step — the
       darktable sigmoid workflow already handled the highlight roll-off
       at develop time.
  5. **Verify & render** — the engine re-measures the result and
     compares it to the input. If it degraded the frame (Q dropped,
     highlights blown, image gone flat/black), it retries once with a
     *safe plan* (tone steps only, no AI) and keeps whichever result
     scores higher; a still-degraded result is badged in review so you
     lean toward the original. It then writes four JPEGs per photo into
     `cache/enhanced/` (EXIF copied from source, orientation baked in):
     `<hash>.before.jpg` and `<hash>.after.jpg` at review resolution
     (`RAWCURATOR_REVIEW_LONG_EDGE`, default 3000) for the slider, plus
     full-res `<hash>.before.full.jpg` / `<hash>.after.full.jpg` that
     `export` later copies. **No TIFF master is written, and no RAW is
     moved or deleted** — retention is decided later, at export.

  Performance reference: 24 MP CR3 → ~48 s/photo on RTX 2060 6 GB. To
  tune fidelity vs. restoration for portraits, override a knob before
  running, e.g. `RAWCURATOR_ENHANCE_CODEFORMER_W=0.9 make enhance`.

### Step 3 — Review before/after and pick a version

When leg 1 finishes, the timeline stops at **Review** and the review
panel opens: a toolbar with **All** / **Clusters** view tabs, sort,
bulk controls and the green **Export selected** button; below it a grid
of thumbnails; and a full-screen detail modal that opens when you click
a tile or press Enter. Enhancement has already run, so every RAW has a
**before** (developed original) and an **after** (enhanced) ready to
compare.

The bulk controls in the header stage the same choice across the whole
batch in one shot (a confirm dialog spells out the destructive ones):
**use enhanced for all**, **use original for all**, **discard all**,
**keep all RAWs**, **keep no RAW**. You can still flip individual photos
afterwards. In the Clusters view, per-cluster "keep best / reject rest"
buttons do the same within one cluster.

#### Grid view

Each tile shows:
- Thumbnail (lazy-loaded from `cache/thumbs/`).
- Stars (top right) — the per-photo rating you assign.
- `REC` badge (top left) — the cluster recommendation.
- A version badge — `ORIG` / `ENH` / `discarded` — for your current
  pick (no badge = undecided), plus a small "RAW kept" indicator.
- Tech/aesthetic scores along the bottom.

Click a tile or press **Enter** to open the first photo in detail
view.

#### Detail view (full-screen modal)

Center: a **before/after slider** — drag the handle to wipe between the
developed original and the enhanced result (both served from
`cache/enhanced/`, ready the moment enhance finished — there is no
separate step to "see the after"). Press `b` to cycle
slider → before-only → after-only; a non-RAW or enhance-failed photo
falls back to the single image it has. Bottom panel: scores, the
version + keep-RAW controls, EXIF, cluster info; a `degraded` badge
flags an enhanced result the engine was not confident about.

Keyboard shortcuts inside the modal:

| Key       | Action                                       |
|-----------|----------------------------------------------|
| `1`–`5`   | Set stars                                    |
| `0`       | Clear stars                                  |
| `o`       | Pick **original** (export the developed RAW) |
| `e`       | Pick **enhanced** (export the AI result)     |
| `x`       | **Discard** (export nothing)                 |
| `r`       | Toggle **keep-RAW** for this photo           |
| `b`       | Cycle before/after view mode                 |
| `f`       | Toggle **favorite**                          |
| `.`       | Jump to the next undecided photo             |
| `←` / `→` | Previous / next photo in current sort        |
| `space`   | Next photo (one-handed reviewing)            |
| `esc`     | Close detail view                            |

Sort: `score (technical)` or `captured`. Export happens via the green
**Export selected** button in the review toolbar (no keyboard shortcut).

#### Staging vs exporting

Every choice (stars / version pick / keep-RAW / favorite) is **staged**
in the `decisions` table. **Nothing on disk moves** until you click the
green **Export selected (N)** button in the review toolbar.

The button shows how many photos have a pending keeper choice
(`original`/`enhanced` not yet applied). Clicking it opens a
confirmation dialog — it reminds you that photos with keep-RAW **off**
get their source RAW deleted once the JPEG is written — and then starts
auto-run leg 2: **export**, unattended.

Each photo carries two independent, staged fields
([`app/decision/export_rules.py`](./app/decision/export_rules.py)):

- `export_choice` ∈ `undecided` / `discard` / `original` / `enhanced`
  — which version, if any, becomes the share JPEG.
- `keep_raw` (default on, from `RAWCURATOR_KEEP_RAW_DEFAULT`) — archive
  the source RAW to `photos/library/`, or delete it after the JPEG is
  written. Only meaningful for kept photos.

If you prefer the CLI, stage the choices in the DB by hand (`make
shell`, then `sqlite3 /data/cache/session.db`) and run:

```bash
make export
```

### Step 4 — Leg 2: export

After **Export selected**, leg 2 runs the single **export** stage
unattended. Watch progress in the timeline; it streams its log into the
UI. **Stop** cancels it.

For every photo you picked `original` or `enhanced`, export writes one
share-ready JPEG into `photos/jpeg/<sub>/<stem>.jpg` (subfolders under
`incoming/` are mirrored), then applies RAW retention **after** the JPEG
exists on disk:

- `enhanced` → the JPEG is copied from the cached full-res enhanced
  render (`cache/enhanced/<hash>.after.full.jpg`).
- `original` → the JPEG is copied from the cached full-res developed
  render (`cache/enhanced/<hash>.before.full.jpg`); for a non-RAW source
  it is re-encoded from the original file itself.
- `keep_raw` **on** → the source RAW is moved into `photos/library/`
  (`Photo.source_path` updated).
- `keep_raw` **off** → the source RAW is **deleted** — but only once the
  JPEG is on disk, so a failed export leaves the original in place.
- `discard` / `undecided` → nothing is written and no RAW is touched.

`applied` gates re-runs, so re-running `make export` only processes rows
you have not exported yet. JPEG defaults: quality `92`
(`RAWCURATOR_JPEG_QUALITY`), native resolution
(`RAWCURATOR_JPEG_LONG_EDGE=0`), progressive, 4:2:0 chroma subsampling;
EXIF is copied from the source and the output `Orientation` tag is
forced to `1` because rawpy/darktable have already baked the rotation
into the pixels.

The equivalent headless CLI, if you prefer to run leg 2 by hand:

```bash
make export
```

Common variations (prepend env vars to the `make` invocation, or `make
shell` first):

```bash
# Quality 95, cap the long edge at 4000 px for web sharing
RAWCURATOR_JPEG_QUALITY=95 RAWCURATOR_JPEG_LONG_EDGE=4000 make export
```

RAWs (~25–50 MB each) are unwieldy for everyday viewing, phones, social
media, or email; the JPEG is the share-ready sibling meant to leave the
box. Keep the RAW (via keep-RAW) only when you want the negative for
archival or re-editing.

### Step 5 — Collect your outputs

After export, the working tree looks like:

```
photos/
  incoming/      <- kept RAWs have moved to library/; keep-RAW-off RAWs are deleted; discarded RAWs remain
  library/       <- RAWs you kept via the keep-RAW toggle
  jpeg/          <- share-ready JPEGs (the chosen version per kept photo)
```

**Copy `library/` and `jpeg/` somewhere safe before resetting.** The
system intentionally has no backup story — that is your job. Example:

```bash
DEST=~/photos/2026-05-shoot
mkdir -p "$DEST"
rsync -a photos/library/  "$DEST/library/"
rsync -a photos/jpeg/     "$DEST/jpeg/"
```

### Step 6 — Reset for the next session

From the UI: click **New batch** in the header and type `RESET` in
the confirmation dialog. From the CLI:

```bash
make reset
```

Both do the same wipe. `make reset` is non-interactive — it deletes
immediately:
- Deletes `cache/session.db` (and `-wal`/`-shm`).
- Empties `cache/previews/`, `cache/thumbs/`, and `cache/enhanced/`
  (the before/after render JPEGs).
- Empties `photos/library/`, `photos/exported/` (legacy — no longer
  written, but still wiped for safety), and `photos/jpeg/`.
- Runs `alembic upgrade head` to give you a fresh empty schema
  (including `quality_reports`).
- Leaves `photos/incoming/`, `models/`, and `xmp/` alone.

The in-container CLI form (`raw-curator reset`) prompts for confirmation
unless invoked as `raw-curator reset --force`.

---

## Tuning notes

The per-photo choice is `export_choice` (`original`/`enhanced`/`discard`)
plus an independent `keep_raw` toggle; there is no score-tier routing.
Clustering is now purely a review aid (grouping + a recommendation) and
gates nothing. The technical/aesthetic blend used for the recommendation
and the display tier lives in
[`app/scoring/combined.py`](./app/scoring/combined.py):

```python
combined = 0.6 * technical_score + 0.4 * normalized_aesthetic
tier = "high" if combined >= 0.55 else "low"
```

Override by picking a different cluster member in the UI — export
respects your per-photo choice, not the recommendation.

The Auto Enhancement Engine's plan is driven by the per-photo
`QualityReport` (see `app/enhancement/engine/`). To inspect what the
engine decided for a photo, query `quality_reports` directly:

```bash
sqlite> SELECT photo_hash, score_q, score_exposure, score_dynamic_range,
   ...> score_color, score_sharpness, score_noise
   ...> FROM quality_reports ORDER BY score_q DESC LIMIT 10;
```

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
| `make enhance` reports CUDA OOM mid-photo  | Lower `RAWCURATOR_ENHANCE_AI_SCALE` (default 1.0) → 0.85 → 0.7 → 0.5  |
| A stage turns red in the Control Center timeline | Click the stage for the exit code + log tail; full log at `cache/logs/<stage>-NNN.log`. Fix the cause, then re-run the stage — the server survives stage crashes |
| Resource bar shows a disk warning          | Free space on the photos volume is below `RAWCURATOR_MONITOR_DISK_WARN_FREE_GB` (default 50 GB) — clear space or set `RAWCURATOR_ENHANCE_TARGET_RES=native` for ~4x smaller enhanced renders |
| UI thumbnails 404                          | Cache dir not writable — `chmod -R u+rw cache/` on the host           |
| Export fails partway                       | DB is in WAL mode and transactional; `applied` gates re-runs, so just rerun `make export`; check `decisions.applied` |
| Enhance output looks oversharpened         | Lower `RAWCURATOR_ENHANCE_REALESRGAN_FIDELITY` toward `0.3` (softer); for faces, *raise* `RAWCURATOR_ENHANCE_CODEFORMER_W` toward `0.95` (higher w = more faithful to original skin) |
| Enhance output looks waxy / airbrushed     | Raise `RAWCURATOR_ENHANCE_CODEFORMER_W` toward `0.95`, and lower `RAWCURATOR_ENHANCE_DENOISE_STRENGTH` to `0.5–0.6` to keep more original micro-texture |
| Backlit subject still too dark             | Raise `RAWCURATOR_ENHANCE_BACKLIT_SHADOW_LIFT` toward `0.6` (>0.7 starts looking HDR); confirm `RAWCURATOR_ENHANCE_BACKLIT_RECOVERY=true` |
| Backlit recovery blew out the sky          | Raise `RAWCURATOR_ENHANCE_BACKLIT_HIGHLIGHT_PROTECT` toward `0.3` |
| RAW files not detected                     | Check the file extension is one of `.CR2 .CR3 .ARW .NEF .DNG .RAF .ORF` (case-insensitive) |

For anything not on the table: `make shell` + `raw-curator info` and
then walk through `app/cli.py` — every command is a thin wrapper around
a job module in `app/<phase>/`.
