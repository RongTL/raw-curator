# Before/after export choice — design

Status: **draft, awaiting owner review** · Branch: `feat/before-after-export` (proposed) · Scope: reorder the pipeline so enhancement runs before review, and let the curator pick — per photo, after seeing both — whether the exported photo is the original or the enhanced version, with an independent keep-RAW toggle.

## 1. Goal

Today the curator decides *blind*: they choose keep-RAW (`yes`) or don't (`no`) **before** the enhancement chain has run, and the only way to see "after" is to run a whole extra `export-jpeg` stage. Both `yes` and `no` still enhance and export; the decision is really only about whether the RAW is archived.

After this change, the flow matches how a photographer actually works: cull/organise, let the AI enhance everything, then review each frame **original vs enhanced side by side** and pick the one to keep. Concretely:

- **Enhancement runs before the review**, on every RAW, so both versions exist to compare the moment review starts.
- The review is a **single screen** with a real before/after **slider** (developed-original vs enhanced), no extra stage to see "after".
- Each photo gets one **version pick** — `Discard` / `Export original` / `Export enhanced` — plus an independent **keep-RAW** toggle.
- **Export** produces the chosen share-JPEG for every kept photo and applies RAW retention. It replaces both `submit` and `export-jpeg`.

Non-goals: changing the enhancement engine's look or steps (that is the separate correctness/Project-B work); scene classification; keeping 16-bit TIFF masters (the owner chose JPEG + optional RAW); a pre-enhance cull pass (owner chose "enhance all, one review").

## 2. Decisions (owner, 2026-09-28)

| # | Decision | Status |
|---|---|---|
| E1 | One unified per-photo decision made **after** enhancement: `Discard` / `Export original` / `Export enhanced`. The old `yes`/`no` keep-RAW routing is retired. | **Made** |
| E2 | "Original" = the RAW **properly developed** (darktable + lens/CA correction) but **not** run through raw-curator's classical+AI enhancement — a fair, full-quality counterpart to "enhanced". Not the ingest preview, not the embedded camera JPEG. | **Made** |
| E3 | Workflow shape = **enhance all, one review**. No pre-enhance cull; the GPU processes every RAW, including frames later discarded. Owner accepts the cost. | **Made** |
| E4 | Retention: the chosen **share-JPEG is always produced**; **keep-RAW is a per-photo toggle**. No 16-bit TIFF master is kept. | **Made** |
| E5 | The review viewer gets a real **side-by-side / slider** before-after comparison in v1 (not just the existing toggle). | **Made** |

## 3. Pipeline after this change

```
OLD:  ingest → filter → score → cluster → [REVIEW keep/no] → submit → enhance → export-jpeg
NEW:  ingest → filter → score → cluster → enhance(all) → [REVIEW pick+keepRAW] → export

  leg 1 (auto-run): ingest → filter → score → cluster → enhance     ← stops here for review
  leg 2 (auto-run): export                                          ← "Export & finish"
```

- `enhance` moves into **leg 1** and runs on every photo with `file_kind == "raw"` (non-RAW skipped — see §8). It no longer reads or depends on any decision, and it **never moves or deletes a RAW**.
- `submit` is **removed** as a stage. Its only real work — moving kept RAWs into `library/` — folds into `export`.
- `export-jpeg` becomes **`export`**: the single "apply my decisions" step.
- Clustering still runs and still marks a per-burst recommendation, but it is now purely a **review aid** (grouping, side-by-side compare, "jump to next undecided"); it gates nothing.
- The `Review` pseudo-stage chip in the timeline moves to sit **after `enhance`**.

## 4. The per-photo decision & data model

Two independent fields per photo, both staged in the DB (nothing touches disk until `export`), matching today's stage-then-apply pattern:

- `export_choice ∈ {undecided, discard, original, enhanced}` — the version pick.
- `keep_raw: bool` — archive the source RAW or not. Only meaningful for kept photos (`original`/`enhanced`).

Resulting retention:

| `export_choice` | Final JPEG in `jpeg/<sub>/X.jpg` | keep_raw = true | keep_raw = false |
|---|---|---|---|
| `enhanced` | from the enhanced result | RAW → `library/<sub>/X.<raw>` | RAW deleted **after** JPEG exists |
| `original` | from the developed RAW | RAW → `library/<sub>/X.<raw>` | RAW deleted **after** JPEG exists |
| `discard` | none | RAW left untouched in `incoming/` | RAW left untouched in `incoming/` |
| `undecided` | none (skipped, reported) | untouched | untouched |

Safety is unchanged from today: a RAW is deleted only **after** its JPEG is verified on disk; a discard is non-destructive (the RAW stays in `incoming/`). Default `keep_raw` is **true** (safe), configurable via a new setting (§9).

### Schema / migration

`Decision` (`app/models.py:139`) gains:

- `export_choice: String(16)` default `"undecided"`
- `keep_raw: Integer` (bool) default `1`

The legacy routing fields (`selected`, `action`, `score_tier`, `enhance_requested`) stop driving anything. `applied` is retained and now set by `export` (was `submit`). One Alembic migration in `db/migrations/` adds the two columns; the DB is wiped on every `make reset` anyway, but `alembic upgrade head` still runs there, so the migration must be real. `rules.py`'s binary table and `ENHANCE_ACTIONS` are removed from the routing path (enhance now selects by `file_kind`, export by `export_choice`); `tier_from_scores` stays as the display-only helper it already is.

## 5. What `enhance` produces (so review is instant)

For every RAW photo, `enhance` writes JPEG artifacts to a cache review area (exact paths finalised in the plan; keyed by hash):

- **`before` (review-res, ~3000 px long edge)** — the developed original: darktable develop + lens/CA correction, converted to display sRGB, **before** `apply_pre_ai`/AI/`apply_post_ai`. This is the "before" the slider shows and the basis for an exported original.
- **`after` (review-res, ~3000 px long edge)** — the enhanced result (the current phase-3 output), converted to sRGB.
- **`after_full` (full-res, final quality)** — the enhanced result at full resolution. Kept because recomputing the AI result is expensive; the review-res `after` is for the slider, `after_full` is what `export` copies when `enhanced` is chosen.

The **full-resolution original is not stored**; it is cheaply **re-developed from the RAW at export** (deterministic — same darktable settings + lens correction as the `before` preview), and only for the subset of photos actually picked `original`. All three JPEGs carry EXIF copied from the source RAW with orientation baked in (reuse `pack_tiff.copy_metadata` / `jpeg_writer` conventions). All are **cache intermediates**, cleaned at `export` and by `make reset`.

Enhancement's existing measure → plan → verify → degraded-retry logic is unchanged; `plan_json`/`verify_json`/`score_q_after`/`degraded` are still persisted per photo (`quality_reports`). What's removed from `enhance`: the decision join in candidate selection, the `photos/exported/*.tif` master write, and the `may_delete_source` RAW deletion (deletion now lives in `export`). A **degraded** verdict no longer just retries silently — it surfaces as a badge in review so the curator leans toward `original`.

## 6. The review screen

Reuse the existing detail viewer (`app/api/static/js/review.js` `DetailModal`, which already has a `b` before/after toggle and per-photo keyboard decisions) and the compare surface (`compare.js`):

- **Slider (E5):** a new before/after split/slider component (developed-original vs enhanced) inside the viewer — drag a handle to wipe between the two aligned images. Both images share aspect ratio and resolution class so the wipe is clean. Falls back to the plain toggle if one side is missing (non-RAW / enhance-failed).
- **Version controls:** buttons + keys `o` = original, `e` = enhanced, `x` = discard; `r` toggles keep-RAW; `.` jumps to next undecided. These replace the `y`/`n`/`u` keep-RAW controls.
- **Grid tiles** (`PhotoTile`) show a version badge (`ORIG` / `ENH` / `discarded`) and a small "RAW kept" indicator; existing `degraded` / `flagged` badges stay.
- **Bulk header controls** (repurpose `POST /api/decide/all`): "use enhanced for all", "use original for all", "keep all RAWs", "keep no RAWs".
- The **"after" is available immediately** after enhance (from the cached `after` JPEG) — this removes the current "run export-jpeg just to see after" round-trip, which was the "jumping between steps" pain.

### API surface

- `POST /api/decide` (`DecisionIn`, `decide.py:18`) gains `export_choice?` and `keep_raw?` (partial patch, like today's fields).
- `POST /api/decide/all` takes `{export_choice}` and/or `{keep_raw}` for the bulk controls.
- Photo/decision serializers (`app/api/serializers.py`) gain `export_choice`, `keep_raw`, `before_url`, `after_url` (cache URLs for the review JPEGs; `after_url` absent when enhance failed). `enhanced`/`degraded`/`q_after` badges stay.
- Cache review JPEGs are served through the existing `/cache` static mount (new subdir), via a helper like the existing `_urls.cache_url`.

## 7. Export (replaces submit + export-jpeg)

`app/export/` `run_export` (renamed from `run_jpeg_export`), driven by the DB, not the filesystem:

For each `Decision` with `export_choice` in (`original`, `enhanced`) and `applied == 0`:

1. Build the JPEG:
   - `enhanced` → copy/encode from the cached `after_full` JPEG.
   - `original` → re-develop the RAW (darktable + lens/CA correction, same as the `before` preview) → encode JPEG.
   - Destination `settings.photos / jpeg_subdir / relative_subpath(src).with_suffix(".jpg")` (unchanged mirroring via `app/paths.py`).
2. Apply RAW retention **after** the JPEG exists:
   - `keep_raw` → move RAW to `library/<sub>/…` via `executor.apply_moves` (the atomic mover `submit` used).
   - `else` → delete the RAW.
3. Set `Decision.applied = 1`, update `Photo.source_path` when moved.

`discard`/`undecided` produce no JPEG and touch nothing. This removes the current **library-vs-exported → same-jpeg collision** (progress.py:121, jpeg_job) entirely, because export now has exactly one source per photo. Per-item exceptions are caught and logged (existing convention); a failed JPEG never triggers a RAW delete.

## 8. Edge cases

- **Non-RAW inputs** (`jpeg`/`heic`/`tiff`/`png`): no enhanced version. `enhance` produces only a `before`/original (a developed/decoded JPEG); review offers `Discard` / `Export original` only (enhanced disabled), `keep_raw` toggle keeps the original file. Export original re-encodes (or byte-copies a JPEG, as `jpeg_writer` already does).
- **Enhance failed for a photo:** no `after` JPEG; review shows original-only with an "enhance failed" note; curator picks `original` or `discard`.
- **Degraded enhanced result:** `after` still produced, badged `degraded`; curator decides. No automatic RAW deletion is involved anymore, so the old "keep RAW if degraded" guard is moot.
- **Undecided at export:** skipped and counted in the summary (like today's undecided).
- **Re-run of export:** `applied` gates re-processing; already-exported rows are skipped.

## 9. Config, reset, orchestration, tests

- **Config** (`app/config.py`, `.env.example`): `RAWCURATOR_REVIEW_LONG_EDGE` (default `3000`) for review JPEG size; `RAWCURATOR_KEEP_RAW_DEFAULT` (default `true`). Every new setting gets a reader (project rule). Existing `jpeg_quality`/`jpeg_long_edge`/`enhance_target_res` still govern final export.
- **Stages** (`app/orchestrator/stages.py`): `STAGES` = leg 1 `ingest, filter, score, cluster, enhance`; leg 2 `export`. `AUTO_RUN_LEGS` updated; `submit`/`export-jpeg` names removed. `progress.py`: enhance progress = enhanced-JPEG count over all RAW photos; export progress = JPEG count over decided photos; drop the submit handler and the library/exported dedup.
- **CLI/Make/compose**: `raw-curator` drops `submit`, keeps `enhance` (now all-RAW), replaces `export-jpeg` with `export`; `make` targets and `compose.yaml`/`README` command tables follow. `run --auto` two-leg behaviour maps onto the new legs.
- **Reset** (`app/orchestrator/reset.py`): add the new cache review dir to the wipe list; `photos/exported/` is no longer written but stays wiped for safety; `incoming/` still never touched.
- **Tests**: rewrite `tests/test_decision_rules.py` around `export_choice`; new tests for export source selection (original re-develop vs enhanced copy), retention (keep_raw move vs delete, delete-only-after-JPEG), non-RAW original-only, migration up. `real_raw`/GPU-marked enhance tests unaffected in look; no contact-sheet needed since the enhancement look does not change.
- **Docs**: `README.md`, `USER_GUIDE.md`, `CLAUDE.md` updated for the new pipeline order, decision model, and stage names.

## 10. Risks / open questions

- **GPU time (accepted, E3):** enhancing frames that get discarded is wasted work on the 2060. Mitigation available later: an optional pre-enhance "obvious reject" cull, out of scope now.
- **Cache disk during review:** `before` + `after` + `after_full` JPEGs per RAW live in cache across the whole review window. JPEGs (not TIFFs), so far smaller than today's `exported/` masters, but non-trivial for large batches. Alternative if it bites: drop `after_full` and re-run… no — the AI result can't be cheaply recomputed, so `after_full` must persist; the tunable lever is its quality/resolution.
- **Original consistency:** the exported original is re-developed at export, not the same bytes as the reviewed `before`. It matches because darktable develop + lens correction are deterministic given the same XMP/params. If this ever drifts, the fallback is to persist a full-res `before` JPEG at enhance time (one more JPEG per photo).
- **Losing 16-bit masters:** per E4 no `exported/*.tif` master is kept. Flagged in case the owner later wants an "also keep TIFF" retention option (easy to add as a third keep-toggle).
