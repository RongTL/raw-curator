# Enhancement engine correctness pass — design

Status: **approved 2026-09-27** (D1–D4 as proposed) · Branch: `feat/enhance-correctness` · Scope: "Project A" (steps 1–3 of the roadmap agreed on 2026-09-27). The content-aware redesign (scene analysis, policy layer, masks) is Project B and is out of scope here except where noted as an interim guard.

## 1. Goal

Make the Auto Enhancement Engine reason about correct numbers and stop doing things a retoucher would never do to a well-exposed frame, without yet changing its architecture. After this pass:

- the engine measures and edits in **linear Rec.2020** (decision D1, made by the owner);
- every output TIFF is a proper master: 16-bit, ICC-tagged, with the RAW's EXIF/XMP;
- the always-on steps that fight the photograph (second tone map, unconditional super-resolution, unconditional face rebuild, gray-world at full strength, denoise on ISO 100) become conditional or restrained;
- every run records what it did and verifies the result before a source RAW may be deleted;
- the batch runs step-major so each AI model loads once.

Non-goals: scene classification, semantic masks, per-scene presets, a new UI. Those are Project B.

## 2. Evidence (25-frame corpus, Canon EOS R8, `tests/data/corpus/`)

Measured on 2026-09-27 with the current code; artefacts under `tests/data/_review/`.

1. **darktable-cli 4.6.1 exports sRGB by default** (`ProfileDescription: sRGB` on the TIFF). `develop_full.darktable_cli` has an `out_colorspace="linear_rec2020"` parameter that is never passed. The engine then sRGB-encodes the data a second time before measuring. On `IMG_0030` the engine sees mean luma 204/255; the true display value is 166; the linear value is 0.47. Across the corpus today's mean-luma readings are 129–205, corrected they are 68–166. Q shifts by up to 25 points per image. The `exposure_gamma` sign flips on one frame; the exposure sub-score is wrong on all of them.
2. **Two tone mappers.** darktable applies its scene-referred tone mapper; `tone_map_final` (filmic shoulder 0.88, toe 0.02) then runs unconditionally on 25/25 frames.
3. **Rules that fire on nearly everything** (corrected input): `white_balance` 22/25, at the 0.85 cap on 11 of them, including the deep-blue-sky monument (`1176/1177`), the blue-hour park (`0958`) and the tungsten cellar (`0098`); `scunet_denoise` 19/25, including six ISO 100 daylight frames (`0037 0242 0581 0604 1112 1188`); `clahe_local_contrast` 14/25, usually at the maximum clip 3.0; `realesrgan_upscale` 25/25.
4. **Face boxes are in preview coordinates** (≤3000 px long edge) but applied to the 6000 px TIFF; the skin-hue metric samples the wrong region.
5. **Output TIFF has no ICC profile and no EXIF**; `export-jpeg` copies EXIF from that TIFF, so enhanced JPEGs carry no date, camera, lens or GPS.
6. **AI steps run on 8-bit sRGB replacements** of the float image; the post-AI polish then works on 8-bit-derived data (banding risk in skies).
7. **Each AI model is loaded from disk per photo.**
8. **The plan is never persisted or verified**; the RAW-deletion guard is a mean/std heuristic.
9. **XMP lookup is by stem only** (`xmp/<stem>.xmp`), so `IMG_0001` in two subfolders collide.
10. **No lens, chromatic-aberration or profiled-noise correction** unless the user supplies an XMP.

## 3. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Working space is linear Rec.2020, float32 in [0,1], from darktable to the final write. | **Made** |
| D1a | Master TIFF is written as 16-bit linear Rec.2020 with the ICC profile darktable produced embedded, plus EXIF/XMP copied from the RAW. Share JPEGs are converted to sRGB with a real colour transform. | **Made** |
| D2 | darktable owns lens/CA correction, highlight reconstruction, ISO-profiled denoise and the single tone mapper via a checked-in baseline applied when no user XMP exists. SCUNet becomes a residual-noise step. | **Made**, mechanism chosen by the spike (§5.2) |
| D3 | Real-ESRGAN runs only when it is needed for the output: target resolution above native, or source long edge below 3000 px. Off by default at `native`. | **Made** |
| D4 | Default look is "as shot, cleaned up": fix measurable defects, never impose contrast or saturation the photographer did not shoot. | **Made** |

All four were approved by the owner on 2026-09-27.

## 4. Architecture after this pass

```
RAW ──darktable-cli (baseline XMP unless user XMP) ──► 16-bit linear Rec.2020 TIFF + ICC
      │
      ▼  load → float32 linear                      (measure once, re-measure at the end)
  measure_all ─► score_report ─► plan_from_report(report, exif, faces)   plan stored as JSON
      │
      ▼  pass 1: classical, float32 linear, native size
      ▼  pass 2: AI steps, step-major across the batch:
              linear → sRGB 8-bit → model → delta back in linear   (never a replacement)
      ▼  pass 3: classical polish, float32 linear
      ▼  verify: re-measure; degraded → retry with safe plan; never delete RAW if still degraded
  write_tiff16(linear Rec.2020 + ICC) → exiftool copies EXIF/XMP from the RAW
```

Colour boundaries are explicit and live in one module (`app/enhancement/colorspace.py`): `rec2020_linear_to_srgb`, `srgb_to_rec2020_linear`, `encode_srgb`, `decode_srgb`. Anything 8-bit is display-referred sRGB; anything float is linear Rec.2020.

## 5. Design by step

### 5.1 Step 1 — colour pipeline and correctness (no new models)

**Develop.** `darktable_cli(...)` passes `--icc-type LIN_REC2020` before `--core`; the dead `out_colorspace` parameter goes. The ICC bytes are read from the developed TIFF (tag 34675) and carried through to the output writer, so the profile is always the one darktable used.

**Metrics.** Input contract becomes "linear Rec.2020 float". `_luma_u8` keeps sRGB-encoding the linear luma; that is now correct and matches the spec thresholds. `color_metrics` stays in linear. Docstrings and the CLAUDE.md phase description are updated; the misleading `to_float01` name becomes `as_linear_float01` with a dtype assertion.

**Tone.** `tone_map_final` is removed from the planner. `filmic_tone_map` stays as a library function. Anything that clips after the chain is handled by the verification step, not by a blanket curve.

**Faces.** `_candidates` scales each box from preview pixels to TIFF pixels: `scale = tiff_long_edge / preview_long_edge`, with the preview size read from the cached JPEG header. Boxes are clamped to the frame.

**Output master.** `write_tiff16(arr, out, *, icc: bytes | None)` embeds the profile via tifffile `extratags`. A new `copy_metadata(raw, tiff)` in `pack_tiff.py` runs exiftool `-TagsFromFile RAW -EXIF:all -XMP:all --Orientation -Orientation=1` (the TIFF is already upright). Both are covered by tests that read the tags back with exiftool.

**Share JPEG and TIFF ingest.** `decode.load_tiff_rgb8` becomes profile-aware: if the TIFF carries our linear Rec.2020 profile, convert with the matrix and OETF; otherwise treat as sRGB (today's behaviour). `export-jpeg` therefore renders correct colour from the new masters without other changes.

**XMP.** Lookup order: `xmp/<relative subdir>/<name>.<ext>.xmp` (darktable's own convention, mirrored), then legacy `xmp/<stem>.xmp` with a one-time warning.

**Persist and verify.** Migration `0004`: `quality_reports.plan_json TEXT`, `verify_json TEXT`, `score_q_after REAL`. `enhance_job` stores the plan (steps, params, reasons) before running and the verification result after. Verification re-runs `measure_all`/`score_report` on the result and flags *degraded* when any holds: `Q_after < Q_before − 5`, `highlight_clip_after > 2 × highlight_clip_before + 0.01`, `mean luma < 0.03`, or `std < 0.02`. A degraded result triggers one retry with the *safe plan* (exposure/highlight steps only, no AI, no CLAHE, no sharpening). If still degraded, the better of the two by Q is written, the row is flagged, and the `enhance_only` RAW is **not** deleted. The photo detail panel shows the recipe and the before/after Q.

**Step-major execution.** `run_enhancement` becomes three phases over the batch: (1) develop → measure → plan → pass 1, writing a float16 `.npy` intermediate per photo under `cache/enhance/`; (2) for each AI model in order, load once, apply to every photo whose plan needs it, update the intermediate, unload; (3) pass 3 → verify → write → cleanup. Per-photo failures mark that photo failed and the batch continues (existing `EnhanceSummary`). Cancellation between phases leaves intermediates that the next run reuses or the reset wipes.

### 5.2 Step 2 — baseline development (spike first)

**Question to answer cheaply:** what is the most reliable way to make `darktable-cli` apply a fixed module set to any RAW in a fresh container? Candidates, in order of preference: (a) a checked-in sidecar XMP template passed as the `<xmp>` argument; (b) a `.dtstyle` imported into a repo-shipped `--configdir`; (c) `--conf` keys for auto-applied presets. The spike develops three corpus frames each way and compares the histories darktable reports. Acceptance: one mechanism that applies lens correction (auto lens detection), chromatic aberrations, highlight reconstruction (inpaint opposed), denoise (profiled, ISO-aware), colour calibration (CAT16, camera WB) and sigmoid, on all three frames, with no per-image edits.

**After the spike:** ship the baseline under `app/enhancement/darktable/`, apply it when no user XMP exists, and document how a user XMP overrides it. `scunet_denoise` then triggers on *residual* noise only: `σ > 2` **and** (`ISO ≥ 800` **or** `σ > 4`). ISO comes from `Photo.iso`, which enhance_job passes into the planner.

### 5.3 Step 3 — the detail path

**AI as a delta, never a replacement.** For each AI step: `x8 = encode_srgb(rec2020_to_srgb(x_lin))` → model → `y_lin = srgb_to_rec2020(decode_srgb(y8))`; the applied result is `x_lin + strength × (y_lin − x_lin_roundtrip)` where `x_lin_roundtrip` is `x8` decoded the same way, so quantisation cancels. For ×2 super-resolution the base is a Lanczos ×2 of the float image and the delta is added at 2×. The float master therefore keeps its 16-bit gradients and wide gamut; the models still see the sRGB 8-bit domain they were trained on.

**Real-ESRGAN gating (D3).** Planner adds it only when `enhance_target_res` resolves above native or the source long edge is below 3000 px.

**CodeFormer gating and identity guard.** Planner adds it only if at least one face is *degraded*: box long edge below 300 px at native, or Laplacian variance of the face crop below 100, or the frame's noise trigger fired. At run time each restored face crop is re-embedded with InsightFace (already in the image, loaded once in the step-major phase) and compared with the stored ArcFace embedding; cosine similarity below 0.5 reverts that face. Fixed weight stays 0.85.

**Restraint rules (interim until Project B's policy layer).**
- White balance: estimate from near-neutral pixels (chroma below a threshold in linear RGB) rather than the whole frame; trigger at cast > 0.12; strength capped at 0.5; skipped when fewer than 2% of pixels are near-neutral.
- CLAHE: clip capped at 2.0; skipped when faces are present.
- Saturation: measured as OKLCh chroma; the boost branch is skipped when mean chroma is near zero (monochrome intent).
- Unsharp mask: measured on the sharpest 25% of 16×16 blocks rather than the whole frame, so bokeh does not trigger global sharpening.

## 6. Testing and review

- **Unit tests first** for every new function: colour transforms (round-trip, known primaries), box scaling, ICC embed/read-back, exiftool copy, XMP resolution order, planner gates (ISO, face size, target resolution, neutral-pixel WB), verification thresholds and the safe-plan retry, delta merge (identity when model is identity), step-major phase bookkeeping.
- **Corpus harness** `tests/test_engine_corpus.py` (`real_raw` marker): per-image expectations in `tests/data/corpus_expectations.yaml`, e.g. `0958: no white_balance`, `0037: no scunet_denoise`, `1176: clahe clip ≤ 2.0`, `0030: highlight_recover present`. Asserts plans, not pixels.
- **Contact sheets** `scripts/contact_sheet.py` renders before/after crops (full frame + a 100% crop on the subject) for every corpus frame into `tests/data/_review/`. Steps 1, 2 and 3 are each reviewed by the owner on these sheets before merge; green tests alone do not merge a look change.
- `make test`/`make lint`/`make typecheck` stay green throughout.

## 7. Risks

- darktable baseline mechanism may not be scriptable cleanly; the spike bounds this to a day and Step 2 can be deferred without blocking Steps 1 and 3.
- Delta-merged super-resolution may recover less apparent detail than replacement; validated on the sheets, with replacement kept behind a setting if needed.
- InsightFace inside enhance adds a fourth model to the VRAM rotation; step-major ordering keeps one resident at a time.
- Migration `0004` is additive; `make reset` still recreates from scratch.

## 8. Deliverables

Three PRs, in order: Step 1, Step 2 (spike report then implementation), Step 3. Each carries its tests, its contact sheet, and a CLAUDE.md update. Roughly one week of work plus review time.
