# raw-curator Control Center — Design Spec

**Date:** 2026-07-03
**Branch:** `feat/control-center`
**Status:** approved by user (layout, review-stage treatment, auto-run flow, and job-runner model all confirmed via brainstorming session)

## Goal

Replace the per-stage `make` workflow with a single web interface that:

1. runs every pipeline stage (reset → ingest → filter → score → cluster → review → submit → enhance → export-jpeg) from one screen,
2. shows clear step/progress information with live logs,
3. monitors system resources (CPU, RAM, GPU, VRAM, disk) in real time,
4. and, alongside that, brings the codebase to best quality and ensures the pipeline saturates the target hardware (AMD R3 3100 — 4C/8T, 24 GB RAM, NVIDIA RTX 2060 — 6 GB VRAM).

`make image` and `make download-models` remain make targets: they are host-level
operations (podman build, first-time model fetch) that cannot run from inside the
container. Everything else moves into the UI. The user-facing flow becomes:
`make serve` → open http://localhost:8080 → do everything there.

## Non-goals

- No multi-user support, auth, or remote access hardening (single user on localhost).
- No changes to pipeline algorithms, decision rules, or the binary yes/no model.
- No Node/npm build step; the frontend stays CDN-React + native ES modules.
- The dead-by-design DB columns (`Decision.score_tier`, `Decision.enhance_requested`) stay untouched.

## Chosen design (user selections)

| Decision | Choice |
|---|---|
| Layout | **C — Pipeline Timeline + Context Panel**: horizontal stage timeline on top, large context area below, slim expandable resource bar docked at the bottom |
| Review stage | **Redesigned to match** the new UI (same features, consistent styling) |
| Auto-run | **Two legs**: leg 1 = ingest → filter → score → cluster, then wait for human review; after "Submit & continue", leg 2 = submit → enhance → export-jpeg |
| Job execution | **Subprocess per stage** inside the long-running `ui` container |

Reference mockups: `.superpowers/brainstorm/11363-1783041835/content/control-center.html` (not committed).

## Architecture

### 1. Orchestrator (`app/orchestrator/`)

New package owning job execution state. Small focused modules:

- `runner.py` — spawns `raw-curator <stage>` as an `asyncio` subprocess with
  stdout/stderr merged, writes the stream to `cache/logs/<stage>-<run_id>.log`,
  keeps a bounded in-memory tail. Exposes `start(stage)`, `cancel()`, `state()`.
  A single global job slot (asyncio lock): starting a job while one runs returns
  HTTP 409. Cancel sends SIGTERM, escalates to SIGKILL after a grace period.
- `stages.py` — the stage registry: ordered list of stage descriptors
  (`name`, `cli_args`, `leg`, `progress_query`). Single source of truth used by
  the API and auto-run.
- `autorun.py` — chains stages of a leg; on stage failure the chain stops and
  the failure is surfaced. Leg 1 = ingest, filter, score, cluster. Leg 2 =
  submit, enhance, export-jpeg.
- `progress.py` — derives per-stage progress from DB counts (e.g. photos with
  embeddings / total photos for score; enhanced TIFFs on disk vs decided rows
  for enhance). Read-only queries against the session DB; no changes to job code.
- `reset.py` — wraps the existing in-container session reset (DB + cache +
  working dirs, then `alembic upgrade head`) so "New batch" works from the UI.

State model per stage: `pending | running | done | failed | cancelled`, plus
`started_at`, `finished_at`, `exit_code`, `progress {current, total}`.
Orchestrator state is in-memory (ephemeral single-batch tool); on server restart,
completed-stage status is re-derived from DB counts by `progress.py`.

### 2. System monitor (`app/monitor/`)

- `stats.py` — samples via `psutil`: CPU total + per-core %, RAM used/total,
  disk used/total for the `photos`, `cache`, `models` mounts. GPU via
  `nvidia-ml-py` (NVML): utilization %, VRAM used/total, temperature. NVML
  failures (no GPU, driver mismatch) degrade gracefully to `gpu: null` — never
  break the endpoint.
- A background sampler task keeps a 60-sample rolling window (1 s interval)
  so the UI can render charts without client-side history.
- Disk guard: `GET /api/system/stats` includes `warnings`, e.g. low free space
  on the photos mount before Enhance (enhance TIFFs are the biggest writers).

New runtime dependencies: `psutil`, `nvidia-ml-py`.

### 3. API additions (`app/api/routes/`)

- `pipeline.py`
  - `GET  /api/pipeline/status` — all stages with state/progress/timing, current job, auto-run state, batch summary (photo count, incoming size).
  - `POST /api/pipeline/run/{stage}` — start one stage (409 if busy; 422 unknown stage).
  - `POST /api/pipeline/auto/{leg}` — start leg 1 or 2.
  - `POST /api/pipeline/cancel` — cancel the running job/auto-run.
  - `GET  /api/pipeline/logs/{stage}?after=N` — incremental log lines from offset N.
  - `POST /api/pipeline/reset` — "New batch"; requires `{"confirm": "RESET"}` body.
- `system.py`
  - `GET /api/system/stats` — current sample + 60 s rolling window + warnings.

Existing routes (`queue`, `photo`, `cluster`, `decide`, `submit`) are unchanged;
the redesigned review UI keeps consuming them. All new endpoints use pydantic
request/response models and the project's response conventions.

### 4. Frontend (`app/api/static/`)

Rewrite as small ES modules (no build step, CDN React as today):

```
static/
  index.html          # shell: CDN imports, root mount
  css/app.css         # dark theme, design tokens
  js/api.js           # fetch client, polling helpers
  js/app.js           # root component, state, routing between panels
  js/timeline.js      # TimelineBar — stage chips: status, %, elapsed/ETA; click to run
  js/stage-panel.js   # StageProgress + LogViewer (auto-scroll, pause)
  js/review/grid.js   # ReviewGrid — clusters + all-photos views, infinite scroll
  js/review/photo.js  # preview panel: large preview, scores, yes/no controls
  js/resources.js     # ResourceBar — docked sparklines; expandable chart panel
```

Behavior:

- Header: batch summary, **Auto-run** (leg-aware label), **Stop**, **New batch**
  (type-to-confirm modal).
- Timeline: 8 chips (ingest, filter, score, cluster, review, submit, enhance,
  jpeg). Colors: pending grey, running blue + %, done green + duration, failed
  red, review amber "waits for you". Clicking a runnable chip starts that stage;
  Review chip switches the context panel to the grid.
- Context panel: while a job runs — progress bar, sub-stage line, live log tail;
  at Review — the redesigned grid with all current features (clusters view,
  all-photos infinite scroll, yes/no staging, bulk "don't keep any RAW",
  keyboard shortcuts, decided counter) plus **Submit & continue** which triggers
  auto-run leg 2.
- Resource bar: always visible; text + unicode sparklines for CPU/RAM/GPU/VRAM/
  disk; expands to 60 s charts with per-core CPU and GPU temperature.
- Updates: `status` + `stats` polled at 1 s while a job runs, 3 s when idle;
  logs fetched incrementally with `after` offsets. No WebSocket/SSE (KISS,
  single user).
- Failures: failed chip turns red; context panel pins the last ~30 log lines
  with a "copy log path" affordance.

### 5. Resource-utilization pass

Audit and tune for R3 3100 / 24 GB / RTX 2060 6 GB:

- Verify every CPU-bound phase (ingest decode/preview, filters, export-jpeg)
  actually uses `settings.cpu_workers` (= all 8 threads by default) and that no
  hidden serialization (e.g. per-item exiftool spawn) caps throughput.
- Set explicit `torch.set_num_threads` / OMP env defaults so GPU stages' CPU
  preprocessing doesn't oversubscribe or starve.
- Confirm VRAM ceilings: stage-by-stage model loading with `empty_cache`
  between stages stays; document measured headroom per stage.
- Ship a commented `.env.example` with tuned defaults for this exact hardware
  (`RAWCURATOR_CLIP_BATCH`, `RAWCURATOR_ENHANCE_AI_SCALE`, worker counts, …).
- The live monitor is the observability half of this requirement: saturation
  becomes visible instead of guessed.

### 6. Code-quality pass

- `ruff check`, `mypy --strict`, and the full pytest suite run clean.
- New code follows repo rules: files < 800 lines (target 200–400), functions
  < 50 lines, early returns, named constants, explicit error handling
  (subprocess failures, NVML absence, disk-full, malformed requests),
  validation at all new API boundaries.
- The 534-line `index.html` monolith is retired by the modular frontend.

## Error handling summary

| Failure | Behavior |
|---|---|
| Stage subprocess exits non-zero | stage → `failed`, auto-run halts, log tail surfaced in UI |
| Start while a job is running | HTTP 409 with the running stage's name |
| Cancel | SIGTERM, SIGKILL after grace; stage → `cancelled` |
| NVML unavailable | `gpu: null`, UI shows "GPU stats unavailable"; endpoint stays 200 |
| Low disk before enhance | warning in stats payload, amber banner in UI |
| Reset without confirm token | HTTP 422; nothing wiped |
| Server restart mid-batch | stage states re-derived from DB counts; a previously running job is reported as failed (`unknown exit`) |

## Testing

- **Unit:** runner state machine (start/finish/fail/cancel, single-slot lock),
  autorun chaining + halt-on-failure, progress derivation from a seeded DB,
  stats sampler with psutil/NVML mocked, reset confirm-token validation.
- **Integration:** new routes via FastAPI TestClient with the runner mocked
  (run/409/cancel/logs-offset/reset), stats endpoint shape.
- **Existing suite:** must keep passing untouched.
- GPU-marked tests stay opt-in via `RUN_GPU_TESTS=1` as today.

## Commit sequence on `feat/control-center`

1. `fix: pin onnxruntime-gpu to CUDA 12 line (<1.23)` — already-staged fix from this session.
2. `chore: quality pass — ruff/mypy/test fixes` (whatever the audit finds).
3. `perf: resource tuning for 8-thread CPU + 6GB VRAM; add .env.example`.
4. `feat(api): orchestrator, pipeline routes, system monitor` (+ tests).
5. `feat(ui): control-center frontend (timeline, review grid, resource bar)`.
6. `docs: README/USER_GUIDE/CLAUDE.md for the one-interface flow`.

## Acceptance criteria

- `make serve` + browser is sufficient to run a complete batch end-to-end
  (reset → … → export-jpeg) without any other make command.
- Auto-run leg 1 stops at Review; "Submit & continue" completes leg 2 unattended.
- Resource bar shows live CPU/RAM/GPU/VRAM/disk while stages run.
- A killed/OOM'd stage leaves the server alive and the failure visible.
- `make lint`, `make typecheck`, `make test` all pass.
