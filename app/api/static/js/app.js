// Control-center root: header, timeline, context panel, resource bar.
import { api } from "./api.js";
import {
  createRoot, html, useCallback, useEffect, useRef, useState,
  usePoll,
} from "./ui.js";
import { TimelineBar } from "./timeline.js";
import { StagePanel } from "./stage-panel.js";
import { ResourceBar } from "./resources.js";
import { ReviewPanel } from "./review.js";

const POLL_BUSY_MS = 1000;
const POLL_IDLE_MS = 3000;
const LOG_KEEP_LINES = 2000;

// Stages that move or delete files on disk. A tile click only ever opens the
// stage view; starting one of these needs the explicit Run button AND this
// spelled-out confirmation, so a stray click can't delete a RAW.
const DESTRUCTIVE_STAGES = {
  submit:
    'Submit moves files on disk. Photos marked "no" have their source RAW ' +
    "deleted after enhancement. This cannot be undone.",
  enhance:
    'Enhance develops every decided RAW; for "no" photos it deletes the source ' +
    "RAW once the TIFF is written. This cannot be undone.",
  "export-jpeg":
    "Export JPEG develops every kept RAW and enhanced TIFF into share JPEGs " +
    "(writes many files).",
};

function ResetModal({ onClose, onDone }) {
  const [text, setText] = useState("");
  const [err, setErr] = useState(null);
  const [pending, setPending] = useState(false);
  const go = async () => {
    setPending(true);
    try { await api.resetSession(); onDone(); }
    catch (e) { setErr(e.message); }
    finally { setPending(false); }
  };
  return html`
    <div class="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div class="bg-zinc-900 rounded-lg p-5 w-96 space-y-3 text-sm">
        <div class="font-semibold text-rose-300">Start a new batch?</div>
        <p class="text-zinc-400">This wipes the session DB, previews, and the
          library/exported working folders (incoming photos stay). Type
          <span class="kbd">RESET</span> to confirm.</p>
        <input class="w-full bg-zinc-800 rounded px-2 py-1" value=${text}
               onInput=${(e) => setText(e.target.value)} placeholder="RESET" />
        ${err && html`<div class="text-rose-400 text-xs">${err}</div>`}
        <div class="flex justify-end gap-2">
          <button class="px-3 py-1 rounded bg-zinc-800" onClick=${onClose}>cancel</button>
          <button class="px-3 py-1 rounded bg-rose-800 disabled:opacity-40"
                  disabled=${text !== "RESET" || pending} onClick=${go}>${pending ? "wiping…" : "wipe session"}</button>
        </div>
      </div>
    </div>`;
}

function App() {
  const [status, setStatus] = useState(null);
  const [stats, setStats] = useState(null);
  const [log, setLog] = useState(null);
  const [panel, setPanel] = useState(() => {
    const m = /panel=(pipeline|review)/.exec(location.hash);
    return m ? m[1] : "pipeline"; // "pipeline" | "review"
  });
  const [selectedStage, setSelectedStage] = useState(null); // stage tile the user is viewing
  const [showReset, setShowReset] = useState(false);
  const [error, setError] = useState(null);
  const logRef = useRef({ seq: 0, next: 0, lines: [] });

  // Remember pipeline vs review in the URL so a reload doesn't always land on
  // the Ingest log; ReviewPanel persists its own sub-view under the same hash.
  useEffect(() => {
    const params = new URLSearchParams(location.hash.slice(1));
    params.set("panel", panel);
    history.replaceState(null, "", `#${params.toString()}`);
  }, [panel]);

  const busy = Boolean(status?.running) || status?.autorun_leg != null;

  const inflight = useRef(false);
  const refresh = useCallback(async () => {
    if (inflight.current) return;
    inflight.current = true;
    try {
      const st = await api.pipelineStatus();
      setStatus(st);
      if (st.run_seq > 0) {
        const sameRun = st.run_seq === logRef.current.seq;
        let chunk = await api.logs(sameRun ? logRef.current.next : 0);
        if (sameRun && chunk.run_seq !== logRef.current.seq) {
          chunk = await api.logs(0);
        }
        const prev = chunk.run_seq === logRef.current.seq ? logRef.current.lines : [];
        logRef.current = {
          seq: chunk.run_seq,
          next: chunk.next,
          lines: [...prev, ...chunk.lines].slice(-LOG_KEEP_LINES),
        };
        setLog({ ...chunk, lines: logRef.current.lines });
      }
    } finally {
      inflight.current = false;
    }
  }, []);
  usePoll(refresh, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
  usePoll(useCallback(async () => setStats(await api.systemStats()), []),
          busy ? POLL_BUSY_MS : POLL_IDLE_MS);

  const act = useCallback(async (fn) => {
    setError(null);
    try { await fn(); await refresh(); }
    catch (e) { setError(e.message); }
  }, [refresh]);

  // Selecting a stage only opens its view — it never starts work.
  const onSelectStage = useCallback((name) => {
    setSelectedStage(name);
    setPanel("pipeline");
  }, []);

  // Starting a stage is the deliberate action, gated by a confirmation whose
  // wording depends on the stage: destructive stages spell out the disk
  // consequences and the undecided count; re-runs just double-check.
  const onRunStage = useCallback((name) => {
    const stage = status?.stages?.find((s) => s.name === name);
    if (!stage) return;
    const warn = DESTRUCTIVE_STAGES[name];
    if (warn) {
      const photos = status?.batch?.photos ?? 0;
      const decided = status?.batch?.decided ?? 0;
      const undecided = Math.max(0, photos - decided);
      const tail = undecided > 0
        ? `\n\n${undecided} of ${photos} photo(s) are still undecided — they will ` +
          "be left untouched (not moved, not enhanced)."
        : "";
      if (!confirm(`Run ${stage.title}?\n\n${warn}${tail}`)) return;
    } else if (stage.state === "done") {
      if (!confirm(`Re-run ${stage.title}?`)) return;
    } else if (stage.state === "failed" || stage.state === "cancelled") {
      if (!confirm(
        `${stage.title} ended ${stage.state}` +
          `${stage.exit_code != null ? ` (exit ${stage.exit_code})` : ""}` +
          " — the log is shown in the panel below. Re-run it?"
      )) return;
    }
    setSelectedStage(name);
    setPanel("pipeline");
    act(() => api.runStage(name));
  }, [status, act]);

  const autoLabel = status?.autorun_leg != null
    ? `auto-run leg ${status.autorun_leg}…`
    : "▶ Auto-run (ingest → cluster)";

  return html`
    <div class="h-full flex flex-col">
      <header class="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <div class="flex items-baseline gap-3">
          <h1 class="text-lg font-semibold">raw-curator</h1>
          <span class="text-sm text-zinc-400">
            ${status?.batch?.photos ?? 0} photos · ${status?.batch?.decided ?? 0} decided
            · <span title="image files currently in photos/incoming/ — they stay there after ingest">
              ${status?.batch?.incoming_files ?? 0} in incoming</span>
          </span>
        </div>
        <div class="flex items-center gap-2 text-sm">
          <button class="px-3 py-1.5 rounded bg-sky-800 hover:bg-sky-700 disabled:opacity-40"
                  disabled=${busy} onClick=${() => { setSelectedStage(null); act(() => api.autoRun(1)); }}>${autoLabel}</button>
          <button class="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40"
                  disabled=${!busy} onClick=${() => act(() => api.cancelJob())}>⏹ Stop</button>
          <button class="px-3 py-1.5 rounded bg-zinc-900 text-rose-300 hover:bg-zinc-800 disabled:opacity-40"
                  disabled=${busy} onClick=${() => setShowReset(true)}>🗑 New batch</button>
        </div>
      </header>
      ${error && html`<div class="px-4 py-1 text-xs bg-rose-950 text-rose-200">${error}</div>`}
      <${TimelineBar} status=${status} panel=${panel} selectedStage=${selectedStage}
        onSelectStage=${onSelectStage}
        onShowReview=${() => setPanel("review")} />
      <main class="flex-1 min-h-0 overflow-auto">
        ${panel === "review"
          ? html`<${ReviewPanel} pipelineBusy=${busy}
                   onSubmitAndContinue=${() => { setSelectedStage(null); setPanel("pipeline"); act(() => api.autoRun(2)); }} />`
          : html`<${StagePanel} status=${status} log=${log} selectedStage=${selectedStage}
                   onRunStage=${onRunStage} busy=${busy} />`}
      </main>
      <${ResourceBar} stats=${stats} />
      ${showReset && html`<${ResetModal} onClose=${() => setShowReset(false)}
          onDone=${() => {
            setShowReset(false);
            logRef.current = { seq: 0, next: 0, lines: [] };
            setLog(null);
            refresh().catch(() => {});
          }} />`}
    </div>`;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
