// Control-center root: header, timeline, context panel, resource bar.
import { api } from "./api.js";
import {
  createRoot, html, useCallback, useRef, useState,
  usePoll,
} from "./ui.js";
import { TimelineBar } from "./timeline.js";
import { StagePanel } from "./stage-panel.js";
import { ResourceBar } from "./resources.js";
import { ReviewPanel } from "./review.js";

const POLL_BUSY_MS = 1000;
const POLL_IDLE_MS = 3000;
const LOG_KEEP_LINES = 2000;

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
  const [panel, setPanel] = useState("pipeline"); // "pipeline" | "review"
  const [showReset, setShowReset] = useState(false);
  const [error, setError] = useState(null);
  const logRef = useRef({ seq: 0, next: 0, lines: [] });

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

  const onRunStage = useCallback((name) => {
    const stage = status?.stages?.find((s) => s.name === name);
    if (!stage) return;
    if (stage.state === "done" && !confirm(`Re-run ${stage.title}?`)) return;
    if (
      (stage.state === "failed" || stage.state === "cancelled") &&
      !confirm(
        `${stage.title} ended ${stage.state}` +
          `${stage.exit_code != null ? ` (exit ${stage.exit_code})` : ""}` +
          ` — the log is shown in the panel below. Re-run it?`
      )
    ) {
      return;
    }
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
            · ${status?.batch?.incoming_files ?? 0} files incoming
          </span>
        </div>
        <div class="flex items-center gap-2 text-sm">
          <button class="px-3 py-1.5 rounded bg-sky-800 hover:bg-sky-700 disabled:opacity-40"
                  disabled=${busy} onClick=${() => act(() => api.autoRun(1))}>${autoLabel}</button>
          <button class="px-3 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40"
                  disabled=${!busy} onClick=${() => act(() => api.cancelJob())}>⏹ Stop</button>
          <button class="px-3 py-1.5 rounded bg-zinc-900 text-rose-300 hover:bg-zinc-800 disabled:opacity-40"
                  disabled=${busy} onClick=${() => setShowReset(true)}>🗑 New batch</button>
        </div>
      </header>
      ${error && html`<div class="px-4 py-1 text-xs bg-rose-950 text-rose-200">${error}</div>`}
      <${TimelineBar} status=${status} panel=${panel}
        onRunStage=${onRunStage}
        onShowStage=${() => setPanel("pipeline")}
        onShowReview=${() => setPanel("review")} />
      <main class="flex-1 min-h-0 overflow-auto">
        ${panel === "review"
          ? html`<${ReviewPanel} pipelineBusy=${busy}
                   onSubmitAndContinue=${() => { setPanel("pipeline"); act(() => api.autoRun(2)); }} />`
          : html`<${StagePanel} status=${status} log=${log} />`}
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
