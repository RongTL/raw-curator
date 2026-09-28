// StagePanel — progress + live log for the selected (or running/last-run)
// stage, plus the Run button that actually starts it.
import { html, useEffect, useRef, useState } from "./ui.js";
import { api } from "./api.js";

export function StagePanel({ status, log, selectedStage, onRunStage, busy }) {
  const [follow, setFollow] = useState(true);
  const [saved, setSaved] = useState({ stage: null, lines: [] });
  const boxRef = useRef(null);
  const lines = log?.lines ?? [];

  const stages = status?.stages ?? [];
  // Which stage to show: the one the user selected, else the running one, else
  // the most recently started. The last-started fallback is why a reload lands
  // on the stage that actually ran instead of "no stage has run yet" while the
  // timeline shows several as done.
  const lastStarted = [...stages]
    .filter((s) => s.started_at)
    .sort((a, b) => new Date(a.started_at) - new Date(b.started_at))
    .pop();
  const stageName = selectedStage ?? log?.stage ?? status?.running ?? lastStarted?.name;
  const stage = stages.find((s) => s.name === stageName);

  // The live tail (/logs) only ever holds the stage whose run is current — it's
  // cleared when the next stage starts. For any other stage, fetch its saved log
  // file so a finished (or earlier) stage still shows its output when selected.
  const isLive = Boolean(stage) && log?.stage === stage.name;
  useEffect(() => {
    if (!stage || isLive) return undefined;
    let cancelled = false;
    api
      .stageLog(stage.name)
      .then((r) => { if (!cancelled) setSaved({ stage: stage.name, lines: r.lines }); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [stage?.name, isLive, stage?.state]);
  const displayLines = isLive ? lines : saved.stage === stage?.name ? saved.lines : [];

  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [displayLines, follow]);

  if (!stage) {
    return html`<div class="p-10 text-zinc-500 text-sm">
      No stage has run yet. Click a stage above to view it, then press Run — or
      press Auto-run to go ingest → filter → score → cluster and stop for your review.
    </div>`;
  }

  const isRunning = status?.running === stage.name;
  const p = stage.progress ?? {};
  const pctv = p.total ? Math.min(100, Math.round((p.current / p.total) * 100)) : null;
  const runLabel = isRunning
    ? "running…"
    : stage.state === "done"
      ? "▶ Re-run"
      : stage.state === "failed" || stage.state === "cancelled"
        ? "▶ Run again"
        : "▶ Run";
  return html`
    <div class="flex flex-col h-full p-3 gap-2">
      <div class="flex items-baseline gap-3">
        <span class="font-semibold">${stage.title}</span>
        <span class="text-sm text-zinc-400">
          ${stage.state}${stage.exit_code != null && stage.state !== "done" ? ` (exit ${stage.exit_code})` : ""}
        </span>
        ${p.total ? html`<span class="text-sm text-zinc-400">${p.current} / ${p.total}</span>` : null}
        <button
          class="ml-auto px-3 py-1 rounded text-sm bg-emerald-700 hover:bg-emerald-600
                 disabled:bg-zinc-800 disabled:text-zinc-500 disabled:cursor-not-allowed"
          disabled=${busy || isRunning} onClick=${() => onRunStage?.(stage.name)}
          title=${isRunning ? "already running" : `run the ${stage.title} stage`}>
          ${runLabel}
        </button>
      </div>
      ${pctv != null && html`
        <div class="h-2 bg-zinc-800 rounded">
          <div class="h-2 rounded ${stage.state === "failed" ? "bg-rose-600" : "bg-sky-500"}"
               style=${{ width: `${pctv}%` }}></div>
        </div>`}
      ${stage.state === "failed" && html`
        <div class="text-xs text-rose-300 bg-rose-950/60 rounded px-2 py-1">
          Stage failed — last log lines below; full log:
          <span class="font-mono">${log?.log_path ?? "cache/logs/"}</span>
        </div>`}
      <div class="flex items-center gap-2 text-xs text-zinc-500">
        <span>${isLive ? "live log" : "saved log"}</span>
        <button class="px-1.5 rounded ${follow ? "bg-zinc-700" : "bg-zinc-900"}"
                onClick=${() => setFollow(!follow)}>auto-scroll ${follow ? "on" : "off"}</button>
      </div>
      <div ref=${boxRef}
           class="flex-1 overflow-auto bg-black/60 rounded p-2 font-mono text-[11px] leading-4 text-zinc-300">
        ${displayLines.length === 0
          ? html`<div class="text-zinc-600">${isRunning
              ? "waiting for log output…"
              : "No log for this stage yet — run it here (Run / Auto-run) to capture its output."}</div>`
          : displayLines.map((l, i) => html`<div key=${i}>${l}</div>`)}
      </div>
    </div>`;
}
