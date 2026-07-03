// StagePanel — progress + live log for the running (or last-run) stage.
import { html, useEffect, useRef, useState } from "./ui.js";

export function StagePanel({ status, log }) {
  const [follow, setFollow] = useState(true);
  const boxRef = useRef(null);
  const lines = log?.lines ?? [];
  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [lines, follow]);

  const stageName = log?.stage ?? status?.running;
  const stage = status?.stages?.find((s) => s.name === stageName);
  if (!stage) {
    return html`<div class="p-10 text-zinc-500 text-sm">
      No stage has run yet. Click a stage above, or press Auto-run to go
      ingest → filter → score → cluster and stop for your review.
    </div>`;
  }
  const p = stage.progress ?? {};
  const pctv = p.total ? Math.min(100, Math.round((p.current / p.total) * 100)) : null;
  return html`
    <div class="flex flex-col h-full p-3 gap-2">
      <div class="flex items-baseline gap-3">
        <span class="font-semibold">${stage.title}</span>
        <span class="text-sm text-zinc-400">
          ${stage.state}${stage.exit_code != null && stage.state !== "done" ? ` (exit ${stage.exit_code})` : ""}
        </span>
        ${p.total ? html`<span class="text-sm text-zinc-400">${p.current} / ${p.total}</span>` : null}
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
        <span>live log</span>
        <button class="px-1.5 rounded ${follow ? "bg-zinc-700" : "bg-zinc-900"}"
                onClick=${() => setFollow(!follow)}>auto-scroll ${follow ? "on" : "off"}</button>
      </div>
      <div ref=${boxRef}
           class="flex-1 overflow-auto bg-black/60 rounded p-2 font-mono text-[11px] leading-4 text-zinc-300">
        ${lines.map((l, i) => html`<div key=${i}>${l}</div>`)}
      </div>
    </div>`;
}
