// TimelineBar — one chip per pipeline stage plus the Review pseudo-stage.
import { html } from "./ui.js";

const CHIP_STYLES = {
  pending: "bg-zinc-900 text-zinc-500",
  running: "bg-sky-950 text-sky-200 ring-2 ring-sky-500",
  done: "bg-emerald-950 text-emerald-200",
  failed: "bg-rose-950 text-rose-200 ring-2 ring-rose-600",
  cancelled: "bg-amber-950 text-amber-200",
  review: "bg-amber-950 text-amber-200",
};

function pct(progress) {
  if (!progress || !progress.total) return null;
  return Math.min(100, Math.round((progress.current / progress.total) * 100));
}

function elapsed(startIso, endIso) {
  if (!startIso) return null;
  const ms = (endIso ? new Date(endIso) : new Date()) - new Date(startIso);
  const s = Math.max(0, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

function chipSub(s) {
  const p = pct(s.progress);
  if (s.state === "running") return p == null ? elapsed(s.started_at) : `${p}% · ${elapsed(s.started_at)}`;
  if (s.state === "done") return elapsed(s.started_at, s.finished_at) ?? "done";
  if (s.state === "failed") return `exit ${s.exit_code ?? "?"}`;
  if (s.state === "cancelled") return "cancelled";
  return null;
}

function Chip({ label, sub, state, active, onClick, disabled }) {
  const style = CHIP_STYLES[state] ?? CHIP_STYLES.pending;
  const ring = active ? "outline outline-2 outline-zinc-400" : "";
  return html`
    <button
      class="flex-1 min-w-0 rounded-lg px-2 py-1.5 text-center text-xs ${style} ${ring}
             disabled:opacity-40 disabled:cursor-not-allowed"
      onClick=${onClick} disabled=${disabled} title=${label}>
      <div class="font-semibold truncate">${label}</div>
      <div class="truncate text-[10px] opacity-80">${sub ?? " "}</div>
    </button>`;
}

export function TimelineBar({ status, panel, onRunStage, onShowStage, onShowReview }) {
  const stages = status?.stages ?? [];
  const busy = Boolean(status?.running) || status?.autorun_leg != null;
  const batch = status?.batch ?? {};
  const chips = [];
  for (const s of stages) {
    chips.push(html`
      <${Chip} key=${s.name} label=${s.title} sub=${chipSub(s)} state=${s.state}
        active=${panel === "pipeline" && (status?.running === s.name)}
        disabled=${busy && s.state !== "running"}
        onClick=${() => (s.state === "running" ? onShowStage(s.name) : onRunStage(s.name))} />`);
    if (s.name === "cluster") {
      chips.push(html`
        <${Chip} key="review" label="Review" state="review"
          sub="${batch.decided ?? 0}/${batch.photos ?? 0} decided"
          active=${panel === "review"} disabled=${false}
          onClick=${onShowReview} />`);
    }
  }
  return html`<div class="flex gap-1.5 px-3 py-2">${chips}</div>`;
}
