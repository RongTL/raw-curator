// Side-by-side compare for a cluster/burst: see the near-duplicates together
// and pick the keeper. Decisions here stage immediately (same API as the grid).
import { html, useKeyboardShortcuts, useMemo } from "./ui.js";

function CompareCard({ p, onDecide, onKeepOnly, busy }) {
  const sel = p.decision?.selected;
  const btn = (val, label, on) => html`
    <button disabled=${busy} onClick=${() => onDecide(p.hash, val)}
      class="flex-1 px-2 py-1 rounded text-xs ${sel === val ? on : "bg-zinc-800 text-zinc-300"}">
      ${label}
    </button>`;
  return html`
    <div class="flex flex-col bg-zinc-900 rounded overflow-hidden w-72 shrink-0">
      <div class="relative bg-black">
        <img src=${p.preview_url ?? p.thumb_url} alt=${p.hash}
             class="block w-full aspect-[3/2] object-contain" />
        ${p.is_recommended && html`
          <span class="absolute top-1 left-1 bg-amber-500/90 text-black text-[10px] px-1 rounded font-semibold">REC</span>`}
      </div>
      <div class="p-2 space-y-1 text-xs">
        <div class="font-mono truncate" title=${p.filename}>${p.filename ?? p.hash.slice(0, 8)}</div>
        <div class="text-zinc-400">
          tech ${p.technical_score?.toFixed(2) ?? "—"} · aes ${p.aesthetic_score?.toFixed(1) ?? "—"}
        </div>
        <div class="flex gap-1">
          ${btn("yes", "yes", "bg-emerald-700 text-white")}
          ${btn("no", "no", "bg-rose-800 text-white")}
        </div>
        <button disabled=${busy} onClick=${() => onKeepOnly(p.hash)}
          class="w-full px-2 py-1 rounded text-xs bg-amber-700 hover:bg-amber-600 text-black disabled:opacity-40">
          keep only this
        </button>
      </div>
    </div>`;
}

export function CompareModal({ cluster, onClose, onDecide, onKeepOnly, busy }) {
  useKeyboardShortcuts(useMemo(() => ({ Escape: onClose }), [onClose]));
  if (!cluster) return null;
  const title = cluster.kind === "unclustered" ? "Unclustered" : `Cluster #${cluster.id}`;
  return html`
    <div class="fixed inset-0 z-50 bg-black flex flex-col" role="dialog" aria-modal="true">
      <div class="flex items-center justify-between px-4 py-2 border-b border-zinc-800 text-sm">
        <div class="flex items-center gap-3 min-w-0">
          <span class="font-semibold">Compare · ${title}</span>
          <span class="text-zinc-500">${cluster.photos.length} photos — pick the keeper</span>
        </div>
        <button onClick=${onClose} class="px-3 py-1 bg-zinc-800 rounded">close (esc)</button>
      </div>
      <div class="flex-1 overflow-auto p-3">
        <div class="flex gap-3 flex-wrap">
          ${cluster.photos.map((p) => html`
            <${CompareCard} key=${p.hash} p=${p}
              onDecide=${onDecide} onKeepOnly=${onKeepOnly} busy=${busy} />`)}
        </div>
      </div>
    </div>`;
}
