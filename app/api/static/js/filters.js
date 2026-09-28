// Client-side review filters. The queue/cluster payloads carry everything the
// predicate needs (decision, favorite, n_faces, enhanced, exposure_flag), so
// filtering is instant and needs no server round-trip.
import { html } from "./ui.js";

export const EMPTY_FILTER = {
  decision: "all", // all | undecided | yes | no
  fav: false,
  faces: false,
  enhanced: false,
  flagged: false,
};

export function isFilterActive(f) {
  return f.decision !== "all" || f.fav || f.faces || f.enhanced || f.flagged;
}

export function matchesFilter(p, f) {
  const sel = p.decision?.selected;
  const undecided = !sel || sel === "undecided";
  if (f.decision === "undecided" && !undecided) return false;
  if (f.decision === "yes" && sel !== "yes") return false;
  if (f.decision === "no" && sel !== "no") return false;
  if (f.fav && !p.decision?.favorite) return false;
  if (f.faces && !(p.n_faces > 0)) return false;
  if (f.enhanced && !p.enhanced) return false;
  if (f.flagged && !(p.exposure_flag && p.exposure_flag !== "ok")) return false;
  return true;
}

export function FilterBar({ filter, setFilter, counts }) {
  // counts is optional; a chip shows " (n)" only when its tally is known.
  const n = (key) => (counts && counts[key] != null ? ` (${counts[key]})` : "");
  const seg = (id, label, countKey) => html`
    <button onClick=${() => setFilter({ ...filter, decision: id })}
      class="px-2.5 py-1 rounded text-xs ${filter.decision === id
        ? "bg-zinc-600 text-zinc-100" : "bg-zinc-900 text-zinc-400"}">
      ${label}${n(countKey)}
    </button>`;
  const toggle = (key, label, countKey) => html`
    <button onClick=${() => setFilter({ ...filter, [key]: !filter[key] })}
      class="px-2.5 py-1 rounded text-xs ${filter[key]
        ? "bg-sky-800 text-sky-100" : "bg-zinc-900 text-zinc-400"}">
      ${label}${n(countKey)}
    </button>`;
  return html`
    <div class="flex items-center gap-2 flex-wrap px-4 py-1.5 border-b border-zinc-800 bg-zinc-950/90">
      <span class="text-[11px] uppercase tracking-wider text-zinc-600 mr-1">filter</span>
      <div class="flex gap-1 p-0.5 bg-zinc-900 border border-zinc-800 rounded">
        ${seg("all", "All", "all")} ${seg("undecided", "Undecided", "undecided")}
        ${seg("yes", "Yes", "yes")} ${seg("no", "No", "no")}
      </div>
      ${toggle("fav", "★ Fav", "fav")}
      ${toggle("faces", "Faces", "faces")}
      ${toggle("enhanced", "Enhanced", "enhanced")}
      ${toggle("flagged", "Exposure flag", "flagged")}
      ${isFilterActive(filter) && html`
        <button onClick=${() => setFilter({ ...EMPTY_FILTER })}
          class="ml-auto text-xs text-zinc-500 hover:text-zinc-300">clear</button>`}
    </div>`;
}
