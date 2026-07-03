// ResourceBar — docked system stats with unicode sparklines; expandable charts.
import { html, useState } from "./ui.js";

const BARS = "▁▂▃▄▅▆▇█";

export function spark(values, max) {
  if (!values.length) return "";
  const m = max || Math.max(...values, 1);
  return values
    .map((v) => BARS[Math.max(0, Math.min(7, Math.floor((v / m) * 7.99)))])
    .join("");
}

const gb = (mb) => (mb / 1024).toFixed(1);
const seriesFrom = (win, pick) =>
  (win ?? []).map(pick).filter((v) => v != null).slice(-30);

export function ResourceBar({ stats }) {
  const [expanded, setExpanded] = useState(false);
  const cur = stats?.current;
  if (!cur) {
    return html`<div class="px-3 py-1.5 text-xs text-zinc-600 border-t border-zinc-800">system stats…</div>`;
  }
  const win = stats.window ?? [];
  const cpuS = seriesFrom(win, (s) => s.cpu_pct);
  const ramS = seriesFrom(win, (s) => s.ram_used_mb);
  const gpuS = seriesFrom(win, (s) => s.gpu?.util_pct);
  const g = cur.gpu;
  const photosDisk = cur.disks?.photos;
  return html`
    <div class="border-t border-zinc-800 text-xs">
      ${(stats.warnings ?? []).map((w) => html`
        <div key=${w} class="px-3 py-1 bg-amber-950/70 text-amber-200">⚠ ${w}</div>`)}
      <div class="flex items-center gap-4 px-3 py-1.5 text-zinc-300 flex-wrap">
        <span>CPU ${Math.round(cur.cpu_pct)}% <span class="spark text-sky-400">${spark(cpuS, 100)}</span></span>
        <span>RAM ${gb(cur.ram_used_mb)}/${gb(cur.ram_total_mb)} GB
          <span class="spark text-emerald-400">${spark(ramS, cur.ram_total_mb)}</span></span>
        ${g ? html`
          <span>GPU ${Math.round(g.util_pct)}% <span class="spark text-amber-400">${spark(gpuS, 100)}</span></span>
          <span>VRAM ${gb(g.vram_used_mb)}/${gb(g.vram_total_mb)} GB</span>`
          : html`<span class="text-zinc-600">GPU stats unavailable</span>`}
        ${photosDisk && html`<span>Disk ${photosDisk.used_gb.toFixed(0)}/${photosDisk.total_gb.toFixed(0)} GB
          (${photosDisk.free_gb.toFixed(0)} free)</span>`}
        <button class="ml-auto text-zinc-500 hover:text-zinc-300" onClick=${() => setExpanded(!expanded)}>
          ${expanded ? "▾ collapse" : "▴ expand"}
        </button>
      </div>
      ${expanded && html`
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 px-3 pb-2 text-zinc-400">
          <div>
            <div class="text-zinc-500">CPU per core</div>
            <div class="font-mono">${(cur.cpu_per_core ?? []).map((c) => Math.round(c)).join(" ")}</div>
            <div class="spark text-sky-400 text-base">${spark(cpuS, 100)}</div>
          </div>
          <div>
            <div class="text-zinc-500">RAM</div>
            <div>${gb(cur.ram_used_mb)} / ${gb(cur.ram_total_mb)} GB</div>
            <div class="spark text-emerald-400 text-base">${spark(ramS, cur.ram_total_mb)}</div>
          </div>
          <div>
            <div class="text-zinc-500">GPU</div>
            ${g ? html`
              <div>${Math.round(g.util_pct)}% · ${g.temp_c}°C · VRAM ${gb(g.vram_used_mb)}/${gb(g.vram_total_mb)} GB</div>
              <div class="spark text-amber-400 text-base">${spark(gpuS, 100)}</div>`
              : html`<div>unavailable</div>`}
          </div>
          <div>
            <div class="text-zinc-500">Disks</div>
            ${Object.entries(cur.disks ?? {}).map(([name, d]) => html`
              <div key=${name}>${name}: ${d
                ? `${d.used_gb.toFixed(0)}/${d.total_gb.toFixed(0)} GB (${d.free_gb.toFixed(0)} free)`
                : "n/a"}</div>`)}
          </div>
        </div>`}
    </div>`;
}
