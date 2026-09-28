// Review UI — ported from the pre-control-center index.html inline app.
// Components are unchanged except: imports, and ReviewPanel replacing App/Header.
import { api } from "./api.js";
import {
  Fragment, html, useCallback, useEffect, useMemo, useRef, useState,
  useKeyboardShortcuts, useQuery,
} from "./ui.js";
import { EMPTY_FILTER, FilterBar, matchesFilter } from "./filters.js";
import { CompareModal } from "./compare.js";

const PAGE_SIZE = 200;

// Human explanation for each exposure flag the filter stage can set (see
// app/filters/exposure.py), shown as a tooltip so the badge isn't a mystery.
const EXPOSURE_HELP = {
  overexposed: "Overexposed — >5% of pixels are blown out (near pure white); highlight detail is lost.",
  underexposed: "Underexposed — >5% of pixels are crushed (near pure black); shadow detail is lost.",
  high_contrast_clipped: "High contrast — both highlights and shadows are clipping at once.",
  very_dark: "Very dark — the overall brightness is very low (mean luma < 32/255).",
  very_bright: "Very bright — the overall brightness is very high (mean luma > 224/255).",
};
function exposureHelp(flag) {
  return EXPOSURE_HELP[flag] ?? `Exposure flag: ${(flag ?? "").replace(/_/g, " ")}`;
}

function DecisionBadge({ decision }) {
  if (!decision || decision.selected === "undecided") return null;
  const s = decision.selected;
  const color = s === "yes" ? "bg-emerald-700" : s === "no" ? "bg-rose-800" : "bg-zinc-700";
  return html`<span class="${color} text-xs px-1.5 py-0.5 rounded">${s}</span>`;
}

// A rateable grid tile. It's a div (not a button) so the star buttons can nest
// legally; Enter/Space still open it, and the stars stopPropagation so a rating
// click never opens the modal.
function PhotoTile({ photo, onOpen, onStar, highlight = false }) {
  const score = photo.technical_score == null ? "—" : photo.technical_score.toFixed(2);
  const aest = photo.aesthetic_score == null ? "—" : photo.aesthetic_score.toFixed(1);
  const ring = highlight ? "ring-2 ring-amber-400" : "";
  const stars = photo.decision?.stars || 0;
  const open = () => onOpen(photo.hash);
  const flag = photo.exposure_flag && photo.exposure_flag !== "ok" ? photo.exposure_flag : null;
  return html`
    <div
      class="photo-tile relative group bg-zinc-900 rounded overflow-hidden text-left cursor-pointer ${ring}"
      role="button" tabindex="0" aria-label=${`open ${photo.filename ?? photo.hash}`}
      onClick=${open}
      onKeyDown=${(e) => {
        // Only the tile itself opens on Enter/Space; a keypress on a nested
        // star button activates the button, not the modal.
        if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          open();
        }
      }}>
      <img src=${photo.thumb_url} alt=${photo.filename ?? photo.hash}
           class="block w-full aspect-[3/2] object-cover" />
      <div class="absolute top-1 left-1 flex gap-1">
        ${photo.is_recommended && html`<span title="recommended keeper for this cluster"
          class="bg-amber-500/90 text-black text-xs px-1.5 py-0.5 rounded font-semibold">REC</span>`}
        ${photo.decision?.applied && html`<span class="bg-blue-700 text-xs px-1.5 py-0.5 rounded">applied</span>`}
        <${DecisionBadge} decision=${photo.decision} />
        ${photo.enhanced && html`<span class="bg-emerald-600/90 text-black text-[10px] px-1 py-0.5 rounded font-semibold"
          title=${`enhanced${photo.q_after != null ? ` · Q ${Math.round(photo.q_after)}` : ""}`}>✓</span>`}
        ${photo.degraded && html`<span class="bg-amber-600/90 text-black text-[10px] px-1 py-0.5 rounded font-semibold" title="verify: degraded">!</span>`}
        ${flag && html`<span class="bg-amber-800/90 text-amber-100 text-[10px] px-1 py-0.5 rounded font-semibold"
          title=${exposureHelp(flag)}>${flag.replace(/_/g, " ")}</span>`}
      </div>
      <div class="absolute top-1 right-1 flex bg-black/60 rounded px-0.5"
           onClick=${(e) => e.stopPropagation()}>
        ${[1, 2, 3, 4, 5].map((n) => html`
          <button key=${n} type="button" title=${`${n} star${n === 1 ? "" : "s"}`}
            class="px-0.5 text-sm leading-none ${n <= stars ? "text-amber-400" : "text-zinc-500 hover:text-amber-300"}"
            onClick=${(e) => { e.stopPropagation(); onStar(photo.hash, n === stars ? 0 : n); }}>★</button>`)}
      </div>
      ${photo.file_kind && html`
        <div class="absolute bottom-7 left-1">
          <span class="bg-zinc-700/90 text-zinc-100 text-[9px] font-semibold uppercase tracking-wider px-1 py-0.5 rounded">
            ${photo.file_kind}
          </span>
        </div>`}
      <div class="absolute bottom-0 inset-x-0 px-2 py-1 bg-black/60 text-[11px] flex justify-between gap-2">
        <span class="font-mono truncate" title=${photo.filename ?? photo.hash}>
          ${photo.filename ?? photo.hash.slice(0, 8)}
        </span>
        <span class="shrink-0">tech ${score} • aes ${aest}</span>
      </div>
    </div>`;
}

function PhotoGrid({ items, onOpen, onStar }) {
  return html`
    <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2 p-2">
      ${items.map(p => html`<${PhotoTile} key=${p.hash} photo=${p} onOpen=${onOpen} onStar=${onStar} />`)}
    </div>`;
}

// Infinite-scroll grid: renders items.slice(0, visible) and grows `visible`
// via onMore whenever a bottom sentinel scrolls into view. Re-observing on
// every `visible`/length change lets it keep filling until the sentinel is
// pushed off-screen (handles short pages that don't cover the viewport).
function PagedGrid({ items, visible, onMore, onOpen, onStar }) {
  const sentinel = useRef(null);
  useEffect(() => {
    const el = sentinel.current;
    if (!el) return;
    const ob = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) onMore(); },
      { rootMargin: "600px" }
    );
    ob.observe(el);
    return () => ob.disconnect();
  }, [onMore, visible, items.length]);

  const shown = items.slice(0, visible);
  const hasMore = visible < items.length;
  return html`
    <${Fragment}>
      <${PhotoGrid} items=${shown} onOpen=${onOpen} onStar=${onStar} />
      ${hasMore && html`
        <div ref=${sentinel} class="py-6 text-center text-xs text-zinc-500">
          loading more… ${shown.length} of ${items.length}
        </div>`}
    </${Fragment}>`;
}

function ClusterSection({ cluster, onOpen, onStar, onClusterAction, onCompare, busy }) {
  const isUnclustered = cluster.kind === "unclustered";
  const headerColor = isUnclustered ? "text-zinc-500" : "text-zinc-200";
  const kindBadge = isUnclustered ? null : html`
    <span class="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
      ${cluster.kind}
    </span>`;
  const title = isUnclustered
    ? "Unclustered (singletons)"
    : `Cluster #${cluster.id}`;
  const multi = cluster.photos.length > 1;
  return html`
    <section class="border-t border-zinc-800">
      <div class="flex items-center gap-3 px-3 py-2 sticky top-0 bg-zinc-950/95 backdrop-blur z-10">
        <span class="${headerColor} text-sm font-semibold">${title}</span>
        ${kindBadge}
        <span class="text-xs text-zinc-500">${cluster.size} photo${cluster.size === 1 ? "" : "s"}</span>
        ${cluster.label && html`<span class="text-xs text-zinc-400">${cluster.label}</span>`}
        ${!isUnclustered && html`
          <div class="ml-auto flex gap-1.5">
            <button disabled=${busy} onClick=${() => onClusterAction(cluster.id, "keep_recommended")}
              class="px-2 py-0.5 rounded text-xs bg-emerald-800 hover:bg-emerald-700 disabled:opacity-40"
              title="keep the recommended frame, reject the rest">keep best</button>
            <button disabled=${busy} onClick=${() => onClusterAction(cluster.id, "reject_all")}
              class="px-2 py-0.5 rounded text-xs bg-rose-900 hover:bg-rose-800 disabled:opacity-40">reject all</button>
            ${multi && html`<button disabled=${busy} onClick=${() => onCompare(cluster)}
              class="px-2 py-0.5 rounded text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40">compare</button>`}
          </div>`}
      </div>
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-2 px-2 pb-3">
        ${cluster.photos.map(p => html`
          <${PhotoTile}
            key=${p.hash}
            photo=${p}
            highlight=${!isUnclustered && p.is_recommended}
            onOpen=${onOpen} onStar=${onStar} />`)}
      </div>
    </section>`;
}

function ClusterView({ clusters, onOpen, onStar, onClusterAction, onCompare, busy }) {
  if (!clusters || clusters.length === 0) {
    return html`<div class="p-8 text-zinc-500">No clusters yet. Run the Cluster stage above.</div>`;
  }
  return html`
    <div>
      ${clusters.map(c => html`<${ClusterSection} key=${c.id} cluster=${c} onOpen=${onOpen} onStar=${onStar}
        onClusterAction=${onClusterAction} onCompare=${onCompare} busy=${busy} />`)}
    </div>`;
}

function EngineQualityPanel({ qr }) {
  const score = (v) => {
    const cls = v >= 80 ? "text-emerald-400" : v >= 60 ? "text-amber-400" : "text-rose-400";
    return html`<span class=${cls}>${v.toFixed(0)}</span>`;
  };
  const num = (v, d = 2) => v == null ? "—" : Number(v).toFixed(d);
  return html`
    <div>
      <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">
        engine quality · Q=${score(qr.score_q)}
      </div>
      <div class="grid grid-cols-2 gap-x-3 gap-y-0.5 text-xs">
        <div>exposure ${score(qr.score_exposure)}</div>
        <div class="text-zinc-500">μ-luma ${num(qr.mean_luma, 0)}</div>
        <div>dynamic ${score(qr.score_dynamic_range)}</div>
        <div class="text-zinc-500">DR ${num(qr.dr_p95_p5, 0)}</div>
        <div>color ${score(qr.score_color)}</div>
        <div class="text-zinc-500">sat ${num(qr.avg_saturation)}</div>
        <div>sharp ${score(qr.score_sharpness)}</div>
        <div class="text-zinc-500">lapVar ${num(qr.lap_var, 0)}</div>
        <div>noise ${score(qr.score_noise)}</div>
        <div class="text-zinc-500">σ ${num(qr.luma_noise)}</div>
      </div>
      ${qr.degraded && qr.verify?.reasons?.length ? html`
        <div class="mt-1 text-xs text-rose-300">verify degraded: ${qr.verify.reasons.join(", ")}</div>` : ""}
      <details class="mt-2 text-xs text-zinc-500">
        <summary class="cursor-pointer text-zinc-400">all metrics</summary>
        <div class="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5">
          <div>shadow clip ${(qr.shadow_clip * 100).toFixed(1)}%</div>
          <div>highlight clip ${(qr.highlight_clip * 100).toFixed(1)}%</div>
          <div>midtone ratio ${(qr.midtone_ratio * 100).toFixed(1)}%</div>
          <div>midtone dev ${num(qr.midtone_deviation)}</div>
          <div>local DR ${num(qr.local_dr_mean, 0)}</div>
          <div>R/G ratio ${num(qr.rg_ratio, 3)}</div>
          <div>B/G ratio ${num(qr.bg_ratio, 3)}</div>
          <div>oversat ${(qr.oversat_ratio * 100).toFixed(1)}%</div>
          ${qr.skin_hue_var != null ? html`<div>skin huevar ${num(qr.skin_hue_var, 3)}</div>` : ""}
          <div>edge density ${(qr.edge_density * 100).toFixed(2)}%</div>
          <div>HF energy ${num(qr.hf_energy, 0)}</div>
          <div>chroma noise ${num(qr.chroma_noise)}</div>
          ${qr.measured_at ? html`<div class="col-span-2 mt-1 text-zinc-600">measured ${qr.measured_at}</div>` : ""}
        </div>
      </details>
      ${qr.plan && html`
        <details class="mt-2 text-xs text-zinc-500" open>
          <summary class="cursor-pointer text-zinc-400">
            recipe · ${qr.plan.steps.length} step${qr.plan.steps.length === 1 ? "" : "s"}
            ${qr.score_q_after != null ? html` · Q after ${qr.score_q_after.toFixed(0)}` : ""}
            ${qr.degraded ? html`<span class="ml-1 px-1 rounded bg-rose-900 text-rose-200">degraded</span>` : ""}
          </summary>
          <ol class="mt-1 list-decimal list-inside">
            ${qr.plan.steps.map((s) => html`<li key=${s.name}>${s.name}
              <span class="text-zinc-600">${Object.entries(s.params).map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(2) : v}`).join(" ")}</span></li>`)}
          </ol>
        </details>`}
    </div>`;
}

function DetailModal({ hash, onClose, onPrev, onNext, onNextUndecided, onMutated }) {
  const { data, refetch } = useQuery(() => api.photo(hash), [hash]);
  const [showAfter, setShowAfter] = useState(false);
  const [afterError, setAfterError] = useState(false);
  const [showFaces, setShowFaces] = useState(false);
  const [nat, setNat] = useState(null); // preview natural size, for face overlay
  const [noteText, setNoteText] = useState("");
  const imgRef = useRef(null);
  const wrapRef = useRef(null);
  const rootRef = useRef(null); // modal container; focused on open so keys don't leak to the grid
  const [box, setBox] = useState(null); // rendered image rect within wrapRef, for face overlays

  // Move focus into the dialog once its content renders. Otherwise the tile that
  // opened it keeps focus, and since React delegates events below window, a plain
  // Enter would re-fire that tile's onKeyDown and jump back to the first photo.
  useEffect(() => { rootRef.current?.focus(); }, [!!data]);

  // Reset the before/after toggle per photo (a new frame may have no enhanced
  // version). The faces toggle is intentionally sticky across navigation, so
  // reviewing faces across a burst doesn't need re-enabling on every frame.
  useEffect(() => { setShowAfter(false); setAfterError(false); }, [hash]);
  useEffect(() => { setNoteText(data?.decision?.note ?? ""); }, [hash, data?.decision?.note]);

  // Measure where the contained image actually renders so face boxes line up.
  const measure = useCallback(() => {
    const im = imgRef.current;
    const wr = wrapRef.current;
    if (!im || !wr) return;
    const ir = im.getBoundingClientRect();
    const wrr = wr.getBoundingClientRect();
    setBox({ left: ir.left - wrr.left, top: ir.top - wrr.top, width: ir.width, height: ir.height });
  }, []);
  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);
  useEffect(() => { measure(); }, [measure, showFaces, showAfter, data]);

  const mutate = useCallback(async (patch) => {
    await api.decide({ photo_hash: hash, ...patch });
    await refetch();
    onMutated?.();
  }, [hash, refetch, onMutated]);

  // The culling flow: a yes/no decision saves, then jumps to the next photo.
  // We advance immediately after the save lands and refresh the grid in the
  // background — no extra keypress per photo.
  const decideAndAdvance = useCallback(async (selected) => {
    await api.decide({ photo_hash: hash, selected });
    onMutated?.();
    onNext?.();
  }, [hash, onMutated, onNext]);

  useKeyboardShortcuts(useMemo(() => ({
    Escape: onClose,
    ArrowLeft: onPrev,
    ArrowRight: onNext,
    " ": onNext,
    ".": () => onNextUndecided?.(),
    "b": () => setShowAfter((v) => !v),
    "1": () => mutate({ stars: 1 }),
    "2": () => mutate({ stars: 2 }),
    "3": () => mutate({ stars: 3 }),
    "4": () => mutate({ stars: 4 }),
    "5": () => mutate({ stars: 5 }),
    "0": () => mutate({ stars: 0 }),
    "y": () => decideAndAdvance("yes"),
    "n": () => decideAndAdvance("no"),
    "u": () => mutate({ selected: "undecided" }),
    "f": () => mutate({ favorite: !data?.decision?.favorite }),
  }), [mutate, decideAndAdvance, data, onClose, onPrev, onNext, onNextUndecided]));

  if (!data) return html`
    <div class="fixed inset-0 bg-black/80 flex items-center justify-center">
      <div class="text-zinc-400">loading…</div>
    </div>`;

  const d = data.decision;
  const enhanced = data.quality_report?.score_q_after != null;
  const showingAfter = showAfter && !afterError && data.enhanced_url;
  const imgSrc = showingAfter ? data.enhanced_url : data.preview_url;
  const faces = data.faces ?? [];
  const mp = data.width && data.height ? (data.width * data.height) / 1e6 : null;
  const exifBits = [
    data.camera_body,
    data.lens,
    `ISO ${data.iso ?? "?"}`,
    `1/${data.shutter ? Math.round(1 / data.shutter) : "?"}s`,
    `f/${data.aperture ?? "?"}`,
    `${data.focal_length ?? "?"}mm`,
    mp ? `${data.width}×${data.height} · ${mp.toFixed(0)}MP` : null,
  ].filter(Boolean);

  return html`
    <div ref=${rootRef} tabindex="-1"
         class="fixed inset-0 z-50 bg-black flex flex-col outline-none" role="dialog" aria-modal="true">
      <div class="flex items-center justify-between px-4 py-2 border-b border-zinc-800 text-sm">
        <div class="flex items-center gap-3 min-w-0">
          <button onClick=${onPrev} class="px-2 py-1 bg-zinc-800 rounded" title="prev (←)">←</button>
          <button onClick=${onNext} class="px-2 py-1 bg-zinc-800 rounded" title="next (→)">→</button>
          <button onClick=${() => onNextUndecided?.()} class="px-2 py-1 bg-zinc-800 rounded text-xs"
                  title="next undecided (.)">next undecided</button>
          <span class="font-mono text-zinc-200 truncate" title=${data.source_path ?? data.filename ?? data.hash}>
            ${data.filename ?? data.hash.slice(0, 10)}
          </span>
          ${data.file_kind && html`
            <span class="bg-zinc-700 text-zinc-100 text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded">
              ${data.file_kind}
            </span>`}
          <span class="text-zinc-500 truncate">${exifBits.join(" • ")}</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex gap-0.5 p-0.5 bg-zinc-900 border border-zinc-800 rounded text-xs">
            <button onClick=${() => setShowAfter(false)}
              class="px-2 py-0.5 rounded ${!showingAfter ? "bg-zinc-600 text-zinc-100" : "text-zinc-400"}">before</button>
            <button onClick=${() => setShowAfter(true)}
              class="px-2 py-0.5 rounded ${showingAfter ? "bg-zinc-600 text-zinc-100" : "text-zinc-400"}"
              title=${enhanced ? "enhanced result (b)" : "enhanced JPEG (needs Export JPEG)"}>after</button>
          </div>
          ${faces.length > 0 && !showingAfter && html`
            <button onClick=${() => setShowFaces((v) => !v)}
              class="px-2 py-0.5 rounded text-xs ${showFaces ? "bg-emerald-800 text-emerald-100" : "bg-zinc-800 text-zinc-400"}">
              faces (${faces.length})</button>`}
          <button onClick=${onClose} class="px-3 py-1 bg-zinc-800 rounded">close (esc)</button>
        </div>
      </div>

      <div ref=${wrapRef} class="flex-1 relative flex items-center justify-center overflow-hidden p-4 min-h-0">
        <img ref=${imgRef} src=${imgSrc} alt=${data.filename ?? data.hash}
             class="max-h-[calc(100vh-18rem)] max-w-full min-w-0 object-contain"
             onLoad=${(e) => {
               if (!showingAfter) setNat({ w: e.target.naturalWidth, h: e.target.naturalHeight });
               measure();
             }}
             onError=${() => { if (showAfter) setAfterError(true); }} />
        ${showFaces && !showingAfter && nat && box && faces.map((f, i) => html`
          <div key=${i} class="absolute border-2 border-emerald-400/80 pointer-events-none"
               style=${{
                 left: `${box.left + (f.x / nat.w) * box.width}px`,
                 top: `${box.top + (f.y / nat.h) * box.height}px`,
                 width: `${(f.w / nat.w) * box.width}px`,
                 height: `${(f.h / nat.h) * box.height}px`,
               }}>
            <span class="absolute -top-4 left-0 text-[10px] text-emerald-300 bg-black/70 px-1 rounded">
              ${f.score != null ? f.score.toFixed(2) : ""}</span>
          </div>`)}
        ${showAfter && afterError && html`
          <div class="absolute inset-x-0 bottom-0 bg-amber-950/80 text-amber-200 text-xs text-center py-1">
            No enhanced output yet — run Submit & continue (or Export JPEG).
          </div>`}
      </div>

      <div class="px-4 py-3 border-t border-zinc-800 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
        <div>
          <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">scores</div>
          <div>aesthetic: <span class="text-zinc-100">${data.aesthetic_score?.toFixed(2) ?? "—"}</span></div>
          <div>technical: <span class="text-zinc-100">${data.technical_score?.toFixed(2) ?? "—"}</span></div>
          <div class="text-zinc-500">musiq ${data.musiq_score?.toFixed(1) ?? "—"} • maniqa ${data.maniqa_score?.toFixed(2) ?? "—"}</div>
          <div class="text-zinc-500">blur var ${data.blur_var?.toFixed(0) ?? "—"} • ${faces.length} face(s)</div>
          ${data.exposure_flag && data.exposure_flag !== "ok" && html`
            <div class="mt-1">
              <span class="bg-amber-800 text-amber-100 text-[10px] px-1.5 py-0.5 rounded"
                    title=${exposureHelp(data.exposure_flag)}>${data.exposure_flag.replace(/_/g, " ")}</span>
              <span class="ml-1 text-[11px] text-zinc-500">${exposureHelp(data.exposure_flag)}</span>
            </div>`}
        </div>

        ${data.quality_report ? html`<${EngineQualityPanel} qr=${data.quality_report} />` : html`
          <div>
            <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">engine quality</div>
            <div class="text-xs text-zinc-500 italic">
              Run the Enhance stage to compute the quality breakdown.
            </div>
          </div>`}

        <div>
          <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">decision</div>
          <div class="flex gap-1.5 mb-2">
            ${[1, 2, 3, 4, 5].map(n => html`
              <button key=${n}
                class="px-2 py-0.5 rounded border ${d?.stars >= n ? "bg-amber-500 text-black border-amber-500" : "border-zinc-700"}"
                onClick=${() => mutate({ stars: n })}>${n}★</button>`)}
          </div>
          <div class="flex gap-1.5 mb-2">
            <button class="px-2 py-1 rounded ${d?.selected === "yes" ? "bg-emerald-700" : "bg-zinc-800"}"
                    onClick=${() => decideAndAdvance("yes")}>yes <span class="kbd">y</span></button>
            <button class="px-2 py-1 rounded ${d?.selected === "no" ? "bg-rose-800" : "bg-zinc-800"}"
                    onClick=${() => decideAndAdvance("no")}>no <span class="kbd">n</span></button>
            <button class="px-2 py-1 rounded ${(!d || d.selected === "undecided") ? "bg-zinc-600" : "bg-zinc-800"}"
                    onClick=${() => mutate({ selected: "undecided" })}>undecided</button>
          </div>
          <div class="flex gap-1.5">
            <button class="px-2 py-1 rounded ${d?.favorite ? "bg-pink-700" : "bg-zinc-800"}"
                    onClick=${() => mutate({ favorite: !d?.favorite })}>★ favorite <span class="kbd">f</span></button>
          </div>
          <textarea rows="2" placeholder="note…" value=${noteText}
            onInput=${(e) => setNoteText(e.target.value)}
            onBlur=${() => { if ((data.decision?.note ?? "") !== noteText) mutate({ note: noteText }); }}
            class="mt-2 w-full bg-zinc-800 rounded px-2 py-1 text-xs text-zinc-200"></textarea>
          <div class="mt-2 text-xs text-zinc-500">
            Yes → kept in <span class="font-mono">library/</span> + enhanced TIFF.
            No → enhanced TIFF only; original RAW deleted after enhance.
          </div>
        </div>

        <div class="text-xs text-zinc-500">
          <div class="uppercase tracking-wider mb-1">shortcuts</div>
          <div><span class="kbd">1-5</span> stars · <span class="kbd">y/n/u</span> select · <span class="kbd">f</span> fav</div>
          <div><span class="kbd">←/→</span> nav · <span class="kbd">.</span> next undecided · <span class="kbd">b</span> before/after</div>
          <div><span class="kbd">space</span> next · <span class="kbd">esc</span> close</div>
          <div class="mt-2">cluster: ${data.cluster_id ?? "—"} · recommended: ${data.is_recommended ? "yes" : "no"}</div>
          <div>captured ${data.captured_at ?? "—"}</div>
        </div>
      </div>
    </div>`;
}

function Toolbar({ count, total, shown, pendingCount, sort, setSort, view, setView,
                   onMarkAllNo, onSubmitAndContinue, busy }) {
  const truncated = shown != null && shown < count;
  const filtered = total != null && total !== count;
  const tabBtn = (id, label) => html`
    <button onClick=${() => setView(id)}
      class="px-3 py-1 rounded text-sm ${view === id ? "bg-zinc-700 text-zinc-100" : "bg-zinc-900 text-zinc-400"}">
      ${label}
    </button>`;
  return html`
    <div class="flex items-center justify-between border-b border-zinc-800 px-4 py-2 sticky top-0 bg-zinc-950/95 backdrop-blur z-20">
      <div class="flex items-baseline gap-4">
        <span class="text-sm text-zinc-400">
          ${count}${filtered ? ` of ${total}` : ""} photo${count === 1 ? "" : "s"}${truncated ? ` (showing ${shown})` : ""}
        </span>
        <span class="text-sm text-amber-400">${pendingCount} pending decision${pendingCount === 1 ? "" : "s"}</span>
      </div>
      <div class="flex items-center gap-2">
        <div class="flex gap-1 mr-2 p-0.5 bg-zinc-900 border border-zinc-800 rounded">
          ${tabBtn("all", "All")}
          ${tabBtn("clusters", "Clusters")}
        </div>
        ${view === "all" && html`
          <label class="text-xs text-zinc-500">sort</label>
          <select value=${sort} onChange=${(e) => setSort(e.target.value)}
                  class="bg-zinc-900 border border-zinc-800 text-sm rounded px-2 py-1">
            <option value="score">score (technical)</option>
            <option value="captured">captured</option>
          </select>`}
        <button onClick=${onMarkAllNo} disabled=${busy || count === 0}
                class="px-3 py-1.5 rounded bg-rose-800 hover:bg-rose-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm">
          don't keep any RAW
        </button>
        <button onClick=${onSubmitAndContinue} disabled=${busy || pendingCount === 0}
                class="px-3 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm">
          ${busy ? "working…" : `✔ Submit & continue (${pendingCount})`}
        </button>
      </div>
    </div>`;
}

export function ReviewPanel({ pipelineBusy, onSubmitAndContinue }) {
  const [sort, setSort] = useState("score");
  const [view, setView] = useState(() => {
    const m = /view=(all|clusters)/.exec(location.hash);
    return m ? m[1] : "all";
  });
  const [filter, setFilter] = useState({ ...EMPTY_FILTER });
  const { data: queueItems, refetch: refetchQueue } = useQuery(() => api.queue(sort), [sort]);
  const { data: clusters, refetch: refetchClusters } = useQuery(() => api.clusters(), []);
  const { data: pending, refetch: refetchPending } = useQuery(() => api.pending(), []);
  const [openHash, setOpenHash] = useState(null);
  const [viewerHashes, setViewerHashes] = useState(null); // navigation order, snapshotted at open
  const [compareId, setCompareId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const triggerRef = useRef(null); // tile to refocus when the viewer closes

  useEffect(() => { setVisibleCount(PAGE_SIZE); }, [view, sort, filter]);
  const loadMore = useCallback(() => setVisibleCount((c) => c + PAGE_SIZE), []);

  // Keep the current sub-view in the URL so a reload doesn't drop you back on
  // the Ingest log (the pipeline panel remembers itself in app.js).
  useEffect(() => {
    const params = new URLSearchParams(location.hash.slice(1));
    params.set("view", view);
    history.replaceState(null, "", `#${params.toString()}`);
  }, [view]);

  const filteredClusters = useMemo(() => {
    if (!clusters) return null;
    return clusters
      .map((c) => ({ ...c, photos: c.photos.filter((p) => matchesFilter(p, filter)) }))
      .filter((c) => c.photos.length > 0);
  }, [clusters, filter]);

  const rawItems = useMemo(() => {
    if (view === "clusters") return clusters ? clusters.flatMap((c) => c.photos) : null;
    return queueItems ?? null;
  }, [view, queueItems, clusters]);

  const items = useMemo(() => {
    if (view === "clusters") return filteredClusters ? filteredClusters.flatMap((c) => c.photos) : null;
    return queueItems ? queueItems.filter((p) => matchesFilter(p, filter)) : null;
  }, [view, queueItems, filteredClusters, filter]);

  // hash -> live photo, over the unfiltered universe, so viewer navigation can
  // resolve a snapshotted hash even after a decision drops it from the filter.
  const byHash = useMemo(() => {
    const m = new Map();
    for (const p of rawItems ?? []) m.set(p.hash, p);
    return m;
  }, [rawItems]);

  // Counts for the filter chips, over the current view's universe.
  const filterCounts = useMemo(() => {
    const all = rawItems ?? [];
    const c = { all: all.length, undecided: 0, yes: 0, no: 0, fav: 0, faces: 0, enhanced: 0, flagged: 0 };
    for (const p of all) {
      const s = p.decision?.selected;
      if (!s || s === "undecided") c.undecided++;
      else if (s === "yes") c.yes++;
      else if (s === "no") c.no++;
      if (p.decision?.favorite) c.fav++;
      if (p.n_faces > 0) c.faces++;
      if (p.enhanced) c.enhanced++;
      if (p.exposure_flag && p.exposure_flag !== "ok") c.flagged++;
    }
    return c;
  }, [rawItems]);

  const compareCluster = useMemo(
    () => (compareId != null && clusters ? clusters.find((c) => c.id === compareId) ?? null : null),
    [compareId, clusters]
  );

  const refreshAll = useCallback(() => {
    refetchQueue(); refetchClusters(); refetchPending();
  }, [refetchQueue, refetchClusters, refetchPending]);

  const onStar = useCallback(async (hash, stars) => {
    await api.decide({ photo_hash: hash, stars });
    refreshAll();
  }, [refreshAll]);

  // Opening a photo snapshots the currently-visible order. Navigation then walks
  // that snapshot, so deciding a photo (which may drop it from the active filter)
  // never shifts the list out from under ← / →. The snapshot clears on close.
  const openPhoto = useCallback((hash) => {
    triggerRef.current = document.activeElement;
    setViewerHashes((items ?? []).map((p) => p.hash));
    setOpenHash(hash);
  }, [items]);

  const closeViewer = useCallback(() => {
    setOpenHash(null);
    setViewerHashes(null);
    const el = triggerRef.current;
    if (el && typeof el.focus === "function") setTimeout(() => el.focus(), 0);
  }, []);

  const openIndex = useMemo(
    () => (viewerHashes ? viewerHashes.indexOf(openHash) : -1),
    [viewerHashes, openHash]
  );
  const next = useCallback(() => {
    if (!viewerHashes || viewerHashes.length === 0) return;
    const i = openIndex < 0 ? 0 : (openIndex + 1) % viewerHashes.length;
    setOpenHash(viewerHashes[i]);
  }, [viewerHashes, openIndex]);
  const prev = useCallback(() => {
    if (!viewerHashes || viewerHashes.length === 0) return;
    const i = openIndex <= 0 ? viewerHashes.length - 1 : openIndex - 1;
    setOpenHash(viewerHashes[i]);
  }, [viewerHashes, openIndex]);
  const nextUndecided = useCallback(() => {
    if (!viewerHashes || viewerHashes.length === 0) return;
    const start = openIndex < 0 ? -1 : openIndex;
    for (let k = 1; k <= viewerHashes.length; k++) {
      const h = viewerHashes[(start + k) % viewerHashes.length];
      const sel = byHash.get(h)?.decision?.selected;
      if (!sel || sel === "undecided") { setOpenHash(h); return; }
    }
    setStatus("No undecided photos left in the current view.");
  }, [viewerHashes, openIndex, byHash]);

  const onClusterAction = useCallback(async (clusterId, mode) => {
    if (mode === "reject_all" && !confirm(
      `Reject every photo in cluster #${clusterId}?\n\n` +
      `Each is staged "no": the RAW is enhanced, then the source RAW is deleted ` +
      `after Submit. You can still change this before Submit.`
    )) return;
    setBusy(true);
    try {
      const res = await api.decideCluster(clusterId, mode);
      setStatus(`Cluster #${clusterId}: staged ${res.staged} decision(s) (${mode.replace(/_/g, " ")}).`);
      refreshAll();
    } catch (e) {
      setStatus(`Cluster action failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [refreshAll]);

  const onCompareDecide = useCallback(async (photoHash, selected) => {
    await api.decide({ photo_hash: photoHash, selected });
    refreshAll();
  }, [refreshAll]);

  const onKeepOnly = useCallback(async (keepHash) => {
    if (!compareCluster) return;
    setBusy(true);
    try {
      // Skip already-applied (submitted) photos, mirroring stage_cluster.
      await Promise.all(compareCluster.photos
        .filter((p) => !p.decision?.applied)
        .map((p) => api.decide({ photo_hash: p.hash, selected: p.hash === keepHash ? "yes" : "no" })));
      setStatus(`Kept ${keepHash.slice(0, 8)}; rejected the rest of the cluster.`);
      refreshAll();
    } catch (e) {
      setStatus(`Keep-only failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [compareCluster, refreshAll]);

  const submitAndContinue = useCallback(() => {
    const n = pending?.length ?? 0;
    const all = queueItems ?? [];
    const undecided = all.filter((p) => {
      const s = p.decision?.selected;
      return !s || s === "undecided";
    }).length;
    const undecidedNote = undecided > 0
      ? `\n\n${undecided} photo(s) are still undecided — they will be left ` +
        "untouched (not moved, not enhanced)."
      : "";
    if (!confirm(
      `Submit ${n} decision(s) and continue?\n\n` +
      `This runs: Submit (moves files on disk) → Enhance → Export JPEG.\n` +
      `Photos marked "no" get their source RAW deleted after enhancement.` +
      undecidedNote
    )) return;
    onSubmitAndContinue();
  }, [pending, queueItems, onSubmitAndContinue]);

  const onMarkAllNo = useCallback(async () => {
    if (!confirm(
      `Mark EVERY photo in the batch as NO? Originals will NOT be kept in ` +
      `library — every RAW is enhanced, then the source RAW is deleted after ` +
      `processing. You can still review before Submit.`
    )) return;
    setBusy(true);
    try {
      const res = await api.decideAll("no");
      setStatus(`Marked ${res.staged} photo(s) as no.`);
      refreshAll();
    } catch (e) {
      setStatus(`Mark all failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [refreshAll]);

  const emptyMsg = (rawItems && rawItems.length > 0)
    ? "No photos match the current filter."
    : "No photos yet. Run the Ingest stage first.";

  const body = (() => {
    if (view === "clusters") {
      if (!filteredClusters) return html`<div class="p-8 text-zinc-500">loading…</div>`;
      if (filteredClusters.length === 0) return html`<div class="p-8 text-zinc-500">${emptyMsg}</div>`;
      return html`<${ClusterView} clusters=${filteredClusters} onOpen=${openPhoto} onStar=${onStar}
                    onClusterAction=${onClusterAction} onCompare=${(c) => setCompareId(c.id)}
                    busy=${busy || pipelineBusy} />`;
    }
    if (!items) return html`<div class="p-8 text-zinc-500">loading…</div>`;
    if (items.length === 0) return html`<div class="p-8 text-zinc-500">${emptyMsg}</div>`;
    return html`<${PagedGrid} items=${items} visible=${visibleCount}
                               onMore=${loadMore} onOpen=${openPhoto} onStar=${onStar} />`;
  })();

  return html`
    <${Fragment}>
      <${Toolbar} count=${items?.length ?? 0} total=${rawItems?.length ?? null}
                  shown=${(view === "all" && items) ? Math.min(visibleCount, items.length) : null}
                  pendingCount=${pending?.length ?? 0}
                  sort=${sort} setSort=${setSort} view=${view} setView=${setView}
                  onMarkAllNo=${onMarkAllNo}
                  onSubmitAndContinue=${submitAndContinue}
                  busy=${busy || pipelineBusy} />
      <${FilterBar} filter=${filter} setFilter=${setFilter} counts=${filterCounts} />
      ${status && html`<div class="px-4 py-1 text-xs bg-zinc-900 border-b border-zinc-800">${status}</div>`}
      ${body}
      ${openHash && html`<${DetailModal} hash=${openHash}
                          onClose=${closeViewer}
                          onPrev=${prev} onNext=${next} onNextUndecided=${nextUndecided}
                          onMutated=${refreshAll} />`}
      ${compareCluster && html`<${CompareModal} cluster=${compareCluster}
                          onClose=${() => setCompareId(null)}
                          onDecide=${onCompareDecide} onKeepOnly=${onKeepOnly}
                          busy=${busy} />`}
    </${Fragment}>`;
}
