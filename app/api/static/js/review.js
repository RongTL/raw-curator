// Review UI — ported from the pre-control-center index.html inline app.
// Components are unchanged except: imports, and ReviewPanel replacing App/Header.
import { api } from "./api.js";
import {
  Fragment, createPortal, html, useCallback, useEffect, useMemo, useRef, useState,
  useKeyboardShortcuts, useQuery,
} from "./ui.js";
import { EMPTY_FILTER, FilterBar, matchesFilter } from "./filters.js";
import { CompareModal } from "./compare.js";
import { BeforeAfterSlider } from "./slider.js";

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
  const c = decision?.export_choice;
  if (!c || c === "undecided") return null;
  const color = c === "original" ? "bg-sky-700" : c === "enhanced" ? "bg-emerald-700" : "bg-rose-800";
  return html`<span class="${color} text-xs px-1.5 py-0.5 rounded">${c}</span>`;
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

// `position`/`total` locate this photo in the list it was opened from (1-based,
// in the current sort); `filtered` marks that list as a filter subset of the view.
function DetailModal({ hash, onClose, onPrev, onNext, onNextUndecided, onMutated,
                       position = null, total = 0, filtered = false }) {
  const { data, refetch } = useQuery(() => api.photo(hash), [hash]);
  const [viewMode, setViewMode] = useState("slider"); // "slider" | "before" | "after"
  const [showFaces, setShowFaces] = useState(false);
  const [nat, setNat] = useState(null); // preview natural size; fallback when the API has no dims
  const [noteText, setNoteText] = useState("");
  const rootRef = useRef(null); // modal container; focused on open so keys don't leak to the grid

  // Move focus into the dialog once its content renders. Otherwise the tile that
  // opened it keeps focus, and since React delegates events below window, a plain
  // Enter would re-fire that tile's onKeyDown and jump back to the first photo.
  useEffect(() => { rootRef.current?.focus(); }, [!!data]);

  // The viewer is a full-screen portal over <body>; lock the grid behind it so it
  // can't scroll or be clicked, and restore it on close. overflow:hidden keeps the
  // grid's scroll position for when the viewer closes.
  useEffect(() => {
    const main = document.querySelector("main");
    if (!main) return undefined;
    const prev = main.style.overflow;
    main.style.overflow = "hidden";
    return () => { main.style.overflow = prev; };
  }, []);

  // Reset the view mode + measured size per photo (a new frame may have no
  // enhanced version). The faces toggle is intentionally sticky across
  // navigation, so reviewing faces across a burst doesn't need re-enabling.
  useEffect(() => { setViewMode("slider"); setNat(null); }, [hash]);
  useEffect(() => { setNoteText(data?.decision?.note ?? ""); }, [hash, data?.decision?.note]);

  const mutate = useCallback(async (patch) => {
    await api.decide({ photo_hash: hash, ...patch });
    await refetch();
    onMutated?.();
  }, [hash, refetch, onMutated]);

  // The culling flow: an export-choice decision saves, then jumps to the next
  // photo. We advance immediately after the save lands and refresh the grid in
  // the background — no extra keypress per photo.
  const chooseAndAdvance = useCallback(async (choice) => {
    await api.decide({ photo_hash: hash, export_choice: choice });
    onMutated?.();
    onNext?.();
  }, [hash, onMutated, onNext]);

  useKeyboardShortcuts(useMemo(() => ({
    Escape: onClose,
    ArrowLeft: onPrev,
    ArrowRight: onNext,
    " ": onNext,
    ".": () => onNextUndecided?.(),
    "1": () => mutate({ stars: 1 }),
    "2": () => mutate({ stars: 2 }),
    "3": () => mutate({ stars: 3 }),
    "4": () => mutate({ stars: 4 }),
    "5": () => mutate({ stars: 5 }),
    "0": () => mutate({ stars: 0 }),
    "o": () => chooseAndAdvance("original"),
    // No-op when there's no enhanced render (non-RAW / enhance-failed), matching
    // the disabled button — `data.quality_report.score_q_after != null` means one exists.
    "e": () => { if (data?.quality_report?.score_q_after != null) chooseAndAdvance("enhanced"); },
    "x": () => chooseAndAdvance("discard"),
    "r": () => mutate({ keep_raw: !data?.decision?.keep_raw }),
    "b": () => setViewMode((m) => (m === "slider" ? "before" : m === "before" ? "after" : "slider")),
    "f": () => mutate({ favorite: !data?.decision?.favorite }),
  }), [mutate, chooseAndAdvance, data, onClose, onPrev, onNext, onNextUndecided]));

  if (!data) return createPortal(html`
    <div class="fixed inset-0 z-50 bg-zinc-950 flex items-center justify-center">
      <div class="text-zinc-400">loading…</div>
    </div>`, document.body);

  const d = data.decision;
  // An enhanced render only exists when enhance ran and scored it. Non-RAW and
  // enhance-failed photos have none, so the "enhanced" choice must be blocked.
  const enhanced = data.quality_report?.score_q_after != null;
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

  // Stage sizing AND face-box placement both use the LOADED preview's natural
  // size (`nat`) — the exact pixel space the face bboxes were detected in and the
  // true displayed aspect ratio. We deliberately do NOT use data.width/height:
  // those are the full-resolution original dims (from EXIF ImageWidth/Height) and
  // may carry the sensor's pre-rotation orientation, so they'd both misplace the
  // boxes (wrong scale) and risk the wrong aspect ratio. Until the image loads we
  // fall back to a plain contained <img> (which already fits); before/after share
  // an aspect ratio, so toggling never reflows.
  const dw = nat?.w ?? null;
  const dh = nat?.h ?? null;
  const ratioReady = Boolean(dw && dh);

  return createPortal(html`
    <div ref=${rootRef} tabindex="-1" role="dialog" aria-modal="true"
         class="fixed inset-0 z-50 bg-zinc-950 outline-none flex flex-col
                lg:grid lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-[auto_minmax(0,1fr)]">
      <div class="flex items-center justify-between px-4 py-2 border-b border-zinc-800 text-sm lg:col-span-2">
        <div class="flex items-center gap-3 min-w-0">
          <button onClick=${onPrev} class="px-2 py-1 bg-zinc-800 rounded" title="prev (←)">←</button>
          <button onClick=${onNext} class="px-2 py-1 bg-zinc-800 rounded" title="next (→)">→</button>
          <button onClick=${() => onNextUndecided?.()} class="px-2 py-1 bg-zinc-800 rounded text-xs"
                  title="next undecided (.)">next undecided</button>
          ${position != null && total > 0 && html`
            <span class="font-mono text-xs text-zinc-400 whitespace-nowrap"
                  title=${filtered
                    ? "position in the filtered list this photo was opened from"
                    : "position in the current sort order"}>
              ${position} / ${total}${filtered ? " · filtered" : ""}
            </span>`}
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
            ${["slider", "before", "after"].map((m) => html`
              <button key=${m} onClick=${() => setViewMode(m)}
                class="px-2 py-0.5 rounded ${viewMode === m ? "bg-zinc-600 text-zinc-100" : "text-zinc-400"}"
                title=${m === "slider" ? "before/after wipe (b)" : `${m} only (b)`}>${m}</button>`)}
          </div>
          ${faces.length > 0 && viewMode === "before" && html`
            <button onClick=${() => setShowFaces((v) => !v)}
              class="px-2 py-0.5 rounded text-xs ${showFaces ? "bg-emerald-800 text-emerald-100" : "bg-zinc-800 text-zinc-400"}">
              faces (${faces.length})</button>`}
          <button onClick=${onClose} class="px-3 py-1 bg-zinc-800 rounded">close (esc)</button>
        </div>
      </div>

      <div class="relative flex-1 min-h-0 flex items-center justify-center p-4 overflow-hidden
                  [container-type:size] lg:col-start-1 lg:row-start-2 lg:min-w-0">
        ${viewMode === "slider"
          ? html`<${BeforeAfterSlider} beforeSrc=${data.before_url} afterSrc=${data.after_url}
                    fallbackSrc=${data.preview_url} dw=${dw} dh=${dh}
                    onNatSize=${(w, h) => setNat({ w, h })} />`
          : ratioReady ? html`
            <div class="relative"
                 style=${{ aspectRatio: `${dw} / ${dh}`, width: `min(100cqw, calc(100cqh * ${dw / dh}))` }}>
              <img src=${viewMode === "after" ? data.after_url : data.before_url}
                   alt=${data.filename ?? data.hash}
                   class="absolute inset-0 w-full h-full object-contain"
                   onLoad=${(e) => setNat({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                   onError=${(e) => { if (viewMode === "before") e.target.src = data.preview_url; }} />
              ${showFaces && viewMode === "before" && faces.map((f, i) => html`
                <div key=${i} class="absolute border-2 border-emerald-400/80 pointer-events-none"
                     style=${{
                       left: `${(f.x / dw) * 100}%`,
                       top: `${(f.y / dh) * 100}%`,
                       width: `${(f.w / dw) * 100}%`,
                       height: `${(f.h / dh) * 100}%`,
                     }}>
                  <span class="absolute -top-4 left-0 text-[10px] text-emerald-300 bg-black/70 px-1 rounded">
                    ${f.score != null ? f.score.toFixed(2) : ""}</span>
                </div>`)}
            </div>`
          : html`
            <img src=${viewMode === "after" ? data.after_url : data.before_url}
                 alt=${data.filename ?? data.hash}
                 class="max-h-full max-w-full object-contain"
                 onLoad=${(e) => setNat({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                 onError=${(e) => { if (viewMode === "before") e.target.src = data.preview_url; }} />`}
      </div>

      <div class="px-4 py-3 border-t border-zinc-800 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm
                  lg:grid-cols-1 lg:content-start lg:border-t-0 lg:border-l lg:overflow-y-auto
                  lg:min-h-0 lg:col-start-2 lg:row-start-2">
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
          <div class="text-zinc-500 text-xs uppercase tracking-wider mb-1">export choice</div>
          <div class="flex gap-1.5 mb-2">
            <button class="px-2 py-1 rounded ${d?.export_choice === "original" ? "bg-sky-700" : "bg-zinc-800"}"
                    onClick=${() => chooseAndAdvance("original")}>original <span class="kbd">o</span></button>
            <button disabled=${!enhanced}
                    class="px-2 py-1 rounded ${d?.export_choice === "enhanced" ? "bg-emerald-700" : "bg-zinc-800"} ${!enhanced ? "opacity-40 cursor-not-allowed" : ""}"
                    title=${enhanced ? "" : "no enhanced version for this photo"}
                    onClick=${() => chooseAndAdvance("enhanced")}>enhanced <span class="kbd">e</span></button>
            <button class="px-2 py-1 rounded ${d?.export_choice === "discard" ? "bg-rose-800" : "bg-zinc-800"}"
                    onClick=${() => chooseAndAdvance("discard")}>discard <span class="kbd">x</span></button>
          </div>
          ${!enhanced && html`<div class="-mt-1 mb-2 text-[11px] text-zinc-500">no enhanced version</div>`}
          <label class="flex items-center gap-2 text-xs mb-2">
            <input type="checkbox" checked=${!!d?.keep_raw} onChange=${() => mutate({ keep_raw: !d?.keep_raw })} />
            keep RAW <span class="kbd">r</span>
          </label>
          <div class="flex gap-1.5 mb-2">
            ${[1, 2, 3, 4, 5].map(n => html`
              <button key=${n}
                class="px-2 py-0.5 rounded border ${d?.stars >= n ? "bg-amber-500 text-black border-amber-500" : "border-zinc-700"}"
                onClick=${() => mutate({ stars: n })}>${n}★</button>`)}
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
            Exports the chosen version as a JPEG. Keep RAW → archived to
            <span class="font-mono">library/</span>; off → RAW deleted after export.
          </div>
        </div>

        <div class="text-xs text-zinc-500">
          <div class="uppercase tracking-wider mb-1">shortcuts</div>
          <div><span class="kbd">1-5</span> stars · <span class="kbd">o/e/x</span> export · <span class="kbd">r</span> keep RAW · <span class="kbd">f</span> fav</div>
          <div><span class="kbd">←/→</span> nav · <span class="kbd">.</span> next undecided · <span class="kbd">b</span> view mode</div>
          <div><span class="kbd">space</span> next · <span class="kbd">esc</span> close</div>
          <div class="mt-2">cluster: ${data.cluster_id ?? "—"} · recommended: ${data.is_recommended ? "✓" : "—"}</div>
          <div>captured ${data.captured_at ?? "—"}</div>
        </div>
      </div>
    </div>`, document.body);
}

function Toolbar({ count, total, shown, pendingCount, sort, setSort, view, setView,
                   onBulk, onSubmitAndContinue, busy }) {
  const truncated = shown != null && shown < count;
  const filtered = total != null && total !== count;
  const tabBtn = (id, label) => html`
    <button onClick=${() => setView(id)}
      class="px-3 py-1 rounded text-sm ${view === id ? "bg-zinc-700 text-zinc-100" : "bg-zinc-900 text-zinc-400"}">
      ${label}
    </button>`;
  const bulkDisabled = busy || count === 0;
  const bulkBtn = (body, label, cls) => html`
    <button onClick=${() => onBulk(body)} disabled=${bulkDisabled}
      class="px-2 py-1.5 rounded text-xs ${cls} disabled:bg-zinc-800 disabled:text-zinc-500">
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
            <option value="captured">date taken</option>
            <option value="filename">filename</option>
            <option value="score">score (technical)</option>
          </select>`}
        <div class="flex gap-1">
          ${bulkBtn({ export_choice: "enhanced" }, "use enhanced for all", "bg-emerald-800 hover:bg-emerald-700")}
          ${bulkBtn({ export_choice: "original" }, "use original for all", "bg-sky-800 hover:bg-sky-700")}
          ${bulkBtn({ export_choice: "discard" }, "discard all", "bg-rose-900 hover:bg-rose-800")}
        </div>
        <div class="flex gap-1">
          ${bulkBtn({ keep_raw: true }, "keep all RAWs", "bg-zinc-700 hover:bg-zinc-600 text-zinc-100")}
          ${bulkBtn({ keep_raw: false }, "keep no RAW", "bg-zinc-700 hover:bg-zinc-600 text-zinc-100")}
        </div>
        <button onClick=${onSubmitAndContinue} disabled=${busy || pendingCount === 0}
                class="px-3 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 disabled:bg-zinc-800 disabled:text-zinc-500 text-sm">
          ${busy ? "working…" : `✔ Export selected (${pendingCount})`}
        </button>
      </div>
    </div>`;
}

// Batch-wide bulk actions each touch every photo, so they confirm first. The
// wording tracks the consequence: export-choice (incl. discard) only stages a
// choice and is non-destructive; keep-RAW off means the source RAW is deleted
// at Export, once its JPEG exists.
function bulkConfirmText(body) {
  if ("export_choice" in body) {
    const c = body.export_choice;
    if (c === "discard") {
      return "Discard EVERY photo? No JPEG is exported for discarded photos. " +
        "This is non-destructive — you can change any photo before Export.";
    }
    return `Set EVERY photo to export its ${c} version? ` +
      "You can still change individual photos before Export.";
  }
  if ("keep_raw" in body) {
    return body.keep_raw
      ? "Keep the source RAW for EVERY photo? Kept RAWs are archived to " +
        "library/ at Export. You can change any photo before Export."
      : "Set keep-RAW OFF for EVERY photo? At Export, each source RAW is " +
        "deleted after its JPEG is written. You can still change any photo before Export.";
  }
  return "Apply this to EVERY photo?";
}

export function ReviewPanel({ pipelineBusy, onSubmitAndContinue }) {
  const [sort, setSort] = useState(() => {
    const m = /sort=(captured|filename|score)/.exec(location.hash);
    return m ? m[1] : "captured";
  });
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

  // Keep the current sub-view and sort in the URL so a reload doesn't drop you
  // back on the Ingest log or lose your ordering (the pipeline panel remembers
  // itself in app.js).
  useEffect(() => {
    const params = new URLSearchParams(location.hash.slice(1));
    params.set("view", view);
    params.set("sort", sort);
    history.replaceState(null, "", `#${params.toString()}`);
  }, [view, sort]);

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
    const c = { all: all.length, undecided: 0, original: 0, enhanced: 0, discard: 0, fav: 0, faces: 0, flagged: 0 };
    for (const p of all) {
      const ch = p.decision?.export_choice;
      if (!ch || ch === "undecided") c.undecided++;
      else if (ch === "original") c.original++;
      else if (ch === "enhanced") c.enhanced++;
      else if (ch === "discard") c.discard++;
      if (p.decision?.favorite) c.fav++;
      if (p.n_faces > 0) c.faces++;
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
  // The viewer walks a snapshot of the visible list; when that snapshot is
  // smaller than the view's universe a filter was active, so the counter says so.
  const viewerFiltered = viewerHashes != null && rawItems != null
    && viewerHashes.length < rawItems.length;
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
      const ch = byHash.get(h)?.decision?.export_choice;
      if (!ch || ch === "undecided") { setOpenHash(h); return; }
    }
    setStatus("No undecided photos left in the current view.");
  }, [viewerHashes, openIndex, byHash]);

  const onClusterAction = useCallback(async (clusterId, mode) => {
    if (mode === "reject_all" && !confirm(
      `Discard every photo in cluster #${clusterId}?\n\n` +
      `Each is staged "discard": no JPEG is exported for it. This is ` +
      `non-destructive — you can change any photo before Export.`
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

  const onCompareDecide = useCallback(async (photoHash, choice) => {
    await api.decide({ photo_hash: photoHash, export_choice: choice });
    refreshAll();
  }, [refreshAll]);

  const onKeepOnly = useCallback(async (keepHash) => {
    if (!compareCluster) return;
    setBusy(true);
    try {
      // Skip already-applied (submitted) photos, mirroring stage_cluster.
      await Promise.all(compareCluster.photos
        .filter((p) => !p.decision?.applied)
        .map((p) => api.decide({
          photo_hash: p.hash,
          export_choice: p.hash === keepHash ? "enhanced" : "discard",
        })));
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
      const c = p.decision?.export_choice;
      return !c || c === "undecided";
    }).length;
    const undecidedNote = undecided > 0
      ? `\n\n${undecided} photo(s) are still undecided — they will be left ` +
        "untouched (not exported)."
      : "";
    if (!confirm(
      `Export ${n} photo(s)? Runs Export: writes JPEGs and applies your ` +
      `keep-RAW choices (RAWs with keep-RAW off are deleted after their JPEG ` +
      `is written).` +
      undecidedNote
    )) return;
    onSubmitAndContinue();
  }, [pending, queueItems, onSubmitAndContinue]);

  const onBulk = useCallback(async (body) => {
    if (!confirm(bulkConfirmText(body))) return;
    setBusy(true);
    try {
      const res = await api.decideAll(body);
      setStatus(`Staged ${res.staged} photo(s).`);
      refreshAll();
    } catch (e) {
      setStatus(`Bulk action failed: ${e.message}`);
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
                  onBulk=${onBulk}
                  onSubmitAndContinue=${submitAndContinue}
                  busy=${busy || pipelineBusy} />
      <${FilterBar} filter=${filter} setFilter=${setFilter} counts=${filterCounts} />
      ${status && html`<div class="px-4 py-1 text-xs bg-zinc-900 border-b border-zinc-800">${status}</div>`}
      ${body}
      ${openHash && html`<${DetailModal} hash=${openHash}
                          onClose=${closeViewer}
                          onPrev=${prev} onNext=${next} onNextUndecided=${nextUndecided}
                          onMutated=${refreshAll}
                          position=${openIndex >= 0 ? openIndex + 1 : null}
                          total=${viewerHashes?.length ?? 0}
                          filtered=${viewerFiltered} />`}
      ${compareCluster && html`<${CompareModal} cluster=${compareCluster}
                          onClose=${() => setCompareId(null)}
                          onDecide=${onCompareDecide} onKeepOnly=${onKeepOnly}
                          busy=${busy} />`}
    </${Fragment}>`;
}
