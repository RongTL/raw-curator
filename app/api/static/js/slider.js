// Before/after wipe comparison. Two stacked <img>; the "after" is clipped by a
// draggable handle. Pure CSS clip-path, pointer-driven; no external deps.
import { html, useRef, useState } from "./ui.js";

export function BeforeAfterSlider({ beforeSrc, afterSrc, fallbackSrc, dw, dh, onNatSize }) {
  const [pos, setPos] = useState(50);        // handle position, % from left
  const [afterOk, setAfterOk] = useState(true);
  const [beforeSrcState, setBeforeSrcState] = useState(beforeSrc);
  const box = useRef(null);

  const onMove = (clientX) => {
    const el = box.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };
  const onDown = (e) => {
    onMove(e.clientX);
    const move = (ev) => onMove(ev.clientX);
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const ratio = dw && dh ? `${dw} / ${dh}` : "3 / 2";
  return html`
    <div ref=${box} class="relative select-none touch-none"
         style=${{ aspectRatio: ratio, width: `min(100cqw, calc(100cqh * ${(dw && dh) ? dw / dh : 1.5}))` }}
         onPointerDown=${onDown}>
      <img src=${beforeSrcState} alt="before"
           class="absolute inset-0 w-full h-full object-contain pointer-events-none"
           onLoad=${(e) => onNatSize?.(e.target.naturalWidth, e.target.naturalHeight)}
           onError=${() => fallbackSrc && setBeforeSrcState(fallbackSrc)} />
      ${afterOk && afterSrc && html`
        <img src=${afterSrc} alt="after"
             class="absolute inset-0 w-full h-full object-contain pointer-events-none"
             style=${{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
             onError=${() => setAfterOk(false)} />`}
      ${afterOk && afterSrc && html`
        <div class="absolute inset-y-0 w-0.5 bg-white/80 pointer-events-none" style=${{ left: `${pos}%` }}>
          <div class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-0 w-6 h-6 rounded-full bg-white/90 shadow flex items-center justify-center text-black text-xs">↔</div>
        </div>`}
      <div class="absolute bottom-1 left-1 text-[10px] px-1 rounded bg-black/60 text-zinc-200">before</div>
      ${afterOk && afterSrc && html`<div class="absolute bottom-1 right-1 text-[10px] px-1 rounded bg-black/60 text-zinc-200">after</div>`}
    </div>`;
}
