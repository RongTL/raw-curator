// Shared React binding + hooks (no build step; esm.sh CDN).
import React from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import htm from "https://esm.sh/htm@3.1.1";

export const html = htm.bind(React.createElement);
export { React, createRoot };
export const { useState, useEffect, useCallback, useMemo, useRef, Fragment } = React;

export function useQuery(fetcher, deps) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const refetch = useCallback(async () => {
    setLoading(true); setError(null);
    try { setData(await fetcher()); }
    catch (e) { setError(e); }
    finally { setLoading(false); }
  }, deps);
  useEffect(() => { refetch(); }, [refetch]);
  return { data, error, loading, refetch };
}

export function useKeyboardShortcuts(handlers) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const h = handlers[e.key];
      if (h) { e.preventDefault(); h(e); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handlers]);
}

// Poll `fn` every `ms` (fires immediately). Errors are swallowed so a
// transient fetch failure doesn't kill the loop.
export function usePoll(fn, ms) {
  useEffect(() => {
    let alive = true;
    const tick = () => { if (alive) fn().catch(() => {}); };
    tick();
    const id = setInterval(tick, ms);
    return () => { alive = false; clearInterval(id); };
  }, [fn, ms]);
}
