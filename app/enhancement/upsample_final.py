"""Parse `settings.enhance_target_res` into an explicit target (w, h).

`settings.enhance_target_res` accepts:
    "native"        -> the source RAW's native (w, h)
    "200%"          -> 2x native (any "<int>%" works)
    "WIDTHxHEIGHT"  -> explicit pixel size, e.g. "3840x2160"
"""

from __future__ import annotations


def parse_target(spec: str, native: tuple[int, int]) -> tuple[int, int]:
    native_w, native_h = native
    s = spec.strip().lower()
    if s == "native":
        return native_w, native_h
    if s.endswith("%"):
        pct = float(s[:-1]) / 100.0
        return max(1, int(round(native_w * pct))), max(1, int(round(native_h * pct)))
    if "x" in s:
        w_str, h_str = s.split("x", 1)
        return max(1, int(w_str)), max(1, int(h_str))
    raise ValueError(f"unrecognised enhance_target_res: {spec!r}")
