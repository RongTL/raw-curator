"""Pixel-geometry helpers shared by the enhancement stages."""

from __future__ import annotations

from collections.abc import Sequence

Box = tuple[int, int, int, int]  # x, y, w, h


def scale_boxes(
    boxes: Sequence[Box], src_size: tuple[int, int], dst_size: tuple[int, int]
) -> list[Box]:
    """Rescale boxes from one image size to another (sizes are (w, h)); clamp to the target."""
    sx = dst_size[0] / max(src_size[0], 1)
    sy = dst_size[1] / max(src_size[1], 1)
    out: list[Box] = []
    for x, y, w, h in boxes:
        nx, ny = int(round(x * sx)), int(round(y * sy))
        nw, nh = int(round(w * sx)), int(round(h * sy))
        nx = min(max(nx, 0), dst_size[0] - 1)
        ny = min(max(ny, 0), dst_size[1] - 1)
        nw = min(nw, dst_size[0] - nx)
        nh = min(nh, dst_size[1] - ny)
        out.append((nx, ny, nw, nh))
    return out
