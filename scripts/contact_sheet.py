"""Before/after contact sheets for owner review: darktable default vs enhanced master.

STALE — pending rework. This tool reads the enhanced master from
``photos/exported/*.tif``, which the before/after export-choice workflow no
longer writes: enhance now emits per-hash render JPEGs under
``cache/enhanced/<hash>.{before,after}.full.jpg`` instead. Repointing is not a
clean swap (this script keys off the TIFF/original-file *stem*, whereas the new
renders are keyed by *photo hash*, so it needs a DB stem->hash lookup). It is a
manual owner-only look-review tool, not run in CI. Until reworked it produces no
sheets (the ``exported/`` glob is empty).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import settings
from app.ingest.decode import load_tiff_rgb8

PREVIEWS = Path("tests/data/_review/previews")  # <stem>_dt_default.jpg from analyze_plans.py


def _crop_center(im: Image.Image, size: int = 600) -> Image.Image:
    w, h = im.size
    return im.crop(((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tif in sorted((settings.photos / "exported").rglob("*.tif")):
        before_path = PREVIEWS / f"{tif.stem}_dt_default.jpg"
        if not before_path.exists():
            continue
        after_full = Image.fromarray(load_tiff_rgb8(tif))
        before_full = Image.open(before_path).convert("RGB").resize(after_full.size)
        tiles = [
            before_full.copy(),
            after_full.copy(),
            _crop_center(before_full),
            _crop_center(after_full),
        ]
        for t in tiles[:2]:
            t.thumbnail((900, 900))
        sheet = Image.new("RGB", (1800, 1500), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 2) * 900, (i // 2) * 900 if i < 2 else 900))
        d = ImageDraw.Draw(sheet)
        d.text(
            (8, 8), f"{tif.stem}  before | after  (bottom: 100% centre crops)", fill=(255, 255, 0)
        )
        sheet.save(out_dir / f"{tif.stem}.jpg", quality=85)
        print(out_dir / f"{tif.stem}.jpg")


if __name__ == "__main__":
    main()
