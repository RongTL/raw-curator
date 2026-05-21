"""Fetch every model weight raw-curator needs into /data/models/.

Layout produced (all under $RAWCURATOR_MODELS, default /data/models):
    hf/<repo>/...                       Hugging Face snapshots (CLIP, aesthetic v2.5)
    torch/hub/pyiqa/*.pth               populated lazily by pyiqa on first score run
    RealESRGAN_x2plus.pth               x2 super-resolution weights
    scunet_color_real_psnr.pth          SCUNet color denoise weights
    CodeFormer/weights/CodeFormer/      CodeFormer face restore weights
    insightface/models/buffalo_l/       extracted InsightFace detect+arcface

Idempotent: every step checks for an existing artifact before downloading.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPOS = [
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "google/siglip-so400m-patch14-384",
    # NOTE: the aesthetic-predictor-v2-5 head ships inside the PyPI package itself;
    # the model HF repo ("discus0434/aesthetic-predictor-v2-5") is intentionally not listed.
]

# (url, dest-relative-to-MODELS-root, expected-min-bytes)
DIRECT_FILES: list[tuple[str, str, int]] = [
    (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus.pth",
        60_000_000,
    ),
    (
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "scunet_color_real_psnr.pth",
        60_000_000,
    ),
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "CodeFormer/weights/CodeFormer/codeformer.pth",
        300_000_000,
    ),
    # CodeFormer's RetinaFace detector and parsing net are normally fetched at
    # runtime by codeformer-pip into its site-packages dir. With --rm containers
    # that's re-downloaded every run, so we pre-cache them here and `app/__init__`
    # symlinks them into the package's expected path.
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "CodeFormer/weights/facelib/detection_Resnet50_Final.pth",
        100_000_000,
    ),
    (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        "CodeFormer/weights/facelib/parsing_parsenet.pth",
        80_000_000,
    ),
]

INSIGHTFACE_ZIP = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)


def _ensure_direct(models_root: Path) -> None:
    for url, rel, min_bytes in DIRECT_FILES:
        dest = models_root / rel
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"[skip] {rel} (already {dest.stat().st_size:,} bytes)")
            continue
        print(f"[get]  {rel}")
        _download(url, dest)


def _ensure_insightface(models_root: Path) -> None:
    target_dir = models_root / "insightface" / "models" / "buffalo_l"
    marker = target_dir / "det_10g.onnx"
    if marker.exists():
        print(f"[skip] insightface buffalo_l (already at {target_dir})")
        return
    print("[get]  insightface buffalo_l")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmpzip:
        zip_path = Path(tmpzip.name)
    try:
        _download(INSIGHTFACE_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
    finally:
        zip_path.unlink(missing_ok=True)


def _ensure_hf(models_root: Path) -> None:
    target = models_root / "hf"
    target.mkdir(parents=True, exist_ok=True)
    for repo in HF_REPOS:
        print(f"[hf]   {repo}")
        snapshot_download(repo_id=repo, cache_dir=str(target))


def main() -> None:
    models_root = Path(os.environ.get("RAWCURATOR_MODELS", "/data/models"))
    models_root.mkdir(parents=True, exist_ok=True)
    print(f"Target: {models_root}")
    # Each step is independent; don't let one failure block the rest.
    errors: list[str] = []
    for label, fn in (
        ("direct files", lambda: _ensure_direct(models_root)),
        ("insightface", lambda: _ensure_insightface(models_root)),
        ("hugging face", lambda: _ensure_hf(models_root)),
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            print(f"[fail] {label}: {exc}")
    if errors:
        print("Completed with errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All model weights ready.")


if __name__ == "__main__":
    main()
