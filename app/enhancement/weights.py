"""Where each AI model's weights live under ``settings.models``.

Single source of truth for the layout that ``scripts/download_models.py``
produces, so the enhancement steps and the downloader can't drift apart.
"""

from __future__ import annotations

from pathlib import Path

from app.config import settings

SCUNET_FILE = "scunet_color_real_psnr.pth"
REALESRGAN_FILE = "RealESRGAN_x2plus.pth"
CODEFORMER_FILE = "CodeFormer/weights/CodeFormer/codeformer.pth"
CODEFORMER_FACELIB_DIR = "CodeFormer/weights/facelib"


def scunet_weights() -> Path:
    return settings.models / SCUNET_FILE


def realesrgan_weights() -> Path:
    return settings.models / REALESRGAN_FILE


def codeformer_weights() -> Path:
    return settings.models / CODEFORMER_FILE


def codeformer_facelib_dir() -> Path:
    return settings.models / CODEFORMER_FACELIB_DIR
