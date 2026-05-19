"""CLIP ViT-L/14 image embedder (open_clip, laion2b_s32b_b82k, 768-d, FP16)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

log = logging.getLogger(__name__)

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"
EMBED_DIM = 768


class ClipEmbedder:
    def __init__(self, device: str = "cuda", precision: str = "fp16") -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        model.eval().to(self.device).to(self.dtype)
        self.model = model
        self.preprocess = preprocess

    def __enter__(self) -> "ClipEmbedder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embed_batch(self, paths: Iterable[Path]) -> Iterator[tuple[Path, np.ndarray]]:
        batch_paths = list(paths)
        if not batch_paths:
            return
        imgs = torch.stack(
            [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(self.device, dtype=self.dtype)
        feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats_np = feats.detach().to(torch.float16).cpu().numpy()
        for path, vec in zip(batch_paths, feats_np, strict=True):
            yield path, vec


def vec_to_bytes(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec.astype(np.float16)).tobytes()


def bytes_to_vec(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float16).copy()
