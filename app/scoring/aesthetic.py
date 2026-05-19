"""Aesthetic Predictor v2.5 — SigLIP-based scorer (lazy import)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from PIL import Image


class AestheticPredictor:
    def __init__(self, device: str = "cuda", precision: str = "fp16") -> None:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip  # type: ignore

        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        model, processor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval().to(self.device).to(self.dtype)
        self.model = model
        self.processor = processor

    def __enter__(self) -> "AestheticPredictor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def score_batch(self, images: Iterable[Image.Image]) -> list[float]:
        pixel_values = (
            self.processor(images=list(images), return_tensors="pt")
            .pixel_values.to(self.device, dtype=self.dtype)
        )
        out = self.model(pixel_values).logits.squeeze(-1)
        return out.float().detach().cpu().tolist()
