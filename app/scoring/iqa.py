"""MUSIQ + MANIQA technical IQA via pyiqa."""

from __future__ import annotations

import numpy as np
import pyiqa
import torch
from PIL import Image


class _IqaBase:
    metric_name: str = ""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = pyiqa.create_metric(self.metric_name, device=self.device, as_loss=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def score(self, img: Image.Image) -> float:
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return float(self.model(t).item())


class MusiqScorer(_IqaBase):
    metric_name = "musiq"


class ManiqaScorer(_IqaBase):
    metric_name = "maniqa"


def normalize(scores: list[float], lo: float, hi: float) -> list[float]:
    span = max(hi - lo, 1e-6)
    return [max(0.0, min(1.0, (s - lo) / span)) for s in scores]
