"""InsightFace buffalo_l: detection + 512-d ArcFace embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import settings


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray


class FaceDetector:
    def __init__(
        self,
        device_id: int = 0,
        det_size: int = 640,
        root: str | None = None,
    ) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Persist weights under the bind-mounted models dir so they survive
        # podman run --rm (default insightface root is ~/.insightface inside the container).
        if root is None:
            root = str(settings.models / "insightface")
        self.app = FaceAnalysis(name="buffalo_l", providers=providers, root=root)
        self.app.prepare(ctx_id=device_id, det_size=(det_size, det_size))

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        del self.app

    def detect(self, image_path: Path) -> list[DetectedFace]:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return []
        out: list[DetectedFace] = []
        for f in self.app.get(bgr):
            x1, y1, x2, y2 = f.bbox.astype(int)
            embed = f.normed_embedding.astype(np.float16)
            out.append(
                DetectedFace(
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    det_score=float(f.det_score),
                    embedding=embed,
                )
            )
        return out
