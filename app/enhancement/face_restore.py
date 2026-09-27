"""CodeFormer face restoration. Falls back to identity if weights/imports fail."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from app.arrays import Array
from app.config import settings
from app.enhancement.geometry import Box
from app.enhancement.weights import codeformer_weights

log = logging.getLogger(__name__)


def guard_faces(
    before: Array,
    after: Array,
    boxes: Sequence[Box],
    *,
    embed: Callable[[Array], Array],
    min_similarity: float,
) -> Array:
    """Revert restored faces whose identity drifted, comparing before vs after.

    Both crops pass through the identical ``embed`` transform, so the cosine
    measures only what restoration changed. For each box (clamped to the image;
    empty crops skipped) the original pixels are pasted back into a copy of
    ``after`` when ``cos(embed(after_crop), embed(before_crop)) < min_similarity``.
    ``before``/``after`` are uint8 RGB of the same shape; ``after`` is not mutated.
    """
    out = after.copy()
    h, w = after.shape[:2]
    for x, y, bw, bh in boxes:
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + bw, w), min(y + bh, h)
        if x1 <= x0 or y1 <= y0:
            continue
        ref = embed(before[y0:y1, x0:x1]).astype(np.float32)
        v = embed(after[y0:y1, x0:x1]).astype(np.float32)
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-8))
        if cos < min_similarity:
            log.warning(
                "identity drift on face at (%d,%d): cos=%.2f < %.2f; reverting",
                x0,
                y0,
                cos,
                min_similarity,
            )
            out[y0:y1, x0:x1] = before[y0:y1, x0:x1]
    return out


# RetinaFace (the detector bundled in FaceRestoreHelper) runs ResNet50 on
# the FULL input frame. A Real-ESRGAN-upscaled 24 MP RAW lands around
# 8400x5600 at enhance_ai_scale=0.7, which OOMs a 6 GB card on the very
# first stride-2 conv (1x64x4200x2800x4 = ~3 GiB). helper's own
# `resize=640` argument only ever UP-scales tiny inputs (scale = max(1,
# scale) in the upstream code), so we cap the long edge ourselves before
# handing the image over. CodeFormer crops every detected face to 512 px
# internally and the result is upsampled back to native by
# upsample_final(), so capping only costs a Lanczos round-trip on the
# non-face areas, which the final resample to native re-flattens anyway.
# The cap is settings.codeformer_max_long_edge.


def _cap_long_edge(rgb: Array) -> Array:
    """Area-downsample so the long edge is <= settings.codeformer_max_long_edge.

    Returns the input object itself when no resize is needed.
    """
    import cv2

    max_edge = settings.codeformer_max_long_edge
    h0, w0 = rgb.shape[:2]
    long_edge = max(h0, w0)
    if long_edge <= max_edge:
        return rgb
    s = max_edge / long_edge
    return cv2.resize(
        rgb,
        (int(round(w0 * s)), int(round(h0 * s))),
        interpolation=cv2.INTER_AREA,
    )


class CodeFormerModel:
    """CodeFormer net loaded once for a batch.

    The ``CodeFormer`` net is loaded in ``__enter__`` and reused across images.
    ``FaceRestoreHelper`` is rebuilt per call inside ``apply`` because it holds
    per-image state (cropped faces, affine transforms).
    """

    def __init__(self) -> None:
        self.available = False
        self._net: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._cv2: Any = None
        self._img2tensor: Any = None
        self._tensor2img: Any = None
        self._helper_cls: Any = None
        self._normalize: Any = None
        self._arcface: Any = None
        self._arcface_failed = False

    def __enter__(self) -> CodeFormerModel:
        weights = codeformer_weights()
        if not weights.exists():
            log.info("codeformer weights missing at %s — face restore disabled", weights)
            return self
        try:
            import cv2
            import torch
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer
            from codeformer.basicsr.utils import img2tensor, tensor2img
            from codeformer.facelib.utils.face_restoration_helper import (
                FaceRestoreHelper,
            )
            from torchvision.transforms.functional import normalize
        except Exception as exc:  # noqa: BLE001
            log.warning("codeformer imports failed: %s — face restore disabled", exc)
            return self

        self._cv2 = cv2
        self._torch = torch
        self._img2tensor = img2tensor
        self._tensor2img = tensor2img
        self._helper_cls = FaceRestoreHelper
        self._normalize = normalize
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self._device)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt.get("params_ema", ckpt))
        net.eval()
        self._net = net
        self.available = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._net = None
        self._arcface = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self.available = False

    def _ensure_arcface(self) -> bool:
        """Lazily load the buffalo_l ArcFace recognition net for the identity guard.

        Returns True once the net is usable. On a missing ONNX file or any
        import/prepare failure it logs a warning once, disables the guard for
        the rest of this model's life, and returns False.
        """
        if self._arcface is not None:
            return True
        if self._arcface_failed:
            return False
        path = settings.models / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        if not path.exists():
            self._arcface_failed = True
            log.warning("ArcFace weights missing at %s — identity guard disabled", path)
            return False
        try:
            from insightface.model_zoo import get_model

            rec = get_model(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            rec.prepare(ctx_id=0)
        except Exception as exc:  # noqa: BLE001 - guard is optional; degrade gracefully
            self._arcface_failed = True
            log.warning("ArcFace load failed (%s) — identity guard disabled", exc)
            return False
        self._arcface = rec
        return True

    def _embed(self, crop_rgb: Array) -> Array:
        """L2-normalised 512-d ArcFace embedding of an 8-bit RGB face crop.

        ``get_feat`` resizes any crop to 112x112 itself and expects BGR.
        """
        bgr = self._cv2.cvtColor(crop_rgb, self._cv2.COLOR_RGB2BGR)
        feat = self._arcface.get_feat([bgr])[0].astype(np.float32)
        return feat / (float(np.linalg.norm(feat)) + 1e-8)

    def apply(
        self,
        rgb: Array,
        *,
        weight: float = 0.7,
        faces: Sequence[Box] = (),
        min_similarity: float = 0.5,
        **_: object,
    ) -> Array:
        if not self.available:
            return rgb
        cv2 = self._cv2
        torch = self._torch
        device = self._device

        h0, w0 = rgb.shape[:2]
        rgb_in = _cap_long_edge(rgb)

        helper = self._helper_cls(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
        helper.clean_all()
        bgr = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR)
        helper.read_image(bgr)
        helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        helper.align_warp_face()

        if not helper.cropped_faces:
            del helper
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return rgb

        for cropped_face in helper.cropped_faces:
            face_t = self._img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            self._normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(device)
            with torch.no_grad():
                output = self._net(face_t, w=weight, adain=True)[0]
                restored = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)
            helper.add_restored_face(restored)
            del face_t, output

        helper.get_inverse_affine(None)
        restored_bgr = helper.paste_faces_to_input_image(upsample_img=None)
        del helper
        if device.type == "cuda":
            torch.cuda.empty_cache()

        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        if restored_rgb.shape[:2] != (h0, w0):
            restored_rgb = cv2.resize(restored_rgb, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        if faces and self._ensure_arcface():
            restored_rgb = guard_faces(
                rgb, restored_rgb, faces, embed=self._embed, min_similarity=min_similarity
            )
        return restored_rgb


def codeformer_restore(rgb: Array, weight: float = 0.7) -> Array:
    with CodeFormerModel() as m:
        return m.apply(rgb, weight=weight)
