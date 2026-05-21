"""CodeFormer face restoration. Falls back to identity if weights/imports fail."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

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
_MAX_LONG_EDGE = int(os.environ.get("RAWCURATOR_CF_MAX_LONG_EDGE", "2048"))


def codeformer_restore(rgb: np.ndarray, faces, weight: float = 0.7) -> np.ndarray:
    weights = Path("/data/models/CodeFormer/weights/CodeFormer/codeformer.pth")
    if not weights.exists():
        log.info("codeformer weights missing at %s — skipping face restore", weights)
        return rgb
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
        from codeformer.basicsr.archs.codeformer_arch import CodeFormer  # type: ignore
        from codeformer.basicsr.utils import img2tensor, tensor2img  # type: ignore
        from codeformer.facelib.utils.face_restoration_helper import (  # type: ignore
            FaceRestoreHelper,
        )
        from torchvision.transforms.functional import normalize  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.warning("codeformer imports failed: %s — skipping face restore", exc)
        return rgb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h0, w0 = rgb.shape[:2]
    long_edge = max(h0, w0)
    if long_edge > _MAX_LONG_EDGE:
        s = _MAX_LONG_EDGE / long_edge
        rgb_in = cv2.resize(
            rgb,
            (int(round(w0 * s)), int(round(h0 * s))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        rgb_in = rgb

    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt.get("params_ema", ckpt))
    net.eval()

    helper = FaceRestoreHelper(
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
        del net, helper
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rgb

    for cropped_face in helper.cropped_faces:
        face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_t = face_t.unsqueeze(0).to(device)
        with torch.no_grad():
            output = net(face_t, w=weight, adain=True)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype(np.uint8)
        helper.add_restored_face(restored)
        del face_t, output

    helper.get_inverse_affine(None)
    restored_bgr = helper.paste_faces_to_input_image(upsample_img=None)
    del net, helper
    if device.type == "cuda":
        torch.cuda.empty_cache()

    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    if restored_rgb.shape[:2] != (h0, w0):
        restored_rgb = cv2.resize(
            restored_rgb, (w0, h0), interpolation=cv2.INTER_LANCZOS4
        )
    return restored_rgb
