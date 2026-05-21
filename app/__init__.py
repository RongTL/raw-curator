"""Raw Curator — AI RAW photo curation pipeline (ephemeral single-batch).

Side effects on import:

1. Installs a compatibility shim so basicsr / realesrgan / codeformer-pip can
   still ``from torchvision.transforms.functional_tensor import ...`` on
   torchvision >= 0.17 (where that submodule was removed).
2. Symlinks CodeFormer's RetinaFace + parsing weights from the persistent
   ``/data/models/CodeFormer/weights/facelib/`` mount into the codeformer-pip
   package's hardcoded download path, so ``make enhance`` doesn't re-download
   them on every fresh container.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

__version__ = "0.1.0"

_CODEFORMER_AUX_WEIGHTS = (
    "detection_Resnet50_Final.pth",
    "parsing_parsenet.pth",
)


def _install_torchvision_compat_shim() -> None:
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    try:
        import torchvision.transforms.functional as F  # type: ignore
    except Exception:
        return
    shim = types.ModuleType(name)
    for attr in (
        "rgb_to_grayscale",
        "to_tensor",
        "normalize",
        "resize",
        "crop",
        "center_crop",
        "pad",
        "hflip",
        "vflip",
    ):
        fn = getattr(F, attr, None)
        if fn is not None:
            setattr(shim, attr, fn)
    sys.modules[name] = shim


def _link_codeformer_aux_weights() -> None:
    models_root = Path(os.environ.get("RAWCURATOR_MODELS", "/data/models"))
    source_dir = models_root / "CodeFormer" / "weights" / "facelib"
    if not source_dir.is_dir():
        return
    try:
        import codeformer  # type: ignore
    except Exception:
        return
    target_dir = Path(codeformer.__file__).parent / "weights" / "facelib"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for name in _CODEFORMER_AUX_WEIGHTS:
        src = source_dir / name
        if not src.exists():
            continue
        dst = target_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            pass


_install_torchvision_compat_shim()
_link_codeformer_aux_weights()
