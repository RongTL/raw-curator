"""Raw Curator — AI RAW photo curation pipeline (ephemeral single-batch).

Side effect on import: installs a compatibility shim so basicsr / realesrgan /
codeformer-pip can still ``from torchvision.transforms.functional_tensor import ...``
on torchvision >= 0.17 (where that submodule was removed).
"""

from __future__ import annotations

import sys
import types

__version__ = "0.1.0"


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


_install_torchvision_compat_shim()
