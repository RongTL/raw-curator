"""Array type alias for image data.

Image arrays cross the uint8 / float32 boundary constantly (AI steps take
uint8, classical steps take float32 in [0, 1]), so the alias is deliberately
dtype-agnostic; the dtype contract is documented per function instead.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

Array = npt.NDArray[Any]
