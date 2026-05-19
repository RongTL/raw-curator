"""Variance of Laplacian — fast blur estimator."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(rgb: np.ndarray) -> float:
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
