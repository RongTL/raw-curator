"""Variance of Laplacian — fast blur estimator."""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
