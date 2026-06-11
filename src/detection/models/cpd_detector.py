from typing import Optional

import numpy as np

from src.detection.models.cpd_base import BaseCPDDetector
from src.detection.utils.features import detrend_savgol, robust_zscore, safe_standardize_2d, smooth_signal


class CPDDetector(BaseCPDDetector):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_name", "cpd")
        super().__init__(*args, **kwargs)

    def prepare_input(self, signal: np.ndarray, period: Optional[float] = None) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        x_smooth = smooth_signal(x, median_window=31, gaussian_sigma=1.0)
        x_detrended = detrend_savgol(x_smooth, self.dt)
        return safe_standardize_2d(robust_zscore(x_detrended).reshape(-1, 1))
