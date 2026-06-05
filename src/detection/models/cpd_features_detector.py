from typing import Optional

import numpy as np

from src.detection.models.cpd_base import BaseCPDDetector
from src.detection.utils.features import build_cpd_feature_matrix


class CPDFeaturesDetector(BaseCPDDetector):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("method_name", "cpd_features")
        super().__init__(*args, **kwargs)

    def prepare_input(self, signal: np.ndarray, period: Optional[float] = None) -> np.ndarray:
        return build_cpd_feature_matrix(signal, self.dt, period=period)
