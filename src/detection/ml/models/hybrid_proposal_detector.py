from typing import Optional

import numpy as np

from src.detection.utils.common import CrashCandidate


class HybridProposalDetector:
    """Combine POSR proposals with unfiltered CPD breakpoints.

    No candidates are merged or suppressed here.  The feature classifier sees
    every proposal; suppression is applied only to its accepted predictions.
    """

    def __init__(self, posr_detector, cpd_detector):
        self.posr_detector = posr_detector
        self.cpd_detector = cpd_detector
        self.last_candidates: list[CrashCandidate] = []
        self.last_source_counts: dict[str, int] = {}

    def detect_candidates(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> list[CrashCandidate]:
        common = {
            "period": period,
            "period_map": period_map,
            "active_mask": active_mask,
            "channel": channel,
            "debug": debug,
        }
        posr = list(self.posr_detector.detect_candidates(signal, **common))
        cpd = list(self.cpd_detector.detect_raw_candidates(signal, **common))
        self.last_source_counts = {"posr": len(posr), "cpd_raw": len(cpd)}
        self.last_candidates = posr + cpd

        if debug:
            print(
                f"[HybridProposal] posr={len(posr)}, cpd_raw={len(cpd)}, "
                f"total={len(self.last_candidates)}, pre_ml_nms=disabled"
            )
        return self.last_candidates
