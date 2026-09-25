from typing import Optional

import numpy as np

from src.detection.utils.common import CrashCandidate


PROPOSAL_SOURCES = ("posr", "cpd", "hybrid")


class FeatureMLProposalDetector:
    """Select raw POSR, raw CPD, or their union for feature-ML."""

    def __init__(self, source, posr_detector, cpd_detector):
        source = str(source).lower()
        if source not in PROPOSAL_SOURCES:
            raise ValueError(
                f"Unsupported proposal source {source!r}; expected one of {PROPOSAL_SOURCES}"
            )
        self.source = source
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
        posr = (
            list(self.posr_detector.detect_raw_candidates(signal, **common))
            if self.source in {"posr", "hybrid"}
            else []
        )
        cpd = (
            list(self.cpd_detector.detect_raw_candidates(signal, **common))
            if self.source in {"cpd", "hybrid"}
            else []
        )
        self.last_source_counts = {"posr": len(posr), "cpd_raw": len(cpd)}
        self.last_candidates = posr + cpd

        if debug:
            print(
                f"[FeatureMLProposal] source={self.source}, "
                f"posr_raw={len(posr)}, cpd_raw={len(cpd)}, "
                f"total={len(self.last_candidates)}, pre_ml_nms=disabled"
            )
        return self.last_candidates
