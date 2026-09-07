from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.utils.features import extract_candidate_features, score_crash_candidate, suppress_by_period


@dataclass
class FeatureMLMetadata:
    backend: str
    dt: float
    period: Optional[float]
    feature_names: list[str]


FEATURE_NAMES = [
    "physics_score",
    "amp_score",
    "slope_score",
    "normal_drop",
    "inverted_rise",
    "seg_mean",
    "seg_std",
    "seg_mad",
    "seg_peak_to_peak",
    "min_derivative",
    "max_derivative",
    "std_derivative",
    "kurtosis",
    "skew",
    "post_minus_pre_mean",
    "post_std_over_pre_std",
]


class FeatureMLCrashDetector:
    """Candidate-based ML detector.

    It uses a proposal detector first (by default DoG/POSR), then classifies the
    proposed points with CatBoost/XGBoost/sklearn model.  This keeps ML tractable
    when you have few labeled crashes.
    """

    def __init__(
        self,
        dt: float,
        model_path: str | Path,
        proposal_detector: Optional[object] = None,
        probability_threshold: Optional[float] = None,
        min_distance_factor: float = 0.45,
    ):
        self.dt = float(dt)
        self.model_path = Path(model_path)
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.metadata = payload.get("metadata")
        self.training_metadata = payload.get("training", {})
        self.proposal_detector = proposal_detector or WaveletPOSRDetector(dt=dt)
        saved_threshold = self.training_metadata.get("selected_threshold", 0.5)
        self.probability_threshold = float(
            saved_threshold if probability_threshold is None else probability_threshold
        )
        self.min_distance_factor = float(min_distance_factor)
        self.last_candidates: list[CrashCandidate] = []

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))[:, 1]
        pred = np.asarray(self.model.predict(X))
        return pred.astype(float)

    def detect_candidates(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> list[CrashCandidate]:
        if hasattr(self.proposal_detector, "detect_candidates"):
            proposal = self.proposal_detector.detect_candidates(
                signal,
                period=period,
                period_map=period_map,
                active_mask=active_mask,
                channel=channel,
                debug=debug,
            )
        else:
            idx = self.proposal_detector.detect(signal, period=period, debug=debug)
            proposal = [CrashCandidate(index=int(i), time=int(i) * self.dt, channel=channel) for i in idx]

        if not proposal:
            self.last_candidates = []
            return []

        X = np.vstack([extract_candidate_features(signal, c.index, self.dt, period) for c in proposal])
        proba = self._predict_proba(X)

        accepted: list[CrashCandidate] = []
        for c, p in zip(proposal, proba):
            if p < self.probability_threshold:
                continue
            score, direction, meta = score_crash_candidate(signal, c.index, self.dt, period)
            meta = dict(meta)
            meta["ml_probability"] = float(p)
            accepted.append(
                CrashCandidate(
                    index=c.index,
                    time=c.index * self.dt,
                    score=float(p + 0.1 * score),
                    direction=direction,
                    channel=channel,
                    method="feature_ml",
                    meta=meta,
                )
            )

        if period is None or not np.isfinite(period) or period <= 0:
            min_distance = max(1, int(0.25e-3 / self.dt))
        else:
            min_distance = max(1, int(self.min_distance_factor * period / self.dt))
        best_by_index: dict[int, CrashCandidate] = {}
        for candidate in accepted:
            previous = best_by_index.get(candidate.index)
            if previous is None or candidate.score > previous.score:
                best_by_index[candidate.index] = candidate
        unique_accepted = list(best_by_index.values())
        kept_idx = suppress_by_period(
            [c.index for c in unique_accepted],
            [c.score for c in unique_accepted],
            min_distance,
        )
        result = [best_by_index[int(index)] for index in kept_idx]
        result.sort(key=lambda c: c.index)
        self.last_candidates = result

        if debug:
            print(
                f"[FeatureML] proposal={len(proposal)}, accepted={len(result)}, "
                f"probability_threshold={self.probability_threshold:.6f}"
            )
        return result

    def detect(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> np.ndarray:
        return candidates_to_indices(
            self.detect_candidates(
                signal,
                period=period,
                period_map=period_map,
                active_mask=active_mask,
                channel=channel,
                debug=debug,
            )
        )
