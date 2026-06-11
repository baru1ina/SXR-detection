from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


def _make_model(backend: str = "auto", random_state: int = 42):
    backend = backend.lower()
    if backend in {"auto", "catboost"}:
        try:
            from catboost import CatBoostClassifier

            return "catboost", CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=random_state,
                auto_class_weights="Balanced",
            )
        except Exception:
            if backend == "catboost":
                raise

    if backend in {"auto", "xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier

            return "xgboost", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
            )
        except Exception:
            if backend in {"xgboost", "xgb"}:
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    return "sklearn_hgb", HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=random_state,
        class_weight="balanced",
    )


class FeatureMLTrainer:
    """Train a candidate classifier from known/pseudo crash indices."""

    def __init__(self, dt: float, period: Optional[float] = None, backend: str = "auto"):
        self.dt = float(dt)
        self.period = period
        self.backend_requested = backend
        self.backend: Optional[str] = None
        self.model = None

    def build_dataset_from_indices(
        self,
        signal: np.ndarray,
        crash_indices: Sequence[int],
        negative_indices: Optional[Sequence[int]] = None,
        n_negative_per_positive: int = 4,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        n = len(signal)
        positives = np.array(sorted(set(map(int, crash_indices))), dtype=int)
        positives = positives[(positives > 3) & (positives < n - 3)]

        if negative_indices is None:
            if self.period is None or not np.isfinite(self.period) or self.period <= 0:
                guard = max(10, int(0.5e-3 / self.dt))
            else:
                guard = max(10, int(0.35 * self.period / self.dt))
            possible = np.arange(guard, max(guard + 1, n - guard))
            for p in positives:
                possible = possible[np.abs(possible - p) > guard]
            k = min(len(possible), max(len(positives) * n_negative_per_positive, 1))
            negatives = rng.choice(possible, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        else:
            negatives = np.array(sorted(set(map(int, negative_indices))), dtype=int)

        xs = []
        ys = []
        for idx in positives:
            xs.append(extract_candidate_features(signal, int(idx), self.dt, self.period))
            ys.append(1)
        for idx in negatives:
            xs.append(extract_candidate_features(signal, int(idx), self.dt, self.period))
            ys.append(0)

        if not xs:
            raise ValueError("No training samples were generated")
        return np.vstack(xs), np.asarray(ys, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.backend, self.model = _make_model(self.backend_requested)
        self.model.fit(X, y)
        return self

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": FeatureMLMetadata(
                    backend=self.backend or self.backend_requested,
                    dt=self.dt,
                    period=self.period,
                    feature_names=FEATURE_NAMES,
                ),
            },
            path,
        )


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
        probability_threshold: float = 0.5,
        min_distance_factor: float = 0.45,
    ):
        self.dt = float(dt)
        self.model_path = Path(model_path)
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.metadata = payload.get("metadata")
        self.proposal_detector = proposal_detector or WaveletPOSRDetector(dt=dt)
        self.probability_threshold = float(probability_threshold)
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
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> list[CrashCandidate]:
        if hasattr(self.proposal_detector, "detect_candidates"):
            proposal = self.proposal_detector.detect_candidates(signal, period=period, channel=channel, debug=debug)
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
        kept_idx = suppress_by_period([c.index for c in accepted], [c.score for c in accepted], min_distance)
        kept = set(map(int, kept_idx))
        result = [c for c in accepted if c.index in kept]
        result.sort(key=lambda c: c.index)
        self.last_candidates = result

        if debug:
            print(f"[FeatureML] proposal={len(proposal)}, accepted={len(result)}")
        return result

    def detect(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> np.ndarray:
        return candidates_to_indices(
            self.detect_candidates(signal, period=period, channel=channel, debug=debug)
        )
