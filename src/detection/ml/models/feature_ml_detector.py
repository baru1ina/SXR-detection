from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from config.channels import get_feature_map_profile
from src.detection.ml.multichannel_features import (
    FeatureSchema,
    MultichannelFeatureBuilder,
)
from src.detection.ml.models.hybrid_proposal_detector import FeatureMLProposalDetector
from src.detection.ml.multisxr_proposals import collect_multisxr_candidates
from src.detection.models.cpd_detector import CPDDetector
from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.features import score_crash_candidate, suppress_by_period


MODEL_SCHEMA_VERSION = 7


def load_feature_ml_payload(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    try:
        payload = joblib.load(model_path)
    except AttributeError as exc:
        if "FeatureMLMetadata" not in str(exc):
            raise
        raise ValueError(
            f"Model {model_path} is an old single-channel feature-ML "
            "artifact; retrain the current multichannel model"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Model {model_path} is not a current feature-ML artifact; "
            "retrain the multichannel model before detection"
        )
    return payload


class FeatureMLCrashDetector:
    """Classify POSR/CPD proposals using the full prepared diagnostic set."""

    def __init__(
        self,
        dt: float,
        downsample: int,
        model_path: str | Path,
        proposal_detector: Optional[object] = None,
        proposal_detector_factory: Optional[object] = None,
        probability_threshold: Optional[float] = None,
        feature_builder: Optional[MultichannelFeatureBuilder] = None,
        payload: Optional[dict] = None,
    ):
        self.dt = float(dt)
        self.downsample = int(downsample)
        if self.dt <= 0 or not np.isfinite(self.dt):
            raise ValueError("Detector dt must be positive and finite")
        if self.downsample < 1:
            raise ValueError("downsample must be at least 1")

        self.model_path = Path(model_path)
        payload = payload or load_feature_ml_payload(self.model_path)

        saved_profile = get_feature_map_profile(payload["feature_map_profile"])
        self.feature_builder = feature_builder or MultichannelFeatureBuilder(saved_profile)
        saved_schema = FeatureSchema.from_dict(payload["feature_schema"])
        if saved_schema != self.feature_builder.schema:
            raise ValueError("Model feature schema does not match the current builder")
        expected_channels = [asdict(channel) for channel in saved_profile.channels]
        if payload.get("diagnostic_channels") != expected_channels:
            raise ValueError("Model diagnostic channel profile does not match the current profile")
        if payload.get("downsample") != self.downsample:
            raise ValueError(
                f"Model downsample={payload.get('downsample')} does not match "
                f"runtime downsample={self.downsample}"
            )
        if not np.isclose(float(payload["effective_dt_s"]), self.dt, rtol=1e-4, atol=1e-12):
            raise ValueError("Model effective dt does not match runtime dt")
        self.model = payload["model"]
        self.proposal_config = dict(payload["proposal_config"])
        if proposal_detector_factory is not None:
            self.proposal_detector_factory = proposal_detector_factory
        elif proposal_detector is not None:
            self.proposal_detector_factory = lambda _dt, _period: proposal_detector
        else:
            self.proposal_detector_factory = self._saved_proposal_detector_factory
        saved_threshold = float(payload["selected_threshold"])
        self.probability_threshold = float(
            saved_threshold if probability_threshold is None else probability_threshold
        )
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be between 0 and 1")
        self.min_distance_factor = float(self.proposal_config["min_distance_factor"])
        self.last_candidates: list[CrashCandidate] = []

    def _saved_proposal_detector_factory(self, dt: float, period: float):
        sigma = (
            80e-6
            if not np.isfinite(period) or period <= 0
            else float(np.clip(0.03 * period, 30e-6, 200e-6))
        )
        return FeatureMLProposalDetector(
            source=self.proposal_config["proposal_source"],
            posr_detector=WaveletPOSRDetector(
                dt=dt,
                sigma=sigma,
                threshold=float(self.proposal_config["threshold"]),
                min_distance_factor=self.min_distance_factor,
            ),
            cpd_detector=CPDDetector(
                dt=dt,
                penalty=float(self.proposal_config["cpd_penalty"]),
                model=str(self.proposal_config["cpd_model"]),
            ),
        )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X), dtype=float)[:, 1]
        return np.asarray(self.model.predict(X), dtype=float)

    def candidate_feature_matrix(
        self,
        prepared,
        proposals: list[CrashCandidate],
        period: Optional[float] = None,
    ) -> np.ndarray:
        standardized = self.feature_builder.prepare_standardized_channels(prepared)
        vectors = []
        for candidate in proposals:
            plasma_index = int(candidate.index) * self.downsample
            if not 0 <= plasma_index < len(prepared.time_plasma):
                raise IndexError(f"Proposal index {candidate.index} is outside the plasma grid")
            vectors.append(
                self.feature_builder.build(
                    prepared,
                    plasma_index,
                    period=period,
                    standardized_channels=standardized,
                    proposal_evidence=candidate.meta,
                )
            )
        if not vectors:
            return np.empty((0, self.feature_builder.schema.n_features), dtype=float)
        return np.vstack(vectors)

    def detect_candidates(
        self,
        prepared,
        channel: Optional[str] = None,
        debug: bool = False,
        min_channels: int = 1,
    ) -> list[CrashCandidate]:
        signal = np.asarray(prepared.reference_signal, dtype=float)[::self.downsample]
        period = float(prepared.estimated_period)
        channel = channel or prepared.channel_name

        proposals = collect_multisxr_candidates(
            prepared,
            downsample=self.downsample,
            channels=self.proposal_config["proposal_channels"],
            coincidence_window_s=self.proposal_config["coincidence_window_s"],
            detector_factory=self.proposal_detector_factory,
            debug=debug,
        )
        if not proposals:
            self.last_candidates = []
            return []

        X = self.candidate_feature_matrix(prepared, proposals, period=period)
        probabilities = self._predict_proba(X)
        accepted: list[CrashCandidate] = []
        for candidate, probability in zip(proposals, probabilities):
            if probability < self.probability_threshold:
                continue
            score, direction, metadata = score_crash_candidate(
                signal,
                int(candidate.index),
                self.dt,
                period,
            )
            metadata = dict(metadata)
            metadata.update(candidate.meta)
            metadata["ml_probability"] = float(probability)
            accepted.append(
                CrashCandidate(
                    index=int(candidate.index),
                    time=float(prepared.time_plasma[int(candidate.index) * self.downsample]),
                    score=float(probability + 0.1 * score),
                    direction=direction,
                    channel=channel,
                    method="feature_ml",
                    meta=metadata,
                )
            )

        min_distance = max(1, int(self.min_distance_factor * period / self.dt))
        best_by_index: dict[int, CrashCandidate] = {}
        for candidate in accepted:
            if len(candidate.meta.get("source_channels", [])) < int(min_channels):
                continue
            previous = best_by_index.get(candidate.index)
            if previous is None or candidate.score > previous.score:
                best_by_index[candidate.index] = candidate
        kept_indices = suppress_by_period(
            list(best_by_index),
            [best_by_index[index].score for index in best_by_index],
            min_distance,
        )
        self.last_candidates = [best_by_index[int(index)] for index in kept_indices]
        if debug:
            print(
                f"[FeatureML] proposal={len(proposals)}, "
                f"accepted={len(self.last_candidates)}, "
                f"probability_threshold={self.probability_threshold:.6f}"
            )
        return self.last_candidates

    def detect(
        self,
        prepared,
        channel: Optional[str] = None,
        debug: bool = False,
        min_channels: int = 1,
    ) -> np.ndarray:
        return candidates_to_indices(
            self.detect_candidates(
                prepared,
                channel=channel,
                debug=debug,
                min_channels=min_channels,
            )
        )
