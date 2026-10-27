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
from src.detection.ml.postprocessing import (
    adaptive_period_nms,
    fixed_gap_merge,
    statistical_gap_merge,
)
from src.detection.models.cpd_detector import CPDDetector
from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.features import score_crash_candidate


MODEL_SCHEMA_VERSION = 1


def load_feature_ml_payload(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    payload = joblib.load(model_path)
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
        event_merge_window_s: Optional[float] = None,
        event_merge_strategy: str = "adaptive_period",
        event_merge_period_fraction: float = 0.50,
        event_merge_min_radius_s: float = 0.12e-3,
        event_merge_max_radius_s: float = 0.80e-3,
        event_merge_confidence_threshold: float = 0.20,
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
        saved_event_merge_window_s = self.proposal_config["event_merge_window_s"]
        self.event_merge_window_s = float(
            saved_event_merge_window_s
            if event_merge_window_s is None
            else event_merge_window_s
        )
        if not np.isfinite(self.event_merge_window_s) or self.event_merge_window_s <= 0:
            raise ValueError("Model event_merge_window_s must be positive and finite")
        self.event_merge_strategy = str(event_merge_strategy).strip().casefold()
        if self.event_merge_strategy not in {"adaptive_period", "fixed", "statistical"}:
            raise ValueError(
                "event_merge_strategy must be adaptive_period, fixed, or statistical"
            )
        self.event_merge_period_fraction = float(event_merge_period_fraction)
        self.event_merge_min_radius_s = float(event_merge_min_radius_s)
        self.event_merge_max_radius_s = float(event_merge_max_radius_s)
        self.event_merge_confidence_threshold = float(
            event_merge_confidence_threshold
        )
        if not np.isfinite(self.event_merge_period_fraction) or self.event_merge_period_fraction <= 0:
            raise ValueError("event_merge_period_fraction must be positive and finite")
        if (
            not np.isfinite(self.event_merge_min_radius_s)
            or not np.isfinite(self.event_merge_max_radius_s)
            or self.event_merge_min_radius_s <= 0
            or self.event_merge_max_radius_s < self.event_merge_min_radius_s
        ):
            raise ValueError("event merge radii must satisfy 0 < min <= max")
        if not 0 <= self.event_merge_confidence_threshold <= 1:
            raise ValueError("event_merge_confidence_threshold must be in [0, 1]")
        self.last_candidates: list[CrashCandidate] = []
        self.last_accepted_candidates: list[CrashCandidate] = []

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
            self.last_accepted_candidates = []
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
            metadata["physics_score"] = float(score)
            accepted.append(
                CrashCandidate(
                    index=int(candidate.index),
                    time=float(prepared.time_plasma[int(candidate.index) * self.downsample]),
                    score=float(probability),
                    direction=direction,
                    channel=channel,
                    method="feature_ml",
                    meta=metadata,
                )
            )

        best_by_index: dict[int, CrashCandidate] = {}
        for candidate in accepted:
            if len(candidate.meta.get("source_channels", [])) < int(min_channels):
                continue
            previous = best_by_index.get(candidate.index)
            if previous is None or candidate.score > previous.score:
                best_by_index[candidate.index] = candidate
        self.last_accepted_candidates = [
            best_by_index[index] for index in sorted(best_by_index)
        ]
        if self.event_merge_strategy == "adaptive_period":
            self.last_candidates = adaptive_period_nms(
                self.last_accepted_candidates,
                dt=self.dt,
                period_map=getattr(prepared, "merge_period_map", None),
                confidence_map=getattr(prepared, "period_confidence_map", None),
                index_scale=self.downsample,
                period_fraction=self.event_merge_period_fraction,
                min_radius_s=self.event_merge_min_radius_s,
                max_radius_s=self.event_merge_max_radius_s,
                fallback_radius_s=self.event_merge_window_s,
                confidence_threshold=self.event_merge_confidence_threshold,
            )
        elif self.event_merge_strategy == "statistical":
            self.last_candidates = statistical_gap_merge(
                self.last_accepted_candidates,
                dt=self.dt,
            )
        else:
            self.last_candidates = fixed_gap_merge(
                self.last_accepted_candidates,
                dt=self.dt,
                gap_s=self.event_merge_window_s,
            )
        if debug:
            local_count = sum(
                candidate.meta.get("event_merge_radius_source") == "local_period"
                for candidate in self.last_candidates
            )
            print(
                f"[FeatureML] proposal={len(proposals)}, "
                f"above_threshold={len(accepted)}, "
                f"merged={len(self.last_candidates)}, "
                f"merge_strategy={self.event_merge_strategy}, "
                f"local_period_events={local_count}, "
                f"fallback_gap_ms={self.event_merge_window_s * 1e3:.3f}, "
                f"probability_threshold={self.probability_threshold:.6f}"
            )
            for candidate in self.last_candidates:
                print(
                    "[FeatureML:event] "
                    f"time={candidate.time:.6f}, "
                    f"p={candidate.meta.get('ml_probability', candidate.score):.4f}, "
                    f"period_s={candidate.meta.get('event_merge_period_s')}, "
                    f"confidence={candidate.meta.get('event_merge_period_confidence')}, "
                    f"radius_ms={candidate.meta.get('event_merge_radius_s', 0.0) * 1e3:.3f}, "
                    f"source={candidate.meta.get('event_merge_radius_source', 'n/a')}"
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
