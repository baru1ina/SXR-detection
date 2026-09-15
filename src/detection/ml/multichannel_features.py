from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from config.channels import (
    DiagnosticChannelProfile,
    MULTICHANNEL_FEATURE_PROFILE,
)
from src.detection.utils.features import (
    SINGLE_CHANNEL_FEATURE_NAMES,
    extract_candidate_features,
)


_EPS = 1e-12
_CHANNEL_FEATURE_NAMES = (
    "coverage",
    "median_delta_z",
    "signed_jump_z",
    "log_energy_ratio",
    "max_abs_period_slope",
    "change_offset_period",
)
_GROUP_FEATURE_NAMES = (
    "coverage",
    "median_delta_z",
    "max_abs_jump_z",
    "max_abs_period_slope",
)
_PROPOSAL_FEATURE_NAMES = (
    "source_channel_count",
    "method_count",
    "proposal_count",
    "median_log_source_score",
    "time_spread_period",
)


@dataclass(frozen=True)
class FeatureWindowConfig:
    period_factor: float = 0.25
    min_window_s: float = 0.2e-3
    max_window_s: float = 1.0e-3
    min_valid_samples: int = 3

    def __post_init__(self):
        if self.period_factor <= 0:
            raise ValueError("period_factor must be positive")
        if self.min_window_s <= 0:
            raise ValueError("min_window_s must be positive")
        if self.max_window_s < self.min_window_s:
            raise ValueError("max_window_s must not be less than min_window_s")
        if self.min_valid_samples < 2:
            raise ValueError("min_valid_samples must be at least 2")


@dataclass(frozen=True)
class FeatureSchema:
    schema_version: int
    profile_name: str
    feature_names: tuple[str, ...]
    window_config: FeatureWindowConfig

    def __post_init__(self):
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be unique")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "feature_names": list(self.feature_names),
            "window_config": asdict(self.window_config),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSchema":
        return cls(
            schema_version=int(payload["schema_version"]),
            profile_name=str(payload["profile_name"]),
            feature_names=tuple(payload["feature_names"]),
            window_config=FeatureWindowConfig(**payload["window_config"]),
        )


@dataclass(frozen=True)
class _ChannelFeatures:
    coverage: float
    median_delta_z: float
    signed_jump_z: float
    log_energy_ratio: float
    max_abs_period_slope: float
    change_offset_period: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.coverage,
                self.median_delta_z,
                self.signed_jump_z,
                self.log_energy_ratio,
                self.max_abs_period_slope,
                self.change_offset_period,
            ],
            dtype=float,
        )


def _finite_median(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else np.nan


def _finite_max_abs(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite))) if len(finite) else np.nan


def _robust_standardize(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    result = np.full(len(values), np.nan, dtype=float)
    if not np.any(valid):
        return result

    valid_values = values[valid]
    center = float(np.median(valid_values))
    scale = float(1.4826 * np.median(np.abs(valid_values - center)))
    if not np.isfinite(scale) or scale <= _EPS:
        scale = float(np.std(valid_values))
    if not np.isfinite(scale) or scale <= _EPS:
        scale = 1.0
    result[valid] = (valid_values - center) / scale
    return result


def _channel_features(
    values: np.ndarray,
    valid_mask: np.ndarray,
    index: int,
    radius: int,
    dt: float,
    period: float,
    min_valid_samples: int,
    standardized: Optional[np.ndarray] = None,
) -> _ChannelFeatures:
    start = max(0, index - radius)
    stop = min(len(values), index + radius + 1)
    pre_slice = slice(start, index)
    post_slice = slice(index, stop)
    around_slice = slice(start, stop)

    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    local_valid = valid_mask[around_slice]
    coverage = float(np.mean(local_valid)) if len(local_valid) else 0.0
    if standardized is None:
        standardized = _robust_standardize(values, valid_mask)

    pre = standardized[pre_slice]
    post = standardized[post_slice]
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if len(pre) < min_valid_samples or len(post) < min_valid_samples:
        return _ChannelFeatures(
            coverage=coverage,
            median_delta_z=np.nan,
            signed_jump_z=np.nan,
            log_energy_ratio=np.nan,
            max_abs_period_slope=np.nan,
            change_offset_period=np.nan,
        )

    median_delta = float(np.median(post) - np.median(pre))
    normal_drop = max(float(np.percentile(pre, 90) - np.percentile(post, 10)), 0.0)
    inverted_rise = max(float(np.percentile(post, 90) - np.percentile(pre, 10)), 0.0)
    signed_jump = -normal_drop if normal_drop >= inverted_rise else inverted_rise

    pre_energy = float(np.sqrt(np.mean(pre * pre)))
    post_energy = float(np.sqrt(np.mean(post * post)))
    log_energy_ratio = float(np.log((post_energy + _EPS) / (pre_energy + _EPS)))

    around = standardized[around_slice]
    adjacent_valid = np.isfinite(around[:-1]) & np.isfinite(around[1:])
    if np.any(adjacent_valid):
        differences = np.diff(around)
        valid_positions = np.flatnonzero(adjacent_valid)
        strongest_local = valid_positions[
            int(np.argmax(np.abs(differences[adjacent_valid])))
        ]
        strongest_difference = float(differences[strongest_local])
        max_abs_period_slope = abs(strongest_difference) * period / dt
        change_index = start + strongest_local + 1
        change_offset_period = (change_index - index) * dt / period
    else:
        max_abs_period_slope = np.nan
        change_offset_period = np.nan

    return _ChannelFeatures(
        coverage=coverage,
        median_delta_z=median_delta,
        signed_jump_z=signed_jump,
        log_energy_ratio=log_energy_ratio,
        max_abs_period_slope=float(max_abs_period_slope),
        change_offset_period=float(change_offset_period),
    )


class MultichannelFeatureBuilder:
    """Build one deterministic ML vector for a candidate on the plasma grid."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        profile: DiagnosticChannelProfile = MULTICHANNEL_FEATURE_PROFILE,
        window_config: Optional[FeatureWindowConfig] = None,
    ):
        self.profile = profile
        self.window_config = window_config or FeatureWindowConfig()
        self.schema = FeatureSchema(
            schema_version=self.SCHEMA_VERSION,
            profile_name=profile.name,
            feature_names=self._feature_names(),
            window_config=self.window_config,
        )

    def _feature_names(self) -> tuple[str, ...]:
        names = [f"sxr__{name}" for name in SINGLE_CHANNEL_FEATURE_NAMES]
        for channel in self.profile.channels:
            names.extend(
                f"channel__{channel.key}__{feature}"
                for feature in _CHANNEL_FEATURE_NAMES
            )
        for group in self.profile.groups:
            names.extend(
                f"group__{group}__{feature}"
                for feature in _GROUP_FEATURE_NAMES
            )
        names.extend(f"proposal__{name}" for name in _PROPOSAL_FEATURE_NAMES)
        return tuple(names)

    def build(
        self,
        prepared,
        candidate_index: int,
        period: Optional[float] = None,
        standardized_channels: Optional[dict[str, np.ndarray]] = None,
        proposal_evidence: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        if prepared.aligned_signals is None:
            raise ValueError("PreparedShot does not contain aligned diagnostic signals")
        if prepared.feature_profile != self.profile.name:
            raise ValueError(
                f"Prepared profile {prepared.feature_profile!r} does not match "
                f"feature schema profile {self.profile.name!r}"
            )

        reference = np.asarray(prepared.reference_signal, dtype=float)
        aligned = prepared.aligned_signals
        if aligned.time.shape != reference.shape:
            raise ValueError("Reference signal and aligned diagnostics length mismatch")

        index = int(candidate_index)
        if not 0 <= index < len(reference):
            raise IndexError(f"Candidate index {index} is outside the plasma signal")
        dt = float(prepared.dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("PreparedShot dt must be positive and finite")

        effective_period = float(period or prepared.estimated_period)
        if not np.isfinite(effective_period) or effective_period <= 0:
            raise ValueError("Sawtooth period must be positive and finite")
        window_s = float(
            np.clip(
                self.window_config.period_factor * effective_period,
                self.window_config.min_window_s,
                self.window_config.max_window_s,
            )
        )
        radius = max(
            self.window_config.min_valid_samples,
            int(round(window_s / dt)),
        )

        reference_features = extract_candidate_features(
            reference,
            index,
            dt,
            effective_period,
        )
        result = list(np.asarray(reference_features, dtype=float))
        features_by_channel: dict[str, _ChannelFeatures] = {}

        for diagnostic in self.profile.channels:
            channel = aligned.get_channel(diagnostic.name)
            features = _channel_features(
                channel.values,
                channel.valid_mask,
                index=index,
                radius=radius,
                dt=dt,
                period=effective_period,
                min_valid_samples=self.window_config.min_valid_samples,
                standardized=(
                    None
                    if standardized_channels is None
                    else standardized_channels[diagnostic.name]
                ),
            )
            features_by_channel[diagnostic.name] = features
            result.extend(features.as_array())

        for group in self.profile.groups:
            group_features = [
                features_by_channel[channel.name]
                for channel in self.profile.channels
                if channel.group == group
            ]
            result.extend(
                [
                    float(np.mean([features.coverage for features in group_features])),
                    _finite_median(
                        [features.median_delta_z for features in group_features]
                    ),
                    _finite_max_abs(
                        [features.signed_jump_z for features in group_features]
                    ),
                    _finite_max_abs(
                        [features.max_abs_period_slope for features in group_features]
                    ),
                ]
            )

        if proposal_evidence is None:
            result.extend([np.nan] * len(_PROPOSAL_FEATURE_NAMES))
        else:
            scores = np.asarray(proposal_evidence.get("source_scores", []), dtype=float)
            finite_scores = scores[np.isfinite(scores)]
            result.extend(
                [
                    float(len(proposal_evidence.get("source_channels", []))),
                    float(len(proposal_evidence.get("source_methods", []))),
                    float(proposal_evidence.get("proposal_count", len(scores))),
                    (
                        float(np.median(np.log1p(np.maximum(finite_scores, 0.0))))
                        if len(finite_scores)
                        else np.nan
                    ),
                    float(proposal_evidence.get("time_spread_s", np.nan))
                    / effective_period,
                ]
            )

        vector = np.asarray(result, dtype=float)
        vector[np.isinf(vector)] = np.nan
        if vector.shape != (self.schema.n_features,):
            raise RuntimeError(
                f"Feature schema expects {self.schema.n_features} values, "
                f"but builder produced {vector.shape}"
            )
        return vector

    def prepare_standardized_channels(self, prepared) -> dict[str, np.ndarray]:
        """Compute full-shot robust scales once before evaluating candidates."""

        if prepared.aligned_signals is None:
            raise ValueError("PreparedShot does not contain aligned diagnostic signals")
        return {
            diagnostic.name: _robust_standardize(
                prepared.aligned_signals.get_channel(diagnostic.name).values,
                prepared.aligned_signals.get_channel(diagnostic.name).valid_mask,
            )
            for diagnostic in self.profile.channels
        }

    def as_dict(self, vector: np.ndarray) -> dict[str, float]:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.schema.n_features,):
            raise ValueError(
                f"Expected a vector with {self.schema.n_features} features, "
                f"got {vector.shape}"
            )
        return dict(zip(self.schema.feature_names, vector.tolist()))
