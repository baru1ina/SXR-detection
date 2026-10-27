from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from config.path import SXR_CHANNELS
from config.channels import (
    CORE_DIAGNOSTICS_FEATURE_PROFILE,
    DiagnosticChannelProfile,
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
_D_ALPHA_FEATURE_NAMES = (
    "coverage",
    "positive_peak_z",
    "peak_prominence_z",
    "positive_area_z",
    "rise_slope_period",
    "peak_offset_period",
)
_EXTREMUM_FEATURE_NAMES = (
    "coverage",
    "signed_extremum_z",
    "extremum_prominence_z",
    "max_abs_period_slope",
    "derivative_reversal_period",
    "extremum_offset_period",
)
_GROUP_FEATURE_NAMES = (
    "coverage",
    "median_delta_z",
    "max_abs_jump_z",
    "max_abs_period_slope",
)
_D_ALPHA_GROUP_FEATURE_NAMES = (
    "coverage",
    "median_positive_peak_z",
    "max_peak_prominence_z",
    "peak_support_fraction",
)
_EXTREMUM_GROUP_FEATURE_NAMES = (
    "coverage",
    "median_abs_extremum_z",
    "max_extremum_prominence_z",
    "extremum_support_fraction",
)
_EXTREMUM_GROUPS = frozenset({"mhd", "fduk"})
_PROPOSAL_FEATURE_NAMES = (
    "source_channel_count",
    "method_count",
    "proposal_count",
    "median_log_source_score",
    "time_spread_period",
)
_SXR_FEATURE_NAMES = (
    "present",
    "proposed",
    "proposed_posr",
    "proposed_cpd",
    "median_delta_z",
    "signed_jump_z",
    "log_energy_ratio",
    "max_abs_period_slope",
    "change_offset_period",
    "local_noise_z",
    "log_jump_to_noise",
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


@dataclass(frozen=True)
class _DAlphaFeatures:
    coverage: float
    positive_peak_z: float
    peak_prominence_z: float
    positive_area_z: float
    rise_slope_period: float
    peak_offset_period: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.coverage,
                self.positive_peak_z,
                self.peak_prominence_z,
                self.positive_area_z,
                self.rise_slope_period,
                self.peak_offset_period,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class _ExtremumFeatures:
    coverage: float
    signed_extremum_z: float
    extremum_prominence_z: float
    max_abs_period_slope: float
    derivative_reversal_period: float
    extremum_offset_period: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.coverage,
                self.signed_extremum_z,
                self.extremum_prominence_z,
                self.max_abs_period_slope,
                self.derivative_reversal_period,
                self.extremum_offset_period,
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


def _d_alpha_features(
    values: np.ndarray,
    valid_mask: np.ndarray,
    index: int,
    radius: int,
    dt: float,
    period: float,
    min_valid_samples: int,
    standardized: Optional[np.ndarray] = None,
) -> _DAlphaFeatures:
    """Describe the positive D-alpha burst expected near an SXR crash."""

    start = max(0, index - radius)
    stop = min(len(values), index + radius + 1)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    local_valid = valid_mask[start:stop]
    coverage = float(np.mean(local_valid)) if len(local_valid) else 0.0
    if standardized is None:
        standardized = _robust_standardize(values, valid_mask)

    pre = standardized[start:index]
    around = standardized[start:stop]
    pre = pre[np.isfinite(pre)]
    finite_positions = np.flatnonzero(np.isfinite(around))
    if len(pre) < min_valid_samples or len(finite_positions) < min_valid_samples:
        return _DAlphaFeatures(
            coverage=coverage,
            positive_peak_z=np.nan,
            peak_prominence_z=np.nan,
            positive_area_z=np.nan,
            rise_slope_period=np.nan,
            peak_offset_period=np.nan,
        )

    baseline = float(np.median(pre))
    peak_local = int(
        finite_positions[np.argmax(around[finite_positions])]
    )
    peak_value = float(around[peak_local])
    positive_peak = max(peak_value - baseline, 0.0)

    tail = around[peak_local + 1:]
    tail = tail[np.isfinite(tail)]
    after_level = float(np.median(tail)) if len(tail) else baseline
    prominence = max(peak_value - max(baseline, after_level), 0.0)

    positive_values = around[np.isfinite(around)] - baseline
    positive_area = float(np.mean(np.maximum(positive_values, 0.0)))

    rise = around[:peak_local + 1]
    adjacent_valid = np.isfinite(rise[:-1]) & np.isfinite(rise[1:])
    if np.any(adjacent_valid):
        differences = np.diff(rise)[adjacent_valid]
        rise_slope = max(float(np.max(differences)), 0.0) * period / dt
    else:
        rise_slope = np.nan

    peak_index = start + peak_local
    peak_offset = (peak_index - index) * dt / period
    return _DAlphaFeatures(
        coverage=coverage,
        positive_peak_z=positive_peak,
        peak_prominence_z=prominence,
        positive_area_z=positive_area,
        rise_slope_period=float(rise_slope),
        peak_offset_period=float(peak_offset),
    )


def _extremum_features(
    values: np.ndarray,
    valid_mask: np.ndarray,
    index: int,
    radius: int,
    dt: float,
    period: float,
    min_valid_samples: int,
    standardized: Optional[np.ndarray] = None,
) -> _ExtremumFeatures:
    """Describe a positive or negative local extremum near a candidate."""

    start = max(0, index - radius)
    stop = min(len(values), index + radius + 1)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    local_valid = valid_mask[start:stop]
    coverage = float(np.mean(local_valid)) if len(local_valid) else 0.0
    if standardized is None:
        standardized = _robust_standardize(values, valid_mask)

    pre = standardized[start:index]
    around = standardized[start:stop]
    pre = pre[np.isfinite(pre)]
    finite_positions = np.flatnonzero(np.isfinite(around))
    if len(pre) < min_valid_samples or len(finite_positions) < min_valid_samples:
        return _ExtremumFeatures(
            coverage=coverage,
            signed_extremum_z=np.nan,
            extremum_prominence_z=np.nan,
            max_abs_period_slope=np.nan,
            derivative_reversal_period=np.nan,
            extremum_offset_period=np.nan,
        )

    baseline = float(np.median(pre))
    deviations = around[finite_positions] - baseline
    extremum_local = int(finite_positions[np.argmax(np.abs(deviations))])
    extremum_value = float(around[extremum_local])
    signed_extremum = extremum_value - baseline

    left = around[:extremum_local]
    right = around[extremum_local + 1:]
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    left_level = float(np.median(left)) if len(left) else baseline
    right_level = float(np.median(right)) if len(right) else baseline
    if signed_extremum >= 0:
        prominence = max(extremum_value - max(left_level, right_level), 0.0)
    else:
        prominence = max(min(left_level, right_level) - extremum_value, 0.0)

    differences = np.diff(around)
    adjacent_valid = np.isfinite(around[:-1]) & np.isfinite(around[1:])
    valid_differences = differences[adjacent_valid]
    max_abs_slope = (
        float(np.max(np.abs(valid_differences))) * period / dt
        if len(valid_differences)
        else np.nan
    )

    before = np.diff(around[:extremum_local + 1])
    after = np.diff(around[extremum_local:])
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if len(before) and len(after):
        if signed_extremum >= 0:
            approach = max(float(np.max(before)), 0.0)
            departure = max(float(-np.min(after)), 0.0)
        else:
            approach = max(float(-np.min(before)), 0.0)
            departure = max(float(np.max(after)), 0.0)
        reversal = min(approach, departure) * period / dt
    else:
        reversal = np.nan

    extremum_index = start + extremum_local
    offset = (extremum_index - index) * dt / period
    return _ExtremumFeatures(
        coverage=coverage,
        signed_extremum_z=float(signed_extremum),
        extremum_prominence_z=float(prominence),
        max_abs_period_slope=float(max_abs_slope),
        derivative_reversal_period=float(reversal),
        extremum_offset_period=float(offset),
    )


class MultichannelFeatureBuilder:
    """Build one deterministic ML vector for a candidate on the plasma grid."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        profile: DiagnosticChannelProfile = CORE_DIAGNOSTICS_FEATURE_PROFILE,
        window_config: Optional[FeatureWindowConfig] = None,
    ):
        self.profile = profile
        self.include_multisxr = profile.name != "reference_sxr"
        self.include_proposals = profile.name != "reference_sxr"
        self.window_config = window_config or FeatureWindowConfig()
        self.schema = FeatureSchema(
            schema_version=self.SCHEMA_VERSION,
            profile_name=profile.name,
            feature_names=self._feature_names(),
            window_config=self.window_config,
        )

    def _feature_names(self) -> tuple[str, ...]:
        names = [f"sxr__{name}" for name in SINGLE_CHANNEL_FEATURE_NAMES]
        if self.include_multisxr:
            for channel in SXR_CHANNELS:
                names.extend(
                    f"sxr_channel__{channel}__{feature}"
                    for feature in _SXR_FEATURE_NAMES
                )
        for channel in self.profile.channels:
            if channel.group == "d_alpha":
                channel_feature_names = _D_ALPHA_FEATURE_NAMES
            elif channel.group in _EXTREMUM_GROUPS:
                channel_feature_names = _EXTREMUM_FEATURE_NAMES
            else:
                channel_feature_names = _CHANNEL_FEATURE_NAMES
            names.extend(
                f"channel__{channel.key}__{feature}"
                for feature in channel_feature_names
            )
        for group in self.profile.groups:
            if group == "d_alpha":
                group_feature_names = _D_ALPHA_GROUP_FEATURE_NAMES
            elif group in _EXTREMUM_GROUPS:
                group_feature_names = _EXTREMUM_GROUP_FEATURE_NAMES
            else:
                group_feature_names = _GROUP_FEATURE_NAMES
            names.extend(
                f"group__{group}__{feature}" for feature in group_feature_names
            )
        if self.include_proposals:
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
        if self.profile.channels and prepared.aligned_signals is None:
            raise ValueError("PreparedShot does not contain aligned diagnostic signals")
        if prepared.feature_profile != self.profile.name:
            raise ValueError(
                f"Prepared profile {prepared.feature_profile!r} does not match "
                f"feature schema profile {self.profile.name!r}"
            )

        reference = np.asarray(prepared.reference_signal, dtype=float)
        aligned = prepared.aligned_signals
        if aligned is not None and aligned.time.shape != reference.shape:
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
        channel_signals = getattr(prepared, "channel_signals", {})
        proposed_channels = set(
            () if proposal_evidence is None else proposal_evidence.get("source_channels", ())
        )
        methods_by_channel = (
            {} if proposal_evidence is None else proposal_evidence.get("source_channel_methods", {})
        )
        if self.include_multisxr:
            for channel_name in SXR_CHANNELS:
                signal = channel_signals.get(channel_name)
                if signal is None:
                    result.extend(
                        [0.0] * 4 + [np.nan] * (len(_SXR_FEATURE_NAMES) - 4)
                    )
                    continue
                values = np.asarray(signal, dtype=float)
                if values.shape != reference.shape:
                    raise ValueError(f"SXR channel {channel_name!r} length mismatch")
                valid = np.isfinite(values)
                standardized = (
                    None
                    if standardized_channels is None
                    else standardized_channels.get(channel_name)
                )
                if standardized is None:
                    standardized = _robust_standardize(values, valid)
                features = _channel_features(
                    values,
                    valid,
                    index=index,
                    radius=radius,
                    dt=dt,
                    period=effective_period,
                    min_valid_samples=self.window_config.min_valid_samples,
                    standardized=standardized,
                )
                pre = standardized[max(0, index - radius):index]
                post = standardized[index:min(len(values), index + radius)]
                differences = np.concatenate((np.diff(pre), np.diff(post)))
                differences = differences[np.isfinite(differences)]
                local_noise = (
                    float(
                        1.4826
                        * np.median(np.abs(differences - np.median(differences)))
                    )
                    if len(differences)
                    else np.nan
                )
                jump_to_noise = (
                    float(
                        np.log1p(
                            abs(features.median_delta_z) / (local_noise + _EPS)
                        )
                    )
                    if np.isfinite(local_noise)
                    and np.isfinite(features.median_delta_z)
                    else np.nan
                )
                result.extend(
                    [
                        1.0,
                        float(channel_name in proposed_channels),
                        float(
                            any(
                                "posr" in method.lower()
                                for method in methods_by_channel.get(channel_name, ())
                            )
                        ),
                        float(
                            any(
                                "cpd" in method.lower()
                                for method in methods_by_channel.get(channel_name, ())
                            )
                        ),
                        features.median_delta_z,
                        features.signed_jump_z,
                        features.log_energy_ratio,
                        features.max_abs_period_slope,
                        features.change_offset_period,
                        local_noise,
                        jump_to_noise,
                    ]
                )
        features_by_channel: dict[
            str,
            _ChannelFeatures | _DAlphaFeatures | _ExtremumFeatures,
        ] = {}

        for diagnostic in self.profile.channels:
            channel = aligned.get_channel(diagnostic.name)
            if diagnostic.group == "d_alpha":
                feature_fn = _d_alpha_features
            elif diagnostic.group in _EXTREMUM_GROUPS:
                feature_fn = _extremum_features
            else:
                feature_fn = _channel_features
            features = feature_fn(
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
            if group == "d_alpha":
                d_alpha_features = [
                    features
                    for features in group_features
                    if isinstance(features, _DAlphaFeatures)
                ]
                finite_prominences = np.asarray(
                    [features.peak_prominence_z for features in d_alpha_features],
                    dtype=float,
                )
                finite_prominences = finite_prominences[
                    np.isfinite(finite_prominences)
                ]
                result.extend(
                    [
                        float(np.mean([features.coverage for features in d_alpha_features])),
                        _finite_median(
                            [features.positive_peak_z for features in d_alpha_features]
                        ),
                        (
                            float(np.max(finite_prominences))
                            if len(finite_prominences)
                            else np.nan
                        ),
                        (
                            float(np.mean(finite_prominences >= 1.0))
                            if len(finite_prominences)
                            else np.nan
                        ),
                    ]
                )
            elif group in _EXTREMUM_GROUPS:
                extremum_features = [
                    features
                    for features in group_features
                    if isinstance(features, _ExtremumFeatures)
                ]
                finite_prominences = np.asarray(
                    [
                        features.extremum_prominence_z
                        for features in extremum_features
                    ],
                    dtype=float,
                )
                finite_prominences = finite_prominences[
                    np.isfinite(finite_prominences)
                ]
                result.extend(
                    [
                        float(
                            np.mean(
                                [features.coverage for features in extremum_features]
                            )
                        ),
                        _finite_median(
                            [
                                abs(features.signed_extremum_z)
                                for features in extremum_features
                            ]
                        ),
                        (
                            float(np.max(finite_prominences))
                            if len(finite_prominences)
                            else np.nan
                        ),
                        (
                            float(np.mean(finite_prominences >= 1.0))
                            if len(finite_prominences)
                            else np.nan
                        ),
                    ]
                )
            else:
                regular_features = [
                    features
                    for features in group_features
                    if isinstance(features, _ChannelFeatures)
                ]
                result.extend(
                    [
                        float(np.mean([features.coverage for features in regular_features])),
                        _finite_median(
                            [features.median_delta_z for features in regular_features]
                        ),
                        _finite_max_abs(
                            [features.signed_jump_z for features in regular_features]
                        ),
                        _finite_max_abs(
                            [features.max_abs_period_slope for features in regular_features]
                        ),
                    ]
                )

        if self.include_proposals and proposal_evidence is None:
            result.extend([np.nan] * len(_PROPOSAL_FEATURE_NAMES))
        elif self.include_proposals:
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

        if self.profile.channels and prepared.aligned_signals is None:
            raise ValueError("PreparedShot does not contain aligned diagnostic signals")
        standardized = {}
        if self.profile.channels:
            standardized.update({
                diagnostic.name: _robust_standardize(
                    prepared.aligned_signals.get_channel(diagnostic.name).values,
                    prepared.aligned_signals.get_channel(diagnostic.name).valid_mask,
                )
                for diagnostic in self.profile.channels
            })
        if self.include_multisxr:
            for channel_name in SXR_CHANNELS:
                signal = getattr(prepared, "channel_signals", {}).get(channel_name)
                if signal is not None:
                    values = np.asarray(signal, dtype=float)
                    standardized[channel_name] = _robust_standardize(
                        values, np.isfinite(values)
                    )
        return standardized

    def as_dict(self, vector: np.ndarray) -> dict[str, float]:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.schema.n_features,):
            raise ValueError(
                f"Expected a vector with {self.schema.n_features} features, "
                f"got {vector.shape}"
            )
        return dict(zip(self.schema.feature_names, vector.tolist()))
