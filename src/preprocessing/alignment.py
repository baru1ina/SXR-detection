from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from src.io.shot import ChannelSignal, MultiRateShot


SUPPORTED_ALIGNMENT_METHODS = frozenset({"linear", "nearest", "previous"})


def _validated_time_grid(time: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(time, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if len(result) == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    if len(result) > 1 and np.any(np.diff(result) <= 0):
        raise ValueError(f"{label} must be strictly increasing")
    return result


@dataclass(frozen=True)
class ChannelAlignmentSpec:
    method: str = "linear"
    max_gap_s: Optional[float] = None

    def __post_init__(self):
        method = self.method.casefold()
        if method not in SUPPORTED_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unsupported alignment method {self.method!r}; expected one of "
                f"{sorted(SUPPORTED_ALIGNMENT_METHODS)}"
            )
        if self.max_gap_s is not None:
            max_gap_s = float(self.max_gap_s)
            if not np.isfinite(max_gap_s) or max_gap_s <= 0:
                raise ValueError("max_gap_s must be a positive finite number")
            object.__setattr__(self, "max_gap_s", max_gap_s)
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class AlignedChannel:
    name: str
    time: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    source_dt: Optional[float]
    method: str
    source_present: bool = True
    resampled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        time = _validated_time_grid(self.time, f"Aligned channel {self.name!r} time")
        values = np.asarray(self.values, dtype=float)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)

        if values.ndim != 1 or valid_mask.ndim != 1:
            raise ValueError(
                f"Aligned channel {self.name!r}: values and valid_mask must be 1D"
            )
        if len(values) != len(time) or len(valid_mask) != len(time):
            raise ValueError(
                f"Aligned channel {self.name!r}: time, values and valid_mask "
                "lengths must match"
            )
        if np.any(valid_mask & ~np.isfinite(values)):
            raise ValueError(
                f"Aligned channel {self.name!r}: valid values must be finite"
            )

        values = values.copy()
        values[~valid_mask] = np.nan
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def coverage(self) -> float:
        return self.n_valid / len(self.time)

    @property
    def is_available(self) -> bool:
        return self.source_present and self.n_valid > 0

    @property
    def target_dt(self) -> Optional[float]:
        if len(self.time) < 2:
            return None
        return float(np.median(np.diff(self.time)))


@dataclass(frozen=True)
class AlignedSignalSet:
    time: np.ndarray
    channels: Mapping[str, AlignedChannel]

    def __post_init__(self):
        time = _validated_time_grid(self.time, "Aligned signal-set time")
        channels = dict(self.channels)
        if not channels:
            raise ValueError("AlignedSignalSet must contain at least one channel")
        for name, channel in channels.items():
            if name != channel.name:
                raise ValueError(
                    f"Channel mapping key {name!r} does not match {channel.name!r}"
                )
            if channel.time.shape != time.shape or not np.array_equal(channel.time, time):
                raise ValueError(
                    f"Aligned channel {name!r} does not use the signal-set time grid"
                )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "channels", channels)

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels)

    @property
    def available_channel_names(self) -> list[str]:
        return [name for name, channel in self.channels.items() if channel.is_available]

    @property
    def missing_channel_names(self) -> list[str]:
        return [name for name, channel in self.channels.items() if not channel.source_present]

    def get_channel(self, name: str) -> AlignedChannel:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise KeyError(f"Aligned channel {name!r} is not present") from exc


def _exact_source_indices(
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(source_time, target_time, side="left")
    clipped = np.clip(indices, 0, len(source_time) - 1)
    exact = (indices < len(source_time)) & (source_time[clipped] == target_time)
    return clipped, exact


def _align_linear(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
    max_gap_s: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(target_time), np.nan, dtype=float)
    valid = np.zeros(len(target_time), dtype=bool)
    exact_indices, exact = _exact_source_indices(source_time, target_time)

    exact_valid = exact & np.isfinite(source_values[exact_indices])
    values[exact_valid] = source_values[exact_indices[exact_valid]]
    valid[exact_valid] = True

    if len(source_time) < 2:
        return values, valid

    right = np.searchsorted(source_time, target_time, side="right")
    bracketed = (~exact) & (right > 0) & (right < len(source_time))
    bracket_positions = np.flatnonzero(bracketed)
    if not len(bracket_positions):
        return values, valid

    right_indices = right[bracket_positions]
    left_indices = right_indices - 1
    gaps = source_time[right_indices] - source_time[left_indices]
    usable = (
        np.isfinite(source_values[left_indices])
        & np.isfinite(source_values[right_indices])
    )
    if max_gap_s is not None:
        usable &= gaps <= max_gap_s

    positions = bracket_positions[usable]
    if not len(positions):
        return values, valid

    left_indices = left_indices[usable]
    right_indices = right_indices[usable]
    weights = (
        (target_time[positions] - source_time[left_indices])
        / (source_time[right_indices] - source_time[left_indices])
    )
    values[positions] = (
        source_values[left_indices]
        + weights * (source_values[right_indices] - source_values[left_indices])
    )
    valid[positions] = True
    return values, valid


def _align_nearest(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
    max_gap_s: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source_time, target_time, side="left")
    right_indices = np.clip(right, 0, len(source_time) - 1)
    left_indices = np.clip(right - 1, 0, len(source_time) - 1)
    left_distance = np.abs(target_time - source_time[left_indices])
    right_distance = np.abs(source_time[right_indices] - target_time)
    use_left = left_distance <= right_distance
    selected = np.where(use_left, left_indices, right_indices)
    distance = np.abs(target_time - source_time[selected])

    valid = (
        (target_time >= source_time[0])
        & (target_time <= source_time[-1])
        & np.isfinite(source_values[selected])
    )
    if max_gap_s is not None:
        valid &= distance <= max_gap_s

    values = np.full(len(target_time), np.nan, dtype=float)
    values[valid] = source_values[selected[valid]]
    return values, valid


def _align_previous(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
    max_gap_s: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.searchsorted(source_time, target_time, side="right") - 1
    selected_clipped = np.clip(selected, 0, len(source_time) - 1)
    age = target_time - source_time[selected_clipped]
    valid = (
        (selected >= 0)
        & (target_time <= source_time[-1])
        & np.isfinite(source_values[selected_clipped])
    )
    if max_gap_s is not None:
        valid &= age <= max_gap_s

    values = np.full(len(target_time), np.nan, dtype=float)
    values[valid] = source_values[selected_clipped[valid]]
    return values, valid


def align_channel(
    channel: ChannelSignal,
    target_time: np.ndarray,
    spec: Optional[ChannelAlignmentSpec] = None,
) -> AlignedChannel:
    """Align one native-rate channel without extrapolating its time range."""

    target_time = _validated_time_grid(target_time, "target_time")
    spec = spec or ChannelAlignmentSpec()
    source_indices = np.searchsorted(channel.time, target_time, side="left")
    clipped_indices = np.clip(source_indices, 0, len(channel.time) - 1)
    exact_subset = np.all(
        (source_indices < len(channel.time))
        & (channel.time[clipped_indices] == target_time)
    )
    if exact_subset:
        direct_values = channel.values[clipped_indices]
        valid_mask = np.isfinite(direct_values)
        return AlignedChannel(
            name=channel.name,
            time=target_time,
            values=direct_values,
            valid_mask=valid_mask,
            source_dt=channel.dt,
            method=spec.method,
            source_present=True,
            resampled=False,
            metadata=channel.metadata,
        )

    aligner = {
        "linear": _align_linear,
        "nearest": _align_nearest,
        "previous": _align_previous,
    }[spec.method]
    values, valid_mask = aligner(
        channel.time,
        channel.values,
        target_time,
        spec.max_gap_s,
    )
    return AlignedChannel(
        name=channel.name,
        time=target_time,
        values=values,
        valid_mask=valid_mask,
        source_dt=channel.dt,
        method=spec.method,
        source_present=True,
        resampled=True,
        metadata=channel.metadata,
    )


def missing_aligned_channel(
    name: str,
    target_time: np.ndarray,
    spec: Optional[ChannelAlignmentSpec] = None,
) -> AlignedChannel:
    target_time = _validated_time_grid(target_time, "target_time")
    spec = spec or ChannelAlignmentSpec()
    return AlignedChannel(
        name=name,
        time=target_time,
        values=np.full(len(target_time), np.nan, dtype=float),
        valid_mask=np.zeros(len(target_time), dtype=bool),
        source_dt=None,
        method=spec.method,
        source_present=False,
        resampled=False,
    )


def align_multirate_shot(
    shot: MultiRateShot,
    target_time: np.ndarray,
    channel_names: list[str],
    specs: Optional[Mapping[str, ChannelAlignmentSpec]] = None,
    default_spec: Optional[ChannelAlignmentSpec] = None,
) -> AlignedSignalSet:
    """Align a fixed ordered channel set, including placeholders for missing data."""

    target_time = _validated_time_grid(target_time, "target_time")
    ordered_names = list(dict.fromkeys(channel_names))
    if not ordered_names:
        raise ValueError("channel_names must not be empty")

    specs = dict(specs or {})
    default_spec = default_spec or ChannelAlignmentSpec()
    unknown_specs = set(specs) - set(ordered_names)
    if unknown_specs:
        raise ValueError(
            f"Alignment specs contain channels that were not requested: "
            f"{sorted(unknown_specs)}"
        )

    aligned = {}
    for name in ordered_names:
        spec = specs.get(name, default_spec)
        if shot.has_channel(name):
            aligned[name] = align_channel(shot.get_channel(name), target_time, spec)
        else:
            aligned[name] = missing_aligned_channel(name, target_time, spec)

    return AlignedSignalSet(time=target_time, channels=aligned)
