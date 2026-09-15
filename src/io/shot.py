from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ChannelSignal:
    name: str
    time: np.ndarray
    values: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        time = np.asarray(self.time, dtype=float)
        values = np.asarray(self.values, dtype=float)

        if time.ndim != 1:
            raise ValueError(f"Channel {self.name!r}: time must be a 1D array")
        if values.ndim != 1:
            raise ValueError(f"Channel {self.name!r}: values must be a 1D array")
        if len(time) != len(values):
            raise ValueError(
                f"Channel {self.name!r}: time and values length mismatch "
                f"({len(time)} != {len(values)})"
            )
        if len(time) == 0:
            raise ValueError(f"Channel {self.name!r}: channel is empty")
        if not np.all(np.isfinite(time)):
            raise ValueError(f"Channel {self.name!r}: time contains non-finite values")
        if len(time) > 1 and np.any(np.diff(time) <= 0):
            raise ValueError(
                f"Channel {self.name!r}: time must be strictly increasing "
                "and contain no duplicate samples"
            )

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dt(self) -> Optional[float]:
        if len(self.time) < 2:
            return None
        return float(np.median(np.diff(self.time)))

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def n_samples(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class MultiRateShot:
    channels: Mapping[str, ChannelSignal]
    shot_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        channels = dict(self.channels)
        if not channels:
            raise ValueError("MultiRateShot must contain at least one channel")
        for name, channel in channels.items():
            if name != channel.name:
                raise ValueError(
                    f"Channel mapping key {name!r} does not match channel name "
                    f"{channel.name!r}"
                )
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def has_channel(self, name: str) -> bool:
        return name in self.channels

    def get_channel(self, name: str) -> ChannelSignal:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise KeyError(f"Channel {name!r} is not loaded") from exc


@dataclass
class Shot:
    time: np.ndarray
    signals: np.ndarray
    channel_names: List[str]
    shot_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if len(self.time.shape) != 1:
            raise ValueError("time must be 1D array")

        if len(self.signals.shape) != 2:
            raise ValueError("signals must be 2D array (T, N_channels)")

        if self.signals.shape[0] != len(self.time):
            raise ValueError("Time and signals length mismatch")

        if self.signals.shape[1] != len(self.channel_names):
            raise ValueError("Channel names count mismatch")

    @property
    def dt(self):
        if len(self.time) < 2:
            return None
        return self.time[1] - self.time[0]

    @property
    def duration(self):
        return self.time[-1] - self.time[0]

    @property
    def n_channels(self):
        return self.signals.shape[1]

    @property
    def n_samples(self):
        return self.signals.shape[0]
