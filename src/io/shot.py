from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


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
