from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CrashCandidate:

    index: int
    time: Optional[float] = None
    score: float = 0.0
    direction: str = "unknown"  # normal / inverted / bidirectional / unknown
    channel: Optional[str] = None
    method: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def shifted(self, offset: int, dt: float) -> "CrashCandidate":
        return CrashCandidate(
            index=int(self.index + offset),
            time=(self.index + offset) * dt,
            score=float(self.score),
            direction=self.direction,
            channel=self.channel,
            method=self.method,
            meta=dict(self.meta),
        )


def candidates_to_indices(candidates: list[CrashCandidate]) -> np.ndarray:
    if not candidates:
        return np.array([], dtype=int)
    return np.array([int(c.index) for c in candidates], dtype=int)


def candidates_to_times(candidates: list[CrashCandidate], dt: float, t0: float = 0.0) -> np.ndarray:
    if not candidates:
        return np.array([], dtype=float)
    return np.array([t0 + c.index * dt for c in candidates], dtype=float)
