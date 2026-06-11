from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from src.detection.utils.common import CrashCandidate


@dataclass
class VotedCrash:
    index: int
    time: float
    score: float
    channels: list[str]
    members: list[CrashCandidate] = field(default_factory=list)


class MultiChannelVotingDetector:
    def __init__(
        self,
        detector_factory: Callable[[], object],
        dt: float,
        coincidence_window: float = 0.3e-3,
        min_channels: int = 2,
        aggregate: str = "weighted_median",
        period_map: Optional[np.ndarray] = None,
    ):
        self.detector_factory = detector_factory
        self.dt = float(dt)
        self.coincidence_window = float(coincidence_window)
        self.min_channels = int(min_channels)
        self.aggregate = aggregate

        self.channel_candidates: Dict[str, list[CrashCandidate]] = {}
        self.last_events: list[VotedCrash] = []

    def _run_one(
        self,
        channel: str,
        signal: np.ndarray,
        period: Optional[float],
        debug: bool,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
    ) -> list[CrashCandidate]:
        detector = self.detector_factory()
        if hasattr(detector, "detect_candidates"):
            candidates = detector.detect_candidates(
                signal,
                period=period,
                period_map=period_map,
                active_mask=active_mask,
                channel=channel,
                debug=debug,
            )
        else:
            indices = detector.detect(
                signal,
                period=period,
                period_map=period_map,
                active_mask=active_mask,
                debug=debug,
            )
            candidates = [
                CrashCandidate(index=int(i), time=int(i) * self.dt, channel=channel, method=detector.__class__.__name__)
                for i in indices
            ]
        for c in candidates:
            c.channel = channel
            c.time = c.index * self.dt
        return candidates

    def _cluster_by_time(self, candidates: list[CrashCandidate]) -> list[list[CrashCandidate]]:
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda c: c.index)
        max_gap = max(1, int(self.coincidence_window / self.dt))
        clusters: list[list[CrashCandidate]] = []
        current = [candidates[0]]
        for c in candidates[1:]:
            if c.index - current[-1].index <= max_gap:
                current.append(c)
            else:
                clusters.append(current)
                current = [c]
        clusters.append(current)
        return clusters

    def _aggregate_cluster(self, cluster: list[CrashCandidate]) -> VotedCrash:
        indices = np.array([c.index for c in cluster], dtype=float)
        scores = np.array([max(c.score, 1e-6) for c in cluster], dtype=float)
        if self.aggregate == "mean":
            idx = int(round(float(np.mean(indices))))
        elif self.aggregate == "weighted_mean":
            idx = int(round(float(np.average(indices, weights=scores))))
        else:
            center = float(np.average(indices, weights=scores))
            idx = int(indices[np.argmin(np.abs(indices - center))])
        return VotedCrash(
            index=idx,
            time=idx * self.dt,
            score=float(np.sum(scores)),
            channels=sorted({c.channel or "unknown" for c in cluster}),
            members=cluster,
        )

    def detect_events(
        self,
        signals: Dict[str, np.ndarray],
        period: Optional[float] = None,
        debug: bool = False,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
    ) -> list[VotedCrash]:
        self.channel_candidates = {}
        all_candidates: list[CrashCandidate] = []

        for channel, signal in signals.items():
            candidates = self._run_one(
                channel,
                signal,
                period=period,
                debug=debug,
                period_map=period_map,
                active_mask=active_mask,
            )
            self.channel_candidates[channel] = candidates
            all_candidates.extend(candidates)

        events: list[VotedCrash] = []
        for cluster in self._cluster_by_time(all_candidates):
            channels = {c.channel for c in cluster}
            if len(channels) < self.min_channels:
                continue
            events.append(self._aggregate_cluster(cluster))

        events.sort(key=lambda e: e.index)
        self.last_events = events
        return events

    def detect(
        self,
        signals: Dict[str, np.ndarray],
        period: Optional[float] = None,
        debug: bool = False,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        events = self.detect_events(
            signals,
            period=period,
            debug=debug,
            period_map=period_map,
            active_mask=active_mask,
        )
        return np.array([e.index for e in events], dtype=int)
