from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from src.detection.models.wavelet_core import WaveletEdgeCore
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.features import score_crash_candidate, suppress_by_period


class WaveletPOSRDetector:
    def __init__(
        self,
        dt: float,
        sigma: float = 80e-6,
        frame: Optional[float] = None,
        threshold: float = 6.0,
        min_distance_factor: float = 0.45,
        score_threshold: float = 2.5,
        detect_both_directions: bool = True,
        crash_time_min: Optional[float] = None,
        crash_time_max: Optional[float] = None,
        wavelet_name: str = "gaus1",
        min_jump_to_amp: float = 0.0,
        edge_margin: float = 0.5e-3,
    ):
        self.dt = float(dt)
        self.sigma = float(sigma)
        self.frame = frame
        self.threshold = float(threshold)
        self.min_distance_factor = float(min_distance_factor)
        self.score_threshold = float(score_threshold)
        self.detect_both_directions = bool(detect_both_directions)
        self.min_jump_to_amp = float(min_jump_to_amp)
        self.edge_margin = float(edge_margin)

        cmin = crash_time_min if crash_time_min is not None else max(10e-6, 0.35 * self.sigma)
        cmax = crash_time_max if crash_time_max is not None else max(100e-6, 2.5 * self.sigma)
        if cmax <= cmin:
            cmax = cmin * 2

        self.core = WaveletEdgeCore(
            dt=self.dt,
            crash_time_min=float(cmin),
            crash_time_max=float(cmax),
            wavelet_name=wavelet_name,
            response_scale_seconds=self.sigma,
            enhance_energy=False,
        )

        self.last_response: Optional[np.ndarray] = None
        self.last_weights: Optional[np.ndarray] = None
        self.last_candidates: list[CrashCandidate] = []

    def _posr_weight_at(self, response: np.ndarray, peak_idx: int, frame_samples: int, exclude_samples: int) -> float:
        n = len(response)
        left = max(0, peak_idx - frame_samples // 2)
        right = min(n, peak_idx + frame_samples // 2 + 1)
        frame = response[left:right]
        if len(frame) < max(8, exclude_samples + 3):
            return 0.0

        abs_order = np.argsort(np.abs(frame))[::-1]
        keep = np.ones(len(frame), dtype=bool)
        keep[abs_order[: min(exclude_samples, len(frame) - 3)]] = False
        reduced = frame[keep]
        if len(reduced) < 3:
            return 0.0

        mu = float(np.mean(reduced))
        sigma = float(np.std(reduced, ddof=1) + 1e-12)
        return abs(float(response[peak_idx]) - mu) / sigma

    def _frame_samples(self, period: Optional[float]) -> int:
        if self.frame is not None:
            frame_samples = max(9, int(round(self.frame / self.dt)))
        elif period is not None and np.isfinite(period) and period > 0:
            frame_samples = max(9, int(round(min(0.9 * period, 5e-3) / self.dt)))
        else:
            frame_samples = max(9, int(round(1.0e-3 / self.dt)))
        if frame_samples % 2 == 0:
            frame_samples += 1
        return frame_samples

    def _distance_samples(self, period: Optional[float]) -> int:
        if period is None or not np.isfinite(period) or period <= 0:
            return max(1, int(round(0.25e-3 / self.dt)))
        return max(1, int(round(self.min_distance_factor * float(period) / self.dt)))

    def _local_period(self, idx: int, period: Optional[float], period_map: Optional[np.ndarray]) -> Optional[float]:
        if period_map is not None and 0 <= idx < len(period_map):
            p = float(period_map[idx])
            if np.isfinite(p) and p > 0:
                return p
        return period

    def compute_posr_weights(self, response: np.ndarray, period: Optional[float]) -> np.ndarray:
        frame_samples = self._frame_samples(period)
        exclude_samples = max(3, int(round(7 * self.sigma / self.dt)))
        y = np.abs(response) if self.detect_both_directions else np.maximum(response, 0.0)

        distance = max(1, int(round(0.15e-3 / self.dt)))
        peaks, _ = find_peaks(y, distance=distance)
        weights = np.zeros_like(response, dtype=float)
        for p in peaks:
            weights[p] = self._posr_weight_at(response, int(p), frame_samples, exclude_samples)
        self.last_weights = weights
        return weights

    def detect_candidates(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> list[CrashCandidate]:
        x = np.asarray(signal, dtype=float)
        if len(x) < 100:
            self.last_candidates = []
            return []

        result = self.core.transform(x)
        response = np.asarray(result.signed_response, dtype=float)
        self.last_response = response

        weights = self.compute_posr_weights(response, period=period)
        weights_for_peaks = weights.copy()

        reject = {"mask": 0, "score": 0, "jump_to_amp": 0}

        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool)
            if len(mask) == len(weights_for_peaks):
                reject["mask"] = int(np.count_nonzero(weights_for_peaks[~mask] >= self.threshold))
                weights_for_peaks[~mask] = 0.0

        margin_samples = max(0, int(round(self.edge_margin / self.dt)))
        if margin_samples > 0 and len(weights_for_peaks) > 2 * margin_samples:
            weights_for_peaks[:margin_samples] = 0.0
            weights_for_peaks[-margin_samples:] = 0.0
        elif margin_samples > 0:
            weights_for_peaks[:] = 0.0

        distance = self._distance_samples(period)
        peaks, _ = find_peaks(weights_for_peaks, height=self.threshold, distance=max(1, distance // 2))

        raw_candidates: list[CrashCandidate] = []
        for p in peaks:
            local_period = self._local_period(int(p), period, period_map)
            score, direction, meta = score_crash_candidate(x, int(p), self.dt, period=local_period)
            if score < self.score_threshold:
                reject["score"] += 1
                continue

            jump_to_amp = float(meta.get("jump_to_amp", 0.0))
            if jump_to_amp < self.min_jump_to_amp:
                reject["jump_to_amp"] += 1
                continue

            meta = dict(meta)
            meta["posr_weight"] = float(weights[p])
            meta["wavelet_response"] = float(response[p])
            meta["response_scale"] = int(result.response_scale)
            raw_candidates.append(
                CrashCandidate(
                    index=int(p),
                    time=int(p) * self.dt,
                    score=float(score + 0.25 * weights[p]),
                    direction=direction,
                    channel=channel,
                    method="wavelet_posr",
                    meta=meta,
                )
            )

        kept_idx = suppress_by_period(
            [cnd.index for cnd in raw_candidates],
            [cnd.score for cnd in raw_candidates],
            min_distance=distance,
        )
        kept = set(map(int, kept_idx))
        candidates = [cnd for cnd in raw_candidates if cnd.index in kept]
        candidates.sort(key=lambda x: x.index)
        self.last_candidates = candidates

        if debug:
            print(
                f"[Wavelet/POSR] channel={channel}, raw_peaks={len(peaks)}, "
                f"accepted={len(candidates)}, threshold={self.threshold}, "
                f"score_threshold={self.score_threshold}, response_scale={result.response_scale}, "
                f"distance={distance}, reject={reject}"
            )
        return candidates

    def detect(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> np.ndarray:
        return candidates_to_indices(
            self.detect_candidates(
                signal,
                period=period,
                period_map=period_map,
                active_mask=active_mask,
                channel=channel,
                debug=debug,
            )
        )


