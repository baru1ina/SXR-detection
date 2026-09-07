from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.features import (
    score_crash_candidate,
    smooth_signal,
    suppress_by_period,
)


class BaseCPDDetector(ABC):
    def __init__(
        self,
        dt: float,
        penalty: float = 8.0,
        model: str = "rbf",
        min_distance_factor: float = 0.65,
        score_threshold: float = 4.0,
        fallback_percentile: float = 99.5,
        max_candidates: Optional[int] = None,
        max_events_factor: float = 1.3,
        min_jump_to_amp: float = 0.05,
        candidate_margin_factor: float = 0.20,
        method_name: str = "cpd",
    ):
        self.dt = float(dt)
        self.penalty = float(penalty)
        self.model = model
        self.min_distance_factor = float(min_distance_factor)
        self.score_threshold = float(score_threshold)
        self.fallback_percentile = float(fallback_percentile)
        self.max_candidates = max_candidates
        self.max_events_factor = float(max_events_factor)
        self.min_jump_to_amp = float(min_jump_to_amp)
        self.candidate_margin_factor = float(candidate_margin_factor)
        self.method_name = method_name

        self.last_input: Optional[np.ndarray] = None
        self.last_raw_breakpoints: np.ndarray = np.array([], dtype=int)
        self.last_scored: list[CrashCandidate] = []
        self.last_candidates: list[CrashCandidate] = []

    @abstractmethod
    def prepare_input(self, signal: np.ndarray, period: Optional[float] = None) -> np.ndarray:
        """Вернуть одномерный сигнал или матрицу признаков для CPD."""

    def _get_cpd_candidates(self, x: np.ndarray) -> np.ndarray:
        try:
            import ruptures as rpt
        except ModuleNotFoundError:
            return np.array([], dtype=int)

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x_fit = x.reshape(-1, 1)
        else:
            x_fit = x

        if len(x_fit) < 20:
            return np.array([], dtype=int)

        if self.model in {"rbf", "linear", "cosine"}:
            algo = rpt.KernelCPD(kernel=self.model).fit(x_fit)
        else:
            # Для моделей ruptures вроде "l1", "l2", "normal".
            algo = rpt.Pelt(model=self.model).fit(x_fit)

        bkps = np.asarray(algo.predict(pen=self.penalty), dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < len(x_fit))]
        if len(bkps) and bkps[-1] == len(x_fit):
            bkps = bkps[:-1]
        return np.unique(bkps).astype(int)

    def _fallback_candidates(self, signal: np.ndarray, period: Optional[float]) -> np.ndarray:
        # self.logger.info("Fallback candidates detected.")
        x = smooth_signal(signal)
        if len(x) < 5:
            return np.array([], dtype=int)

        dx = np.abs(np.gradient(x, self.dt))
        threshold = np.percentile(dx, self.fallback_percentile)

        if period is None or not np.isfinite(period) or period <= 0:
            distance = max(1, int(0.25e-3 / self.dt))
        else:
            distance = max(1, int(0.25 * period / self.dt))

        peaks, _ = find_peaks(dx, height=threshold, distance=distance)
        return peaks.astype(int)

    def _candidate_margin_samples(self, period: Optional[float]) -> int:
        if period is None or not np.isfinite(period) or period <= 0:
            return max(1, int(0.25e-3 / self.dt))
        return max(1, int(self.candidate_margin_factor * period / self.dt))

    def _min_distance_samples(self, period: Optional[float]) -> int:
        if period is None or not np.isfinite(period) or period <= 0:
            return max(1, int(0.5e-3 / self.dt))
        return max(1, int(self.min_distance_factor * period / self.dt))

    def _max_events(self, n_samples: int, period: Optional[float]) -> Optional[int]:
        limits: list[int] = []

        if self.max_candidates is not None:
            limits.append(int(self.max_candidates))

        if period is not None and np.isfinite(period) and period > 0 and self.max_events_factor > 0:
            duration = n_samples * self.dt
            expected = max(1, int(np.ceil(duration / period)))
            limits.append(max(1, int(np.ceil(self.max_events_factor * expected))))

        if not limits:
            return None
        return max(1, min(limits))

    def _local_period(
            self,
            idx: int,
            period: Optional[float],
            period_map: Optional[np.ndarray],
    ) -> Optional[float]:
        if period_map is not None and 0 <= idx < len(period_map):
            p = float(period_map[idx])
            if np.isfinite(p) and p > 0:
                return p
        return period

    def _rank_and_filter(
        self,
        signal: np.ndarray,
        raw_candidates: np.ndarray,
        period: Optional[float],
        period_map: Optional[np.ndarray],
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
    ) -> list[CrashCandidate]:
        x = np.asarray(signal, dtype=float)
        n = len(x)
        margin = self._candidate_margin_samples(period)

        scored: list[CrashCandidate] = []
        accepted: list[CrashCandidate] = []
        rejected_by_mask = 0
        rejected_by_score = 0
        rejected_by_jump = 0

        for idx in np.asarray(raw_candidates, dtype=int):
            if idx <= margin or idx >= n - margin:
                continue
            if active_mask is not None and 0 <= int(idx) < len(active_mask) and not bool(active_mask[int(idx)]):
                rejected_by_mask += 1
                continue

            local_period = self._local_period(idx, period, period_map)

            score, direction, meta = score_crash_candidate(x, int(idx), self.dt, period=local_period)
            candidate = CrashCandidate(
                index=int(idx),
                time=int(idx) * self.dt,
                score=float(score),
                direction=direction,
                channel=channel,
                method=self.method_name,
                meta=meta,
            )
            scored.append(candidate)

            if score < self.score_threshold:
                rejected_by_score += 1
                continue
            if meta.get("jump_to_amp", 0.0) < self.min_jump_to_amp:
                rejected_by_jump += 1
                continue
            accepted.append(candidate)

        self.last_rejection_stats = {
            "mask": rejected_by_mask,
            "score": rejected_by_score,
            "jump_to_amp": rejected_by_jump,
            "accepted_before_nms": len(accepted),
        }
        self.last_scored = scored
        if not accepted:
            return []

        min_distance = self._min_distance_samples(period)
        kept_idx = suppress_by_period(
            [c.index for c in accepted],
            [c.score for c in accepted],
            min_distance=min_distance,
        )
        kept_set = set(map(int, kept_idx))
        result = [c for c in accepted if c.index in kept_set]
        result.sort(key=lambda c: c.index)

        max_events = self._max_events(n, period)
        if max_events is not None and len(result) > max_events:
            result = sorted(result, key=lambda c: c.score, reverse=True)[:max_events]
            result.sort(key=lambda c: c.index)

        return result

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

        cpd_input = self.prepare_input(x, period=period)
        self.last_input = cpd_input

        raw = self._get_cpd_candidates(cpd_input)
        if len(raw) == 0:
            raw = self._fallback_candidates(x, period)
        self.last_raw_breakpoints = raw

        candidates = self._rank_and_filter(
            x,
            raw,
            period=period,
            period_map=period_map,
            active_mask=active_mask,
            channel=channel,
        )

        self.last_candidates = candidates

        if debug:
            print(
                f"[{self.method_name}] raw={len(raw)}, "
                f"scored={len(self.last_scored)}, accepted={len(candidates)}, "
                f"penalty={self.penalty}, score_threshold={self.score_threshold}, "
                f"min_jump_to_amp={self.min_jump_to_amp}, "
                f"reject={getattr(self, 'last_rejection_stats', {})}"
            )
        return candidates

    def detect_raw_candidates(
        self,
        signal: np.ndarray,
        period: Optional[float] = None,
        period_map: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        channel: Optional[str] = None,
        debug: bool = False,
    ) -> list[CrashCandidate]:
        """Return CPD breakpoints without detector post-processing.

        This path intentionally does not apply the active mask, physics-score
        threshold, jump threshold, fallback detector, period suppression, or
        event-count limit.  Candidate scores are calculated only as metadata for
        diagnostics; the downstream classifier decides which points to keep.
        """

        del active_mask  # Explicitly ignored: masking would be post-processing.
        x = np.asarray(signal, dtype=float)
        if len(x) < 20:
            self.last_input = None
            self.last_raw_breakpoints = np.array([], dtype=int)
            self.last_scored = []
            self.last_candidates = []
            return []

        cpd_input = self.prepare_input(x, period=period)
        self.last_input = cpd_input
        raw = self._get_cpd_candidates(cpd_input)
        self.last_raw_breakpoints = raw

        candidates: list[CrashCandidate] = []
        for idx in np.asarray(raw, dtype=int):
            if not 0 <= int(idx) < len(x):
                continue
            local_period = self._local_period(int(idx), period, period_map)
            score, direction, meta = score_crash_candidate(
                x,
                int(idx),
                self.dt,
                period=local_period,
            )
            meta = dict(meta)
            meta["raw_cpd"] = True
            candidates.append(
                CrashCandidate(
                    index=int(idx),
                    time=int(idx) * self.dt,
                    score=float(score),
                    direction=direction,
                    channel=channel,
                    method=f"{self.method_name}_raw",
                    meta=meta,
                )
            )

        self.last_scored = candidates
        self.last_candidates = candidates
        if debug:
            print(
                f"[{self.method_name}:raw] breakpoints={len(raw)}, "
                "postprocessing=disabled"
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
        candidates = self.detect_candidates(
            signal,
            period=period,
            period_map=period_map,
            active_mask=active_mask,
            channel=channel,
            debug=debug,
        )
        return candidates_to_indices(candidates)
