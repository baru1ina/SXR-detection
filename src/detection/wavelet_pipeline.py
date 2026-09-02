from typing import Iterable, Optional
from scipy.signal import find_peaks

import numpy as np

from src.detection.models.wavelet_core import WaveletEdgeCore
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.features import score_crash_candidate
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)
from src.physics.autocorr_period import refine_period_map_from_events


class WaveletEnergyDetector:
    def __init__(
        self,
        dt: float,
        period: float,
        crash_time_min: float = 10e-6,
        crash_time_max: float = 100e-6,
        percentile_threshold: float = 98.0,
        min_period: float = 0.25e-3,
        wavelet_name: str = "mexh",
        wt_threshold: float = 1.0,
        edge_margin: float = 0.5e-3,
        distance_factor: float = 0.55,
        rescue_threshold_factor: float = 0.55,
        raw_peak_distance: float = 0.15e-3,
        use_local_period_nms: bool = True,
        use_periodic_support_filter: bool = True,
        periodic_min_chain_len: int = 2,
        periodic_tolerance: float = 0.40,
        periodic_suppression_factor: float = 0.50,
        periodic_max_skip: int = 6,
        refine_period_from_events: bool = True,
        period_seed_merge_distance: float = 0.35e-3,
        period_seed_score_threshold: float = 2.25,
        period_seed_jump_threshold: float = 0.35,
        period_refinement_min_period: float = 0.75e-3,
    ):
        self.dt = float(dt)
        self.period = float(period) if period is not None else np.nan
        self.percentile_threshold = float(percentile_threshold)
        self.min_period = float(min_period)
        self.wt_threshold = float(wt_threshold)
        self.edge_margin = float(edge_margin)
        self.distance_factor = float(distance_factor)
        self.rescue_threshold_factor = float(rescue_threshold_factor)

        self.raw_peak_distance = float(raw_peak_distance)
        self.use_local_period_nms = bool(use_local_period_nms)
        self.use_periodic_support_filter = bool(use_periodic_support_filter)
        self.periodic_min_chain_len = max(1, int(periodic_min_chain_len))
        self.periodic_tolerance = float(periodic_tolerance)
        self.periodic_suppression_factor = float(periodic_suppression_factor)
        self.periodic_max_skip = max(1, int(periodic_max_skip))
        self.refine_period_from_events = bool(refine_period_from_events)
        self.period_seed_merge_distance = float(period_seed_merge_distance)
        self.period_seed_score_threshold = float(period_seed_score_threshold)
        self.period_seed_jump_threshold = float(period_seed_jump_threshold)
        self.period_refinement_min_period = float(period_refinement_min_period)

        self.core = WaveletEdgeCore(
            dt=self.dt,
            crash_time_min=crash_time_min,
            crash_time_max=crash_time_max,
            wavelet_name=wavelet_name,
            enhance_energy=True,
        )

        self.last_energy: Optional[np.ndarray] = None
        self.last_threshold: Optional[float] = None
        self.last_candidates: list[CrashCandidate] = []

        self.last_relaxed_threshold: Optional[float] = None
        self.last_primary_peaks: np.ndarray = np.array([], dtype=int)
        self.last_rescue_peaks: np.ndarray = np.array([], dtype=int)
        self.last_all_peaks: np.ndarray = np.array([], dtype=int)
        self.last_periodic_peaks: np.ndarray = np.array([], dtype=int)
        self.last_periodic_component_sizes: list[int] = []
        self.last_period_seed_peaks: np.ndarray = np.array([], dtype=int)
        self.last_period_map: Optional[np.ndarray] = None

    def _fallback_period(self) -> float:
        if np.isfinite(self.period) and self.period > 0:
            return float(self.period)
        return float(self.min_period)

    def _distance_samples(self, period: Optional[float]) -> int:
        p = period if period is not None and np.isfinite(period) and period > 0 else self._fallback_period()
        return max(1, int(round(self.distance_factor * float(p) / self.dt)))

    def _raw_peak_distance_samples(self, period_map: Optional[np.ndarray] = None) -> int:
        candidates = [self.raw_peak_distance]
        if period_map is not None:
            pm = np.asarray(period_map, dtype=float)
            finite = pm[np.isfinite(pm) & (pm > 0)]
            if len(finite):
                candidates.append(0.25 * float(np.percentile(finite, 10)))
        candidates.append(0.25 * self._fallback_period())
        dist_sec = max(self.dt, min(c for c in candidates if np.isfinite(c) and c > 0))
        return max(1, int(round(dist_sec / self.dt)))

    def _local_period(
        self,
        idx: int,
        period: Optional[float],
        period_map: Optional[np.ndarray],
    ) -> float:
        if period_map is not None and 0 <= int(idx) < len(period_map):
            p = float(period_map[int(idx)])
            if np.isfinite(p) and p > 0:
                return p
        if period is not None and np.isfinite(period) and period > 0:
            return float(period)
        return self._fallback_period()

    def _local_distance_samples(
        self,
        idx: int,
        period: Optional[float],
        period_map: Optional[np.ndarray],
    ) -> int:
        p = self._local_period(idx, period, period_map)
        return max(1, int(round(self.distance_factor * p / self.dt)))

    def _nms_by_energy(
        self,
        peaks: np.ndarray,
        energy: np.ndarray,
        *,
        period: Optional[float],
        period_map: Optional[np.ndarray],
    ) -> np.ndarray:
        if len(peaks) == 0:
            return np.array([], dtype=int)

        peaks = np.asarray(peaks, dtype=int)
        order = sorted(peaks, key=lambda i: float(energy[i]), reverse=True)
        kept: list[int] = []

        for p in order:
            p = int(p)
            p_dist = self._local_distance_samples(p, period, period_map)
            is_duplicate = False
            for k in kept:
                if self.use_local_period_nms:
                    k_dist = self._local_distance_samples(k, period, period_map)
                    required = min(p_dist, k_dist)
                else:
                    required = self._distance_samples(period)
                if abs(p - int(k)) < required:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(p)

        return np.array(sorted(kept), dtype=int)

    def _fixed_nms_by_energy(
        self,
        peaks: np.ndarray,
        energy: np.ndarray,
        distance_seconds: float,
    ) -> np.ndarray:
        """Merge neighbouring wavelet lobes without using any period map."""
        if len(peaks) == 0:
            return np.array([], dtype=int)
        min_distance = max(1, int(round(float(distance_seconds) / self.dt)))
        order = sorted(np.asarray(peaks, dtype=int), key=lambda i: float(energy[i]), reverse=True)
        kept: list[int] = []
        for peak in order:
            if all(abs(int(peak) - other) >= min_distance for other in kept):
                kept.append(int(peak))
        return np.asarray(sorted(kept), dtype=int)

    def _select_period_seed_peaks(
        self,
        signal: np.ndarray,
        primary_peaks: np.ndarray,
        energy: np.ndarray,
    ) -> np.ndarray:
        """Select strong, crash-shaped events for event-based map refinement."""
        merged = self._fixed_nms_by_energy(
            primary_peaks,
            energy,
            self.period_seed_merge_distance,
        )
        selected: list[int] = []
        for peak in merged:
            score, direction, meta = score_crash_candidate(
                signal,
                int(peak),
                self.dt,
                period=None,
            )
            jump_to_amp = float(meta.get("jump_to_amp", 0.0))
            if (
                direction != "edge"
                and score >= self.period_seed_score_threshold
                and jump_to_amp >= self.period_seed_jump_threshold
            ):
                selected.append(int(peak))
        return np.asarray(selected, dtype=int)

    def _best_consistent_period_chain(
        self,
        peaks: np.ndarray,
        energy: np.ndarray,
        reference_threshold: float,
        max_adjacent_period_ratio: float = 2.25,
    ) -> np.ndarray:
        """Find the strongest period-consistent subsequence of event seeds.

        The dynamic program may skip a weak peak between two real crashes.
        This prevents a single extra candidate from breaking an accelerating
        3 -> 2 -> 1.5 ms packet into unrelated fragments.
        """
        peaks = np.asarray(sorted(np.asarray(peaks, dtype=int)), dtype=int)
        if len(peaks) < 3:
            return np.array([], dtype=int)

        eps = 1e-12
        threshold = max(float(reference_threshold), eps)
        weights = np.maximum(0.0, np.log((energy[peaks] + eps) / threshold))
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

        for left in range(len(peaks) - 1):
            for right in range(left + 1, len(peaks)):
                gap = (int(peaks[right]) - int(peaks[left])) * self.dt
                if not (self.period_refinement_min_period <= gap <= 10e-3):
                    continue
                skip_penalty = 0.15 * max(0, right - left - 1)
                states[(left, right)] = (
                    float(weights[left] + weights[right] - skip_penalty),
                    (left, right),
                )

        for middle in range(1, len(peaks) - 1):
            incoming = [key for key in states if key[1] == middle]
            for left, _ in incoming:
                state_score, path = states[(left, middle)]
                previous_gap = (int(peaks[middle]) - int(peaks[left])) * self.dt
                for right in range(middle + 1, len(peaks)):
                    next_gap = (int(peaks[right]) - int(peaks[middle])) * self.dt
                    if not (self.period_refinement_min_period <= next_gap <= 10e-3):
                        continue
                    ratio = max(previous_gap, next_gap) / max(min(previous_gap, next_gap), eps)
                    if ratio > max_adjacent_period_ratio:
                        continue
                    change_penalty = 0.35 * abs(np.log(ratio))
                    skip_penalty = 0.15 * max(0, right - middle - 1)
                    candidate_score = float(
                        state_score + weights[right] - change_penalty - skip_penalty
                    )
                    key = (middle, right)
                    existing = states.get(key)
                    if existing is None or candidate_score > existing[0]:
                        states[key] = (candidate_score, path + (right,))

        chains = [value for value in states.values() if len(value[1]) >= 3]
        if not chains:
            return np.array([], dtype=int)
        best_score, best_path = max(chains, key=lambda value: value[0])
        if best_score <= 0:
            return np.array([], dtype=int)
        return peaks[np.asarray(best_path, dtype=int)]

    def _refine_period_map(
        self,
        signal: np.ndarray,
        candidate_peaks: np.ndarray,
        energy: np.ndarray,
        period_map: Optional[np.ndarray],
        reference_threshold: float,
    ) -> Optional[np.ndarray]:
        if not self.refine_period_from_events:
            self.last_period_seed_peaks = np.array([], dtype=int)
            return period_map

        seeds = self._select_period_seed_peaks(signal, candidate_peaks, energy)
        seeds = self._best_consistent_period_chain(
            seeds,
            energy,
            reference_threshold=reference_threshold,
        )
        self.last_period_seed_peaks = seeds
        refined = refine_period_map_from_events(
            len(signal),
            seeds,
            dt=self.dt,
            base_period_map=period_map,
            min_period=self.period_refinement_min_period,
            max_period=10e-3,
            min_chain_events=max(3, self.periodic_min_chain_len),
        )
        return refined

    def _is_periodic_gap(
        self,
        left: int,
        right: int,
        *,
        period: Optional[float],
        period_map: Optional[np.ndarray],
    ) -> tuple[bool, float]:
        gap = (int(right) - int(left)) * self.dt
        if gap <= 0:
            return False, np.inf

        left_period = self._local_period(left, period, period_map)
        right_period = self._local_period(right, period, period_map)
        expected = 0.5 * (left_period + right_period)
        if not np.isfinite(expected) or expected <= 0:
            expected = self._fallback_period()

        ratio = gap / expected
        period_steps = int(round(ratio))
        if period_steps < 1 or period_steps > self.periodic_max_skip:
            return False, np.inf

        error = abs(ratio - period_steps) / period_steps
        return error <= self.periodic_tolerance, error

    def _filter_by_periodic_support(
        self,
        peaks: np.ndarray,
        energy: np.ndarray,
        *,
        period: Optional[float],
        period_map: Optional[np.ndarray],
    ) -> np.ndarray:
        self.last_periodic_component_sizes = []
        if len(peaks) < self.periodic_min_chain_len:
            return np.array([], dtype=int)

        peaks = np.asarray(sorted(np.asarray(peaks, dtype=int)), dtype=int)
        n = len(peaks)
        adjacency: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                ok, _ = self._is_periodic_gap(
                    int(peaks[i]),
                    int(peaks[j]),
                    period=period,
                    period_map=period_map,
                )
                if ok:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        visited = np.zeros(n, dtype=bool)
        kept: list[int] = []
        component_sizes: list[int] = []

        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            component: list[int] = []
            visited[start] = True
            while stack:
                idx = stack.pop()
                component.append(idx)
                for nxt in adjacency[idx]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)

            if len(component) >= self.periodic_min_chain_len:
                kept.extend(int(peaks[i]) for i in component)
            component_sizes.append(len(component))

        self.last_periodic_component_sizes = component_sizes
        return np.asarray(sorted(set(kept)), dtype=int)

    def _threshold(self, energy: np.ndarray) -> float:
        finite = energy[np.isfinite(energy)]
        if len(finite) == 0:
            return np.inf
        return float(np.median(finite) + self.wt_threshold * np.std(finite))

    @staticmethod
    def _safe_slice(values: np.ndarray, indices: np.ndarray) -> list[float]:
        if len(indices) == 0:
            return []
        return [float(values[int(i)]) for i in indices if 0 <= int(i) < len(values)]

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

        if len(x) < 10:
            self.last_candidates = []
            return []

        if period is not None and np.isfinite(period) and period > 0:
            self.period = float(period)

        pm = None
        if period_map is not None:
            pm = np.asarray(period_map, dtype=float)
            if len(pm) != len(x):
                pm = None

        result = self.core.transform(x)
        energy = np.asarray(result.energy, dtype=float)
        threshold = self._threshold(energy)

        self.last_energy = energy
        self.last_threshold = float(threshold)

        energy_for_peaks = energy.copy()
        mask = None
        if active_mask is not None:
            candidate_mask = np.asarray(active_mask, dtype=bool)
            if len(candidate_mask) == len(energy_for_peaks):
                mask = candidate_mask

        margin_samples = max(0, int(round(self.edge_margin / self.dt)))
        if margin_samples > 0 and len(energy_for_peaks) > 2 * margin_samples:
            energy_for_peaks[:margin_samples] = -np.inf
            energy_for_peaks[-margin_samples:] = -np.inf
        elif margin_samples > 0:
            energy_for_peaks[:] = -np.inf

        # min_distance = self._distance_samples(period)
        raw_distance = self._raw_peak_distance_samples(pm)

        primary_peaks, _ = find_peaks(
            energy_for_peaks,
            height=float(threshold),
            # distance=min_distance,
            distance=raw_distance,
        )

        finite_energy = energy[np.isfinite(energy)]
        median_energy = float(np.median(finite_energy)) if len(finite_energy) else 0.0
        relaxed_threshold = median_energy + self.rescue_threshold_factor * (float(threshold) - median_energy)
        relaxed_threshold = min(float(threshold), float(relaxed_threshold))
        self.last_relaxed_threshold = float(relaxed_threshold)

        rescue_peaks, _ = find_peaks(
            energy_for_peaks,
            height=relaxed_threshold,
            # distance=min_distance,
            distance=raw_distance,
        )

        all_peaks = np.unique(np.concatenate([primary_peaks, rescue_peaks])).astype(int)
        refined_pm = self._refine_period_map(
            x,
            all_peaks,
            energy,
            pm,
            reference_threshold=relaxed_threshold,
        )
        self.last_period_map = None if refined_pm is None else np.asarray(refined_pm, dtype=float)

        if mask is not None:
            combined_mask = mask.copy()
            if refined_pm is not None and len(refined_pm) == len(combined_mask):
                combined_mask |= np.isfinite(refined_pm)
            all_peaks = all_peaks[combined_mask[all_peaks]]

        peaks = self._nms_by_energy(
            all_peaks,
            energy,
            period=period,
            period_map=refined_pm,
        )
        nms_peak_count = len(peaks)
        if self.use_periodic_support_filter:
            peaks = self._filter_by_periodic_support(
                peaks,
                energy,
                period=period,
                period_map=refined_pm,
            )
        periodic_peak_count = len(peaks)
        periodic_peaks = np.asarray(peaks, dtype=int)

        self.last_primary_peaks = np.asarray(primary_peaks, dtype=int)
        self.last_rescue_peaks = np.asarray(rescue_peaks, dtype=int)
        self.last_all_peaks = np.asarray(all_peaks, dtype=int)
        self.last_periodic_peaks = periodic_peaks

        candidates: list[CrashCandidate] = []
        for p in peaks:
            local_period = self._local_period(int(p), period, refined_pm)
            local_distance = self._local_distance_samples(int(p), period, refined_pm)
            score = float(max(energy[p] - relaxed_threshold, 1e-6))
            candidates.append(
                CrashCandidate(
                    index=int(p),
                    time=int(p) * self.dt,
                    score=score,
                    direction="unknown",
                    channel=channel,
                    method="wavelet_energy",
                    meta={
                        "wavelet_energy": float(energy[p]),
                        "wavelet_threshold": float(threshold),
                        "wavelet_relaxed_threshold": float(relaxed_threshold),
                        "raw_distance_samples": int(raw_distance),
                        "local_distance_samples": int(local_distance),
                        "local_period": float(local_period),
                        "response_scale": int(result.response_scale),
                    },
                )
            )

        candidates.sort(key=lambda c: c.index)
        self.last_candidates = candidates

        if debug:
            selected_periods = [c.meta.get("local_period") for c in candidates]
            primary_periods = self._safe_slice(pm, primary_peaks) if pm is not None else []
            print(
                f"[WaveletEnergy] channel={channel}, primary={len(primary_peaks)}, "
                f"rescue={len(rescue_peaks)}, raw_all={len(all_peaks)}, nms={nms_peak_count}, "
                f"periodic={periodic_peak_count}, "
                f"accepted={len(candidates)}, "
                f"threshold={threshold:.4f}, relaxed={relaxed_threshold:.4f}, "
                f"raw_distance={raw_distance}, global_distance={self._distance_samples(period)}, "
                f"local_nms={self.use_local_period_nms}, "
                f"periodic_filter={self.use_periodic_support_filter}, response_scale={result.response_scale}"
            )
            if self.refine_period_from_events:
                print(
                    f"[WaveletEnergy] period refinement seeds={len(self.last_period_seed_peaks)}, "
                    f"times_ms={np.round(self.last_period_seed_peaks * self.dt * 1e3, 3).tolist()}"
                )
            if self.use_periodic_support_filter:
                print(
                    f"[WaveletEnergy] periodic component sizes: "
                    f"{self.last_periodic_component_sizes}"
                )
            if selected_periods:
                print(
                    f"[WaveletEnergy] accepted local periods (ms): "
                    f"{np.round(np.array(selected_periods, dtype=float) * 1e3, 3).tolist()}"
                )
            if primary_periods:
                print(
                    f"[WaveletEnergy] primary peak periods sample (ms): "
                    f"{np.round(np.array(primary_periods[:20], dtype=float) * 1e3, 3).tolist()}"
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


def _make_wavelet_detector(
    dt: float,
    period: float,
    wt_threshold: float = 1.0,
    crash_time_min: float = 10e-6,
    crash_time_max: float = 100e-6,
    percentile_threshold: float = 98.0,
    min_period: float = 0.25e-3,
    wavelet_name: str = "gaus1",
    edge_margin: float = 0.5e-3,
    distance_factor: float = 0.55,
    rescue_threshold_factor: float = 0.55,
    raw_peak_distance: float = 0.15e-3,
    use_local_period_nms: bool = True,
    use_periodic_support_filter: bool = True,
    periodic_min_chain_len: int = 2,
    periodic_tolerance: float = 0.40,
    periodic_suppression_factor: float = 0.50,
    periodic_max_skip: int = 6,
    refine_period_from_events: bool = True,
    period_seed_merge_distance: float = 0.35e-3,
    period_seed_score_threshold: float = 2.25,
    period_seed_jump_threshold: float = 0.35,
    period_refinement_min_period: float = 0.75e-3,
) -> WaveletEnergyDetector:
    return WaveletEnergyDetector(
        dt=dt,
        period=period,
        crash_time_min=crash_time_min,
        crash_time_max=crash_time_max,
        percentile_threshold=percentile_threshold,
        min_period=min_period,
        wavelet_name=wavelet_name,
        wt_threshold=wt_threshold,
        edge_margin=edge_margin,
        distance_factor=distance_factor,
        rescue_threshold_factor=rescue_threshold_factor,
        raw_peak_distance=raw_peak_distance,
        use_local_period_nms=use_local_period_nms,
        use_periodic_support_filter=use_periodic_support_filter,
        periodic_min_chain_len=periodic_min_chain_len,
        periodic_tolerance=periodic_tolerance,
        periodic_suppression_factor=periodic_suppression_factor,
        periodic_max_skip=periodic_max_skip,
        refine_period_from_events=refine_period_from_events,
        period_seed_merge_distance=period_seed_merge_distance,
        period_seed_score_threshold=period_seed_score_threshold,
        period_seed_jump_threshold=period_seed_jump_threshold,
        period_refinement_min_period=period_refinement_min_period,
    )


def detect_on_shot(
    source_dir,
    filename,
    logger,
    path_to_load="./data/model_data",
    channel_name="SXR 50 mkm",
    wt_threshold=1.0,
    downsample=1,
    multichannel=False,
    channels: Optional[Iterable[str]] = None,
    min_channels=3,
    coincidence_window=0.3e-3,
    plot=True,
    debug=False,
    crash_time_min=10e-6,
    crash_time_max=100e-6,
    percentile_threshold=98.0,
    min_period=0.25e-3,
    wavelet_name="mexh",
    edge_margin=0.5e-3,
    wavelet_distance_factor=0.55,
    wavelet_rescue_threshold_factor=0.55,
    wavelet_raw_peak_distance=0.15e-3,
    wavelet_use_local_period_nms=True,
    wavelet_use_periodic_support_filter=True,
    wavelet_periodic_min_chain_len=3,
    wavelet_periodic_tolerance=0.40,
    wavelet_periodic_suppression_factor=0.50,
    wavelet_periodic_max_skip=6,
    wavelet_refine_period_from_events=True,
    wavelet_period_seed_merge_distance=0.35e-3,
    wavelet_period_seed_score_threshold=2.25,
    wavelet_period_seed_jump_threshold=0.35,
    wavelet_period_refinement_min_period=0.75e-3,
):
    prepared = prepare_shot_for_detection(
        source_dir=source_dir,
        filename=filename,
        logger=logger,
        channel_name=channel_name,
        channels=channels,
    )
    if prepared is None:
        return None

    ds = max(1, int(downsample))
    dt_ds = prepared.dt * ds

    def factory() -> WaveletEnergyDetector:
        return _make_wavelet_detector(
            dt=dt_ds,
            period=prepared.estimated_period,
            wt_threshold=wt_threshold,
            crash_time_min=crash_time_min,
            crash_time_max=crash_time_max,
            percentile_threshold=percentile_threshold,
            min_period=min_period,
            wavelet_name=wavelet_name,
            edge_margin=edge_margin,
            distance_factor=wavelet_distance_factor,
            rescue_threshold_factor=wavelet_rescue_threshold_factor,
            raw_peak_distance=wavelet_raw_peak_distance,
            use_local_period_nms=wavelet_use_local_period_nms,
            use_periodic_support_filter=wavelet_use_periodic_support_filter,
            periodic_min_chain_len=wavelet_periodic_min_chain_len,
            periodic_tolerance=wavelet_periodic_tolerance,
            periodic_suppression_factor=wavelet_periodic_suppression_factor,
            periodic_max_skip=wavelet_periodic_max_skip,
            refine_period_from_events=wavelet_refine_period_from_events,
            period_seed_merge_distance=wavelet_period_seed_merge_distance,
            period_seed_score_threshold=wavelet_period_seed_score_threshold,
            period_seed_jump_threshold=wavelet_period_seed_jump_threshold,
            period_refinement_min_period=wavelet_period_refinement_min_period,
        )

    if multichannel:
        return run_multichannel_detector(
            prepared=prepared,
            detector_factory=factory,
            logger=logger,
            downsample=ds,
            coincidence_window=coincidence_window,
            min_channels=min_channels,
            plot=plot,
            mode="wavelet_multichannel",
            debug=debug,
        )

    detector = factory()
    return run_single_channel_detector(
        prepared=prepared,
        detector=detector,
        logger=logger,
        downsample=ds,
        plot=plot,
        mode="wavelet",
        debug=debug,
    )


if __name__ == "__main__":
    pass
