from typing import Iterable, Optional
from scipy.signal import find_peaks

import numpy as np

from src.detection.models.wavelet_core import WaveletEdgeCore
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)


class WaveletEnergyDetector:
    def __init__(
        self,
        dt: float,
        period: float,
        crash_time_min: float = 10e-6,
        crash_time_max: float = 100e-6,
        percentile_threshold: float = 98.0,
        min_period: float = 0.25e-3,
        wavelet_name: str = "gaus1",
        wt_threshold: float = 1.0,
        edge_margin: float = 0.5e-3,
        distance_factor: float = 0.55,
        rescue_threshold_factor: float = 0.55,
        raw_peak_distance: float = 0.15e-3,
        use_local_period_nms: bool = True,
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
        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool)
            if len(mask) == len(energy_for_peaks):
                energy_for_peaks[~mask] = -np.inf

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
        peaks = self._nms_by_energy(
            all_peaks,
            energy,
            period=period,
            period_map=pm,
        )

        self.last_primary_peaks = np.asarray(primary_peaks, dtype=int)
        self.last_rescue_peaks = np.asarray(rescue_peaks, dtype=int)
        self.last_all_peaks = np.asarray(all_peaks, dtype=int)

        candidates: list[CrashCandidate] = []
        for p in peaks:
            local_period = self._local_period(int(p), period, pm)
            local_distance = self._local_distance_samples(int(p), period, pm)
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
                f"rescue={len(rescue_peaks)}, raw_all={len(all_peaks)}, accepted={len(candidates)}, "
                f"threshold={threshold:.4f}, relaxed={relaxed_threshold:.4f}, "
                f"raw_distance={raw_distance}, global_distance={self._distance_samples(period)}, "
                f"local_nms={self.use_local_period_nms}, response_scale={result.response_scale}"
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
    min_channels=2,
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
