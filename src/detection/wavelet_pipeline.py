from typing import Iterable, Optional
from scipy.signal import find_peaks

import numpy as np

from src.detection.models.wavelet_detector import WaveletSawtoothDetector
from src.detection.utils.common import CrashCandidate, candidates_to_indices
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)


class WaveletDetectorAdapter:
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
    ):
        self.dt = float(dt)
        self.edge_margin = float(edge_margin)

        self.distance_factor = float(distance_factor)
        self.rescue_threshold_factor = float(rescue_threshold_factor)

        self.detector = WaveletSawtoothDetector(
            dt=self.dt,
            period=period,
            period_map=None,
            crash_time_min=crash_time_min,
            crash_time_max=crash_time_max,
            percentile_threshold=percentile_threshold,
            min_period=min_period,
            wavelet_name=wavelet_name,
            wt_threshold=wt_threshold,
        )
        self.last_energy: Optional[np.ndarray] = None
        self.last_threshold: Optional[float] = None
        self.last_candidates: list[CrashCandidate] = []

    def _distance_samples(self, period: Optional[float]) -> int:
        p = period if period is not None and np.isfinite(period) and period > 0 else self.period
        return max(1, int(round(self.distance_factor * float(p) / self.dt)))

    @staticmethod
    def _nms_by_energy(peaks: np.ndarray, energy: np.ndarray, min_distance: int) -> np.ndarray:
        if len(peaks) == 0:
            return np.array([], dtype=int)
        peaks = np.asarray(peaks, dtype=int)
        order = sorted(peaks, key=lambda i: float(energy[i]), reverse=True)
        kept: list[int] = []
        for p in order:
            if all(abs(int(p) - int(k)) >= min_distance for k in kept):
                kept.append(int(p))
        return np.array(sorted(kept), dtype=int)

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
            self.detector.period = float(period)

        # Important: do not pass the full period_map into the old detector.  Its
        # median can be dominated by 1 ms boundary values and then the peak distance
        # becomes too small.  The period_map can still be used later for scoring in
        # other detectors; wavelet peak separation is intentionally global here.
        self.detector.period_map = None

        local_time = np.arange(len(x), dtype=float) * self.dt
        _, energy, threshold = self.detector.detect(
            x,
            local_time,
            plot_flag=False,
        )

        energy = np.asarray(energy, dtype=float)
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

        min_distance = self._distance_samples(period)

        primary_peaks, _ = find_peaks(
            energy_for_peaks,
            height=float(threshold),
            distance=min_distance,
        )

        finite_energy = energy[np.isfinite(energy)]
        if len(finite_energy) > 0:
            median_energy = float(np.median(finite_energy))
        else:
            median_energy = 0.0

        relaxed_threshold = median_energy + self.rescue_threshold_factor * (float(threshold) - median_energy)
        relaxed_threshold = min(float(threshold), float(relaxed_threshold))

        rescue_peaks, _ = find_peaks(
            energy_for_peaks,
            height=relaxed_threshold,
            distance=min_distance,
        )

        all_peaks = np.unique(np.concatenate([primary_peaks, rescue_peaks])).astype(int)
        peaks = self._nms_by_energy(all_peaks, energy, min_distance=min_distance)

        candidates: list[CrashCandidate] = []
        for p in peaks:
            score = float(max(energy[p] - relaxed_threshold, 1e-6))
            candidates.append(
                CrashCandidate(
                    index=int(p),
                    time=int(p) * self.dt,
                    score=score,
                    direction="unknown",
                    channel=channel,
                    method="wavelet",
                    meta={
                        "wavelet_energy": float(energy[p]),
                        "wavelet_threshold": float(threshold),
                        "wavelet_relaxed_threshold": float(relaxed_threshold),
                        "distance_samples": int(min_distance),
                    },
                )
            )

        candidates.sort(key=lambda c: c.index)
        self.last_candidates = candidates

        if debug:
            print(
                f"[Wavelet] channel={channel}, primary={len(primary_peaks)}, "
                f"rescue={len(rescue_peaks)}, accepted={len(candidates)}, "
                f"threshold={threshold:.4f}, relaxed={relaxed_threshold:.4f}, "
                f"distance={min_distance}"
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
    wavelet_name: str = "mexh",
    edge_margin: float = 0.5e-3,
    distance_factor: float = 0.55,
    rescue_threshold_factor: float = 0.55,
) -> WaveletDetectorAdapter:
    return WaveletDetectorAdapter(
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

    def factory() -> WaveletDetectorAdapter:
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
