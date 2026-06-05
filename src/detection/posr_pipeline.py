from typing import Iterable, Optional

import numpy as np

from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)


def _default_sigma(period: float | None) -> float:
    if period is None or not np.isfinite(period) or period <= 0:
        return 80e-6
    return float(np.clip(0.03 * period, 30e-6, 200e-6))


def _make_posr_detector(
    dt: float,
    period: float | None,
    sigma: Optional[float] = None,
    frame: Optional[float] = None,
    threshold: float = 6.0,
    score_threshold: float = 2.5,
    min_distance_factor: float = 0.45,
    min_jump_to_amp: float = 0.0,
    crash_time_min: Optional[float] = None,
    crash_time_max: Optional[float] = None,
    wavelet_name: str = "gaus1"
):
    return WaveletPOSRDetector(
        dt=dt,
        sigma=_default_sigma(period) if sigma is None else sigma,
        frame=frame,
        threshold=threshold,
        score_threshold=score_threshold,
        min_distance_factor=min_distance_factor,
        min_jump_to_amp=min_jump_to_amp,
        crash_time_min=crash_time_min,
        crash_time_max=crash_time_max,
        wavelet_name=wavelet_name,
    )


def detect_on_shot(
    source_dir,
    filename,
    logger,
    path_to_load="./data/model_data",
    channel_name="SXR 50 mkm",
    sigma: Optional[float] = None,
    frame: Optional[float] = None,
    threshold=6.0,
    score_threshold=2.5,
    min_distance_factor=0.45,
    min_jump_to_amp=0.0,
    crash_time_min: Optional[float] = None,
    crash_time_max: Optional[float] = None,
    wavelet_name="gaus1",
    downsample=1,
    multichannel=False,
    channels: Optional[Iterable[str]] = None,
    min_channels=2,
    coincidence_window=0.3e-3,
    plot=True,
    debug=False,
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

    def factory() -> WaveletPOSRDetector:
        return _make_posr_detector(
            dt=dt_ds,
            period=prepared.estimated_period,
            sigma=sigma,
            frame=frame,
            threshold=threshold,
            score_threshold=score_threshold,
            min_distance_factor=min_distance_factor,
            min_jump_to_amp=min_jump_to_amp,
            crash_time_min=crash_time_min,
            crash_time_max=crash_time_max,
            wavelet_name=wavelet_name,
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
            mode="wavelet_posr_multichannel",
            debug=debug,
        )

    detector = factory()
    return run_single_channel_detector(
        prepared=prepared,
        detector=detector,
        logger=logger,
        downsample=ds,
        plot=plot,
        mode="wavelet_posr",
        debug=debug,
    )

