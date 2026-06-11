from typing import Iterable, Optional

import numpy as np

from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.ml.models.feature_ml_detector import FeatureMLCrashDetector
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)


def _default_sigma(period: float | None) -> float:
    if period is None or not np.isfinite(period) or period <= 0:
        return 80e-6
    return float(np.clip(0.03 * period, 30e-6, 200e-6))


def _make_ml_detector(
    dt: float,
    period: float | None,
    model_path: str,
    probability_threshold: float = 0.5,
    proposal_sigma: Optional[float] = None,
    proposal_threshold: float = 5.0,
    proposal_score_threshold: float = 1.5,
):
    proposal_detector = WaveletPOSRDetector(
        dt=dt,
        sigma=_default_sigma(period) if proposal_sigma is None else proposal_sigma,
        threshold=proposal_threshold,
        score_threshold=proposal_score_threshold,
    )
    return FeatureMLCrashDetector(
        dt=dt,
        model_path=model_path,
        proposal_detector=proposal_detector,
        probability_threshold=probability_threshold,
    )


def detect_on_shot(
    source_dir,
    filename,
    logger,
    path_to_load="./data/model_data",
    channel_name="SXR 50 mkm",
    model_path: Optional[str] = None,
    probability_threshold=0.5,
    proposal_sigma: Optional[float] = None,
    proposal_threshold=5.0,
    proposal_score_threshold=1.5,
    downsample=1,
    multichannel=False,
    channels: Optional[Iterable[str]] = None,
    min_channels=2,
    coincidence_window=0.3e-3,
    plot=True,
    debug=False,
):
    """Feature-ML crash detector pipeline.

    The ML model classifies candidates proposed by DoG/POSR.  Train and save the
    model with FeatureMLTrainer before using this pipeline.
    """
    if model_path is None:
        raise ValueError("ml_pipeline requires model_path='...joblib'.")

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

    if multichannel:
        return run_multichannel_detector(
            prepared=prepared,
            detector_factory=lambda: _make_ml_detector(
                dt=dt_ds,
                period=prepared.estimated_period,
                model_path=model_path,
                probability_threshold=probability_threshold,
                proposal_sigma=proposal_sigma,
                proposal_threshold=proposal_threshold,
                proposal_score_threshold=proposal_score_threshold,
            ),
            logger=logger,
            downsample=ds,
            coincidence_window=coincidence_window,
            min_channels=min_channels,
            plot=plot,
            mode="feature_ml_multichannel",
            debug=debug,
        )

    detector = _make_ml_detector(
        dt=dt_ds,
        period=prepared.estimated_period,
        model_path=model_path,
        probability_threshold=probability_threshold,
        proposal_sigma=proposal_sigma,
        proposal_threshold=proposal_threshold,
        proposal_score_threshold=proposal_score_threshold,
    )
    return run_single_channel_detector(
        prepared=prepared,
        detector=detector,
        logger=logger,
        downsample=ds,
        plot=plot,
        mode="feature_ml",
        debug=debug,
    )
