from typing import Iterable, Optional

import numpy as np

from src.detection.ml.models.hybrid_proposal_detector import HybridProposalDetector
from src.detection.models.cpd_detector import CPDDetector
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
    probability_threshold: Optional[float] = None,
    proposal_sigma: Optional[float] = None,
    proposal_threshold: float = 3.0,
    proposal_score_threshold: float = 0.5,
    use_cpd: bool = True,
    cpd_penalty: float = 3.0,
    cpd_model: str = "rbf",
):
    posr_detector = WaveletPOSRDetector(
        dt=dt,
        sigma=_default_sigma(period) if proposal_sigma is None else proposal_sigma,
        threshold=proposal_threshold,
        score_threshold=proposal_score_threshold,
    )
    if use_cpd:
        proposal_detector = HybridProposalDetector(
            posr_detector=posr_detector,
            cpd_detector=CPDDetector(
                dt=dt,
                penalty=cpd_penalty,
                model=cpd_model,
            ),
        )
    else:
        proposal_detector = posr_detector
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
    probability_threshold: Optional[float] = None,
    proposal_sigma: Optional[float] = None,
    proposal_threshold=3.0,
    proposal_score_threshold=0.5,
    use_cpd=True,
    cpd_penalty=3.0,
    cpd_model="rbf",
    downsample=1,
    multichannel=False,
    channels: Optional[Iterable[str]] = None,
    min_channels=2,
    coincidence_window=0.3e-3,
    plot=True,
    debug=False,
):
    """Feature-ML crash detector pipeline.

    The ML model classifies POSR candidates and raw CPD breakpoints.  CPD score
    filtering, fallback, period suppression and event limits are disabled here.
    """
    if model_path is None:
        raise ValueError("ml_pipeline requires model_path='...joblib'.")

    prepared = prepare_shot_for_detection(
        source_dir=source_dir,
        filename=filename,
        logger=logger,
        channel_name=channel_name,
        channels=channels,
        require_sawtooth=False,
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
                use_cpd=use_cpd,
                cpd_penalty=cpd_penalty,
                cpd_model=cpd_model,
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
        use_cpd=use_cpd,
        cpd_penalty=cpd_penalty,
        cpd_model=cpd_model,
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
