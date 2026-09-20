from typing import Iterable, Optional

import numpy as np

from config.channels import MULTICHANNEL_FEATURE_PROFILE
from config.path import DEFAULT_MODEL_PATH, SXR_CHANNELS
from src.detection.ml.models.feature_ml_detector import FeatureMLCrashDetector
from src.detection.ml.models.hybrid_proposal_detector import HybridProposalDetector
from src.detection.models.cpd_detector import CPDDetector
from src.detection.models.posr_detector import WaveletPOSRDetector
from src.detection.utils.pipeline_utils import prepare_shot_for_detection
from src.visualization.plots import plot_with_crashes


def _default_sigma(period: float | None) -> float:
    if period is None or not np.isfinite(period) or period <= 0:
        return 80e-6
    return float(np.clip(0.03 * period, 30e-6, 200e-6))


def _make_ml_detector(
    dt: float,
    downsample: int,
    period: float | None,
    model_path: str,
    probability_threshold: Optional[float] = None,
    proposal_sigma: Optional[float] = None,
    proposal_threshold: float = 3.0,
    proposal_score_threshold: float = 0.5,
    use_cpd: bool = True,
    cpd_penalty: float = 3.0,
    cpd_model: str = "rbf",
    proposal_channels: Optional[Iterable[str]] = None,
    coincidence_window_s: float = 0.15e-3,
):
    if proposal_sigma is not None:
        raise ValueError(
            "A custom proposal sigma is not supported by the trained v5 model; "
            "use the saved period-dependent POSR scale"
        )
    def detector_factory(_dt, _period):
        posr_detector = WaveletPOSRDetector(
            dt=dt,
            sigma=_default_sigma(period),
            threshold=proposal_threshold,
            score_threshold=proposal_score_threshold,
        )
        if not use_cpd:
            return posr_detector
        return HybridProposalDetector(
            posr_detector=posr_detector,
            cpd_detector=CPDDetector(dt=dt, penalty=cpd_penalty, model=cpd_model),
        )

    proposal_channels = list(dict.fromkeys(proposal_channels or SXR_CHANNELS))
    proposal_config = {
        "threshold": proposal_threshold,
        "score_threshold": proposal_score_threshold,
        "min_distance_factor": 0.45,
        "use_cpd": use_cpd,
        "cpd_penalty": cpd_penalty,
        "cpd_model": cpd_model,
        "proposal_channels": proposal_channels,
        "coincidence_window_s": float(coincidence_window_s),
    }
    return FeatureMLCrashDetector(
        dt=dt,
        downsample=downsample,
        model_path=model_path,
        proposal_detector_factory=detector_factory,
        proposal_config=proposal_config,
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
    coincidence_window=0.15e-3,
    plot=True,
    debug=False,
):
    """Use the v5 classifier on multi-SXR proposals and aligned diagnostics."""

    model_path = model_path or DEFAULT_MODEL_PATH

    proposal_channels = list(dict.fromkeys(channels or SXR_CHANNELS))
    prepared = prepare_shot_for_detection(
        source_dir=source_dir,
        filename=filename,
        logger=logger,
        channel_name=channel_name,
        channels=proposal_channels,
        require_sawtooth=False,
        auto_reference_sxr=True,
        minimum_snr=2.0,
        feature_channel_profile=MULTICHANNEL_FEATURE_PROFILE,
    )
    if prepared is None:
        return None

    ds = int(downsample)
    if ds < 1:
        raise ValueError("downsample must be at least 1")
    detector = _make_ml_detector(
        dt=prepared.dt * ds,
        downsample=ds,
        period=prepared.estimated_period,
        model_path=model_path,
        probability_threshold=probability_threshold,
        proposal_sigma=proposal_sigma,
        proposal_threshold=proposal_threshold,
        proposal_score_threshold=proposal_score_threshold,
        use_cpd=use_cpd,
        cpd_penalty=cpd_penalty,
        cpd_model=cpd_model,
        proposal_channels=proposal_channels,
        coincidence_window_s=coincidence_window,
    )

    indices = detector.detect(
        prepared,
        debug=debug,
        min_channels=min_channels if multichannel else 1,
    ) * ds
    mode = "feature_ml_multichannel" if multichannel else "feature_ml"

    indices = indices[(indices >= 0) & (indices < len(prepared.time_plasma))]
    crash_times = prepared.time_plasma[indices]
    logger.info(f"Feature-ML detections: {len(crash_times)}")
    logger.info(f"Detected reset times: {crash_times}")
    logger.newline()
    if plot:
        plot_with_crashes(
            prepared.shot,
            crash_times,
            channel_name=prepared.channel_name,
            mode=mode,
        )
    return crash_times
