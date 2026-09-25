from typing import Optional

import numpy as np

from config.channels import get_feature_map_profile
from config.path import DEFAULT_MODEL_PATH
from src.detection.ml.models.feature_ml_detector import (
    FeatureMLCrashDetector,
    load_feature_ml_payload,
)
from src.detection.utils.pipeline_utils import prepare_shot_for_detection
from src.visualization.plots import plot_with_crashes


def _make_ml_detector(
    dt: float,
    model_path: str,
    payload: dict,
    probability_threshold: Optional[float] = None,
):
    """Build a detector exclusively from settings stored in the model."""
    return FeatureMLCrashDetector(
        dt=dt,
        downsample=int(payload["downsample"]),
        model_path=model_path,
        payload=payload,
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
    multichannel=False,
    min_channels=2,
    plot=True,
    debug=False,
):
    """Run feature-ML using preprocessing and proposals saved with the model."""

    del path_to_load  # Kept for compatibility with the common detector interface.
    model_path = model_path or DEFAULT_MODEL_PATH
    payload = load_feature_ml_payload(model_path)
    proposal_config = payload["proposal_config"]
    proposal_channels = list(proposal_config["proposal_channels"])
    downsample = int(payload["downsample"])
    feature_profile = get_feature_map_profile(payload["feature_map_profile"])
    prepared = prepare_shot_for_detection(
        source_dir=source_dir,
        filename=filename,
        logger=logger,
        channel_name=channel_name,
        channels=proposal_channels,
        require_sawtooth=False,
        auto_reference_sxr=True,
        minimum_snr=2.0,
        feature_channel_profile=feature_profile,
        enable_plasma_plateau_gate=True,
        plateau_level_fraction=float(proposal_config["plateau_level_fraction"]),
        plateau_hold_s=float(proposal_config["plateau_hold_s"]),
    )
    if prepared is None:
        return None

    detector = _make_ml_detector(
        dt=prepared.dt * downsample,
        model_path=model_path,
        payload=payload,
        probability_threshold=probability_threshold,
    )

    indices = detector.detect(
        prepared,
        debug=debug,
        min_channels=min_channels if multichannel else 1,
    ) * downsample
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
