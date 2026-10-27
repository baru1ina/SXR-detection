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


ML_MINIMUM_SNR = 2.0


def _select_display_channel(prepared, requested_channel_name: str, logger) -> str:
    """Use the requested trace only when it passes the same SNR gate as ML."""

    available_channels = set(prepared.shot.channel_names)
    channel_snr = prepared.channel_snr or {}
    requested_snr = channel_snr.get(requested_channel_name)
    reference_snr = channel_snr.get(prepared.channel_name)
    requested_is_usable = (
        requested_channel_name in available_channels
        and requested_snr is not None
        and np.isfinite(requested_snr)
        and float(requested_snr) >= ML_MINIMUM_SNR
    )
    if requested_is_usable:
        return requested_channel_name

    if requested_channel_name not in available_channels:
        reason = "channel is absent from the SHT file"
    elif requested_snr is None or not np.isfinite(requested_snr):
        reason = "its SNR could not be estimated"
    else:
        reason = (
            f"SNR={float(requested_snr):.2f} is below the required "
            f"SNR={ML_MINIMUM_SNR:.2f}"
        )
    reference_snr_text = (
        "unknown"
        if reference_snr is None or not np.isfinite(reference_snr)
        else f"{float(reference_snr):.2f}"
    )
    logger.warning(
        f"Requested display SXR {requested_channel_name!r} was rejected: {reason}. "
        f"Detected events will be plotted on the most reliable available channel "
        f"{prepared.channel_name!r} (SNR={reference_snr_text}). This avoids "
        "overlaying events found in other SXR channels on a noise-only trace."
    )
    return prepared.channel_name


def _make_ml_detector(
    dt: float,
    model_path: str,
    payload: dict,
    probability_threshold: Optional[float] = None,
    event_merge_window_s: Optional[float] = None,
    event_merge_strategy: str = "adaptive_period",
    event_merge_period_fraction: float = 0.50,
    event_merge_min_radius_s: float = 0.12e-3,
    event_merge_max_radius_s: float = 0.80e-3,
    event_merge_confidence_threshold: float = 0.20,
):
    """Build a detector from the model, with an optional runtime merge window."""
    return FeatureMLCrashDetector(
        dt=dt,
        downsample=int(payload["downsample"]),
        model_path=model_path,
        payload=payload,
        probability_threshold=probability_threshold,
        event_merge_window_s=event_merge_window_s,
        event_merge_strategy=event_merge_strategy,
        event_merge_period_fraction=event_merge_period_fraction,
        event_merge_min_radius_s=event_merge_min_radius_s,
        event_merge_max_radius_s=event_merge_max_radius_s,
        event_merge_confidence_threshold=event_merge_confidence_threshold,
    )


def detect_on_shot(
    source_dir,
    filename,
    logger,
    channel_name="SXR 50 mkm",
    model_path: Optional[str] = None,
    probability_threshold: Optional[float] = None,
    event_merge_window_s: Optional[float] = None,
    event_merge_strategy: str = "adaptive_period",
    event_merge_period_fraction: float = 0.50,
    event_merge_min_radius_s: float = 0.12e-3,
    event_merge_max_radius_s: float = 0.80e-3,
    event_merge_confidence_threshold: float = 0.20,
    multichannel=False,
    min_channels=2,
    plot=True,
    debug=False,
):
    """Run feature-ML using preprocessing and proposals saved with the model."""

    requested_channel_name = channel_name
    model_path = model_path or DEFAULT_MODEL_PATH
    payload = load_feature_ml_payload(model_path)
    proposal_config = payload["proposal_config"]
    proposal_channels = list(proposal_config["proposal_channels"])
    loaded_sxr_channels = list(
        dict.fromkeys([*proposal_channels, requested_channel_name])
    )
    downsample = int(payload["downsample"])
    feature_profile = get_feature_map_profile(payload["feature_map_profile"])
    prepared = prepare_shot_for_detection(
        source_dir=source_dir,
        filename=filename,
        logger=logger,
        channel_name=channel_name,
        channels=loaded_sxr_channels,
        require_sawtooth=False,
        auto_reference_sxr=True,
        minimum_snr=ML_MINIMUM_SNR,
        feature_channel_profile=feature_profile,
        enable_plasma_plateau_gate=True,
        plateau_level_fraction=float(proposal_config["plateau_level_fraction"]),
        plateau_hold_s=float(proposal_config["plateau_hold_s"]),
        period_min_s=0.5e-3,
    )
    if prepared is None:
        return None

    detector = _make_ml_detector(
        dt=prepared.dt * downsample,
        model_path=model_path,
        payload=payload,
        probability_threshold=probability_threshold,
        event_merge_window_s=event_merge_window_s,
        event_merge_strategy=event_merge_strategy,
        event_merge_period_fraction=event_merge_period_fraction,
        event_merge_min_radius_s=event_merge_min_radius_s,
        event_merge_max_radius_s=event_merge_max_radius_s,
        event_merge_confidence_threshold=event_merge_confidence_threshold,
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
    display_channel_name = _select_display_channel(
        prepared,
        requested_channel_name,
        logger,
    )
    logger.info(
        f"ML reference SXR: {prepared.channel_name}; "
        f"display SXR: {display_channel_name}"
    )
    logger.newline()
    if plot:
        plot_with_crashes(
            prepared.shot,
            crash_times,
            channel_name=display_channel_name,
            mode=mode,
        )
    return crash_times
