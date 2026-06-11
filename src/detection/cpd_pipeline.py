from typing import Iterable, Optional, Type

from src.detection.models.cpd_detector import CPDDetector
from src.detection.models.cpd_features_detector import CPDFeaturesDetector
from src.detection.models.cpd_base import BaseCPDDetector
from src.detection.utils.pipeline_utils import (
    prepare_shot_for_detection,
    run_multichannel_detector,
    run_single_channel_detector,
)


def _make_cpd_detector(
    dt: float,
    detector_cls: Type[BaseCPDDetector],
    penalty: float = 8.0,
    model: str = "rbf",
    min_distance_factor: float = 0.65,
    score_threshold: float = 4.0,
    fallback_percentile: float = 99.5,
    max_candidates: Optional[int] = None,
    max_events_factor: float = 1.3,
    min_jump_to_amp: float = 0.05,
    candidate_margin_factor: float = 0.20,
) -> BaseCPDDetector:
    return detector_cls(
        dt=dt,
        penalty=penalty,
        model=model,
        min_distance_factor=min_distance_factor,
        score_threshold=score_threshold,
        fallback_percentile=fallback_percentile,
        max_candidates=max_candidates,
        max_events_factor=max_events_factor,
        min_jump_to_amp=min_jump_to_amp,
        candidate_margin_factor=candidate_margin_factor,
    )


def detect_on_shot(
    source_dir: str,
    filename: str,
    logger,
    path_to_load: str = "./data/model_data",
    channel_name: str = "SXR 50 mkm",
    penalty: float = 3.0,
    model: str = "rbf",
    score_threshold: float = 4.0,
    fallback_percentile: float = 99.5,
    downsample: int = 10,
    multichannel: bool = False,
    channels: Optional[Iterable[str]] = None,
    min_channels: int = 2,
    coincidence_window: float = 0.3e-3,
    plot: bool = True,
    debug: bool = False,
    use_features: bool = True,
    min_distance_factor: float = 0.65,
    min_jump_to_amp: float = 0.05,
    max_events_factor: float = 1.3,
    max_candidates: Optional[int] = None,
    candidate_margin_factor: float = 0.20,
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
    detector_cls: Type[BaseCPDDetector] = CPDFeaturesDetector if use_features else CPDDetector
    mode_name = "cpd_features" if use_features else "cpd"

    if debug:
        logger.info(
            f"{mode_name} settings: dt={prepared.dt:.3e}, downsample={ds}, "
            f"dt_ds={dt_ds:.3e}, period={prepared.estimated_period:.3e}, "
            f"penalty={penalty}, model={model}, score_threshold={score_threshold}, "
            f"min_distance_factor={min_distance_factor}, min_jump_to_amp={min_jump_to_amp}, "
            f"max_events_factor={max_events_factor}"
        )

    def detector_factory():
        return _make_cpd_detector(
            dt=dt_ds,
            detector_cls=detector_cls,
            penalty=penalty,
            model=model,
            min_distance_factor=min_distance_factor,
            score_threshold=score_threshold,
            fallback_percentile=fallback_percentile,
            max_candidates=max_candidates,
            max_events_factor=max_events_factor,
            min_jump_to_amp=min_jump_to_amp,
            candidate_margin_factor=candidate_margin_factor,
        )

    if multichannel:
        return run_multichannel_detector(
            prepared=prepared,
            detector_factory=detector_factory,
            logger=logger,
            downsample=ds,
            coincidence_window=coincidence_window,
            min_channels=min_channels,
            plot=plot,
            mode=f"{mode_name}_multichannel",
            debug=debug,
        )

    return run_single_channel_detector(
        prepared=prepared,
        detector=detector_factory(),
        logger=logger,
        downsample=ds,
        plot=plot,
        mode=mode_name,
        debug=debug,
    )


if __name__ == "__main__":
    pass
