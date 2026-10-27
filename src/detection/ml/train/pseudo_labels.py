import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from config.path import DEFAULT_PSEUDO_LABELS_PATH, SXR_CHANNELS
from src.detection.ml.train.manifest import ShotRecord, ShotSplit, build_shot_manifest, stratified_shot_split
from src.detection.utils.multichannel import MultiChannelVotingDetector


PSEUDO_LABEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WaveletLabelConfig:
    channels: tuple[str, ...] = tuple(SXR_CHANNELS)
    min_snr: float = 2.0
    min_support_channels: int = 2
    coincidence_window_s: float = 0.15e-3
    downsample: int = 5
    wt_threshold: float = 1.0
    wavelet_name: str = "mexh"
    crash_time_min_s: float = 10e-6
    crash_time_max_s: float = 100e-6
    edge_margin_s: float = 0.5e-3
    distance_factor: float = 0.55
    rescue_threshold_factor: float = 0.55
    raw_peak_distance_s: float = 0.15e-3
    periodic_min_chain_len: int = 3
    periodic_tolerance: float = 0.40

    def __post_init__(self):
        channels = tuple(dict.fromkeys(self.channels))
        if not channels:
            raise ValueError("At least one SXR teacher channel is required")
        if self.min_snr <= 0 or not np.isfinite(self.min_snr):
            raise ValueError("min_snr must be positive and finite")
        if self.min_support_channels < 2:
            raise ValueError("min_support_channels must be at least 2")
        if self.coincidence_window_s <= 0 or not np.isfinite(self.coincidence_window_s):
            raise ValueError("coincidence_window_s must be positive and finite")
        if self.downsample < 1:
            raise ValueError("downsample must be at least 1")
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class PseudoLabelEvent:
    time_s: float
    score: float


@dataclass(frozen=True)
class PseudoLabeledShot:
    shot_id: str
    category: str
    relative_path: str
    status: str
    channel: str
    events: tuple[PseudoLabelEvent, ...]
    error: Optional[str] = None


@dataclass(frozen=True)
class PseudoLabelDataset:
    source_dir: str
    config: WaveletLabelConfig
    train: tuple[PseudoLabeledShot, ...]
    validation: tuple[PseudoLabeledShot, ...]
    schema_version: int = PSEUDO_LABEL_SCHEMA_VERSION

    def summary(self) -> dict[str, Any]:
        shots = self.train + self.validation
        return {
            "train_shots": len(self.train),
            "validation_shots": len(self.validation),
            "events": sum(len(shot.events) for shot in shots),
            "statuses": dict(sorted(Counter(shot.status for shot in shots).items())),
        }


DetectorFactory = Callable[[float, float, WaveletLabelConfig], object]
PrepareShotFn = Callable[..., object]


def _default_detector_factory(
    dt: float,
    period: float,
    config: WaveletLabelConfig,
):
    from src.detection.wavelet_pipeline import _make_wavelet_detector

    return _make_wavelet_detector(
        dt=dt,
        period=period,
        wt_threshold=config.wt_threshold,
        wavelet_name=config.wavelet_name,
        crash_time_min=config.crash_time_min_s,
        crash_time_max=config.crash_time_max_s,
        edge_margin=config.edge_margin_s,
        distance_factor=config.distance_factor,
        rescue_threshold_factor=config.rescue_threshold_factor,
        raw_peak_distance=config.raw_peak_distance_s,
        use_local_period_nms=True,
        use_periodic_support_filter=True,
        periodic_min_chain_len=config.periodic_min_chain_len,
        periodic_tolerance=config.periodic_tolerance,
        refine_period_from_events=True,
    )


def _default_prepare_shot(**kwargs):
    from src.detection.utils.pipeline_utils import prepare_shot_for_detection

    return prepare_shot_for_detection(**kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def label_prepared_shot(
    record: ShotRecord,
    prepared,
    config: WaveletLabelConfig,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> PseudoLabeledShot:

    downsample = config.downsample
    time_plasma = np.asarray(prepared.time_plasma, dtype=float)
    if len(time_plasma) == 0:
        raise ValueError(f"Empty plasma interval for {record.shot_id}")
    if record.category.casefold() == "notsaw":
        return PseudoLabeledShot(
            shot_id=record.shot_id,
            category=record.category,
            relative_path=record.relative_path,
            status="labeled",
            channel=prepared.channel_name,
            events=(),
        )
    dt_effective = float(prepared.dt) * downsample
    channel_snr = getattr(prepared, "channel_snr", None) or {}
    signals_ds = {
        channel_name: np.asarray(prepared.channel_signals[channel_name], dtype=float)[
            ::downsample
        ]
        for channel_name in config.channels
        if channel_name in prepared.channel_signals
        and np.isfinite(channel_snr.get(channel_name, np.nan))
        and channel_snr[channel_name] >= config.min_snr
    }
    voter = MultiChannelVotingDetector(
        detector_factory=lambda: detector_factory(
            dt_effective,
            float(prepared.estimated_period),
            config,
        ),
        dt=dt_effective,
        coincidence_window=config.coincidence_window_s,
        min_channels=config.min_support_channels,
    )
    period_map_ds = (
        None
        if prepared.period_map is None
        else np.asarray(prepared.period_map)[::downsample]
    )
    active_mask_ds = (
        None
        if prepared.saw_mask is None
        else np.asarray(prepared.saw_mask, dtype=bool)[::downsample]
    )
    voted_events = voter.detect_events(
        signals_ds,
        period=float(prepared.estimated_period),
        period_map=period_map_ds,
        active_mask=active_mask_ds,
        debug=debug,
    )
    confirmed_events = []
    for voted_event in voted_events:
        plasma_index = int(voted_event.index) * downsample
        if not 0 <= plasma_index < len(time_plasma):
            continue
        confirmed_events.append(PseudoLabelEvent(
            time_s=float(time_plasma[plasma_index]),
            score=float(voted_event.score),
        ))

    return PseudoLabeledShot(
        shot_id=record.shot_id,
        category=record.category,
        relative_path=record.relative_path,
        status="labeled",
        channel=prepared.channel_name,
        events=tuple(sorted(confirmed_events, key=lambda event: event.time_s)),
    )


def _empty_shot(record: ShotRecord, config: WaveletLabelConfig, status: str, error: str | None = None):
    return PseudoLabeledShot(
        shot_id=record.shot_id,
        category=record.category,
        relative_path=record.relative_path,
        status=status,
        channel="auto_sxr",
        events=(),
        error=error,
    )


def _label_records(
    records: tuple[ShotRecord, ...],
    source_dir: str | Path,
    logger,
    config: WaveletLabelConfig,
    prepare_shot: PrepareShotFn,
    detector_factory: DetectorFactory,
    debug: bool,
) -> tuple[PseudoLabeledShot, ...]:
    labeled: list[PseudoLabeledShot] = []
    for record in records:
        logger.info(f"Generating wavelet pseudo-labels for {record.relative_path}")
        try:
            prepared = prepare_shot(
                source_dir=str(source_dir),
                filename=record.relative_path,
                logger=logger,
                channel_name=config.channels[0],
                channels=config.channels,
                require_sawtooth=False,
                minimum_snr=config.min_snr,
                auto_reference_sxr=True,
            )
            if prepared is None:
                labeled.append(_empty_shot(record, config, "skipped_preprocessing"))
                continue
            labeled.append(
                label_prepared_shot(
                    record,
                    prepared,
                    config,
                    detector_factory=detector_factory,
                    debug=debug,
                )
            )
        except Exception as exc:
            logger.error(f"Could not label {record.relative_path}: {exc}")
            labeled.append(_empty_shot(record, config, "error", str(exc)))
    return tuple(labeled)


def generate_wavelet_pseudo_labels(
    source_dir: str | Path,
    split: ShotSplit,
    logger,
    config: WaveletLabelConfig | None = None,
    prepare_shot: PrepareShotFn = _default_prepare_shot,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> PseudoLabelDataset:
    config = config or WaveletLabelConfig()
    train = _label_records(
        split.train, source_dir, logger, config, prepare_shot, detector_factory, debug
    )
    validation = _label_records(
        split.validation, source_dir, logger, config, prepare_shot, detector_factory, debug
    )
    return PseudoLabelDataset(
        source_dir=Path(source_dir).as_posix(),
        config=config,
        train=train,
        validation=validation,
    )


def save_pseudo_labels(dataset: PseudoLabelDataset, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(dataset))
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_pseudo_labels(input_path: str | Path) -> PseudoLabelDataset:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != PSEUDO_LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pseudo-label schema version: {schema_version}; "
            f"expected {PSEUDO_LABEL_SCHEMA_VERSION}"
        )

    def parse_shot(item: dict[str, Any]) -> PseudoLabeledShot:
        events = tuple(PseudoLabelEvent(**event) for event in item.get("events", []))
        shot_data = dict(item)
        shot_data["events"] = events
        return PseudoLabeledShot(**shot_data)

    return PseudoLabelDataset(
        source_dir=payload["source_dir"],
        config=WaveletLabelConfig(**payload["config"]),
        train=tuple(parse_shot(item) for item in payload["train"]),
        validation=tuple(parse_shot(item) for item in payload["validation"]),
        schema_version=schema_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wavelet pseudo-labels for the feature-ML dataset")
    parser.add_argument("--source-dir", default="data/raw")
    parser.add_argument("--output", default=DEFAULT_PSEUDO_LABELS_PATH)
    parser.add_argument("--channels", nargs="+", default=SXR_CHANNELS)
    parser.add_argument("--min-snr", type=float, default=2.0)
    parser.add_argument("--min-support-channels", type=int, default=2)
    parser.add_argument("--coincidence-window-ms", type=float, default=0.15)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from src.logger import setup_logger

    logger = setup_logger(log_to_file=False)
    manifest = build_shot_manifest(args.source_dir)
    split = stratified_shot_split(
        manifest,
        validation_ratio=args.validation_ratio,
        random_state=args.random_state,
    )
    dataset = generate_wavelet_pseudo_labels(
        source_dir=args.source_dir,
        split=split,
        logger=logger,
        config=WaveletLabelConfig(
            channels=tuple(args.channels),
            min_snr=args.min_snr,
            min_support_channels=args.min_support_channels,
            coincidence_window_s=args.coincidence_window_ms * 1e-3,
            downsample=args.downsample,
        ),
        debug=args.debug,
    )
    output = save_pseudo_labels(dataset, args.output)
    logger.info(f"Pseudo-labels saved to {output.resolve()}")
    logger.info(f"Summary: {dataset.summary()}")


if __name__ == "__main__":
    main()
