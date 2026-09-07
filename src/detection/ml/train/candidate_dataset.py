import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.detection.ml.train.pseudo_labels import (
    PseudoLabelDataset,
    PseudoLabeledShot,
    load_pseudo_labels,
)


CANDIDATE_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureMLCandidateConfig:
    downsample: int = 10
    threshold: float = 3.0
    score_threshold: float = 0.5
    min_distance_factor: float = 0.45
    use_cpd: bool = True
    cpd_penalty: float = 3.0
    cpd_model: str = "rbf"
    positive_tolerance_s: float = 0.3e-3
    negative_exclusion_s: float = 0.6e-3

    def __post_init__(self):
        if self.downsample < 1:
            raise ValueError("downsample must be at least 1")
        if self.positive_tolerance_s < 0:
            raise ValueError("positive_tolerance_s must be non-negative")
        if self.negative_exclusion_s <= self.positive_tolerance_s:
            raise ValueError("negative_exclusion_s must exceed positive_tolerance_s")


@dataclass(frozen=True)
class LabeledCandidate:
    plasma_index: int
    time_s: float
    relative_time_s: float
    label: int
    proposer_score: float
    direction: str
    method: str
    closest_teacher_time_s: Optional[float]
    distance_to_teacher_s: Optional[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateShot:
    shot_id: str
    category: str
    relative_path: str
    status: str
    channel: str
    dt_s: Optional[float]
    estimated_period_s: Optional[float]
    teacher_event_count: int
    matched_teacher_event_count: int
    proposal_recall: Optional[float]
    ignored_candidate_count: int
    candidates: tuple[LabeledCandidate, ...]
    error: Optional[str] = None


@dataclass(frozen=True)
class CandidateDataset:
    source_dir: str
    teacher_schema_version: int
    config: FeatureMLCandidateConfig
    train: tuple[CandidateShot, ...]
    validation: tuple[CandidateShot, ...]
    schema_version: int = CANDIDATE_DATASET_SCHEMA_VERSION

    def summary(self) -> dict[str, Any]:
        shots = self.train + self.validation
        candidates = [candidate for shot in shots for candidate in shot.candidates]
        teacher_events = sum(shot.teacher_event_count for shot in shots)
        matched_teacher_events = sum(shot.matched_teacher_event_count for shot in shots)
        return {
            "train_shots": len(self.train),
            "validation_shots": len(self.validation),
            "candidates": len(candidates),
            "positive_candidates": sum(candidate.label == 1 for candidate in candidates),
            "negative_candidates": sum(candidate.label == 0 for candidate in candidates),
            "ignored_candidates": sum(shot.ignored_candidate_count for shot in shots),
            "teacher_events": teacher_events,
            "matched_teacher_events": matched_teacher_events,
            "proposal_recall": (
                matched_teacher_events / teacher_events if teacher_events else None
            ),
            "statuses": dict(sorted(Counter(shot.status for shot in shots).items())),
        }


DetectorFactory = Callable[[float, float, FeatureMLCandidateConfig], object]
PrepareShotFn = Callable[..., object]


def _default_detector_factory(dt: float, period: float, config: FeatureMLCandidateConfig):
    from src.detection.ml.models.hybrid_proposal_detector import HybridProposalDetector
    from src.detection.models.cpd_detector import CPDDetector
    from src.detection.models.posr_detector import WaveletPOSRDetector

    sigma = float(np.clip(0.03 * period, 30e-6, 200e-6))
    posr_detector = WaveletPOSRDetector(
        dt=dt,
        sigma=sigma,
        threshold=config.threshold,
        score_threshold=config.score_threshold,
        min_distance_factor=config.min_distance_factor,
    )
    if not config.use_cpd:
        return posr_detector
    return HybridProposalDetector(
        posr_detector=posr_detector,
        cpd_detector=CPDDetector(
            dt=dt,
            penalty=config.cpd_penalty,
            model=config.cpd_model,
        ),
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
    return value


def _empty_candidate_shot(
    teacher: PseudoLabeledShot,
    status: str,
    error: str | None = None,
) -> CandidateShot:
    return CandidateShot(
        shot_id=teacher.shot_id,
        category=teacher.category,
        relative_path=teacher.relative_path,
        status=status,
        channel=teacher.channel,
        dt_s=None,
        estimated_period_s=None,
        teacher_event_count=len(teacher.events),
        matched_teacher_event_count=0,
        proposal_recall=None,
        ignored_candidate_count=0,
        candidates=(),
        error=error,
    )


def label_candidates(
    teacher: PseudoLabeledShot,
    prepared,
    config: FeatureMLCandidateConfig,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> CandidateShot:
    """Generate POSR + raw CPD proposals and label them by teacher distance."""

    downsample = config.downsample
    signal = np.asarray(prepared.reference_signal)[::downsample]
    period_map = None if prepared.period_map is None else np.asarray(prepared.period_map)[::downsample]
    active_mask = None if prepared.saw_mask is None else np.asarray(prepared.saw_mask)[::downsample]
    period = float(prepared.estimated_period)

    detector = detector_factory(float(prepared.dt) * downsample, period, config)
    proposals = detector.detect_candidates(
        signal,
        period=period,
        period_map=period_map,
        active_mask=active_mask,
        channel=teacher.channel,
        debug=debug,
    )

    time_plasma = np.asarray(prepared.time_plasma, dtype=float)
    if len(time_plasma) == 0:
        raise ValueError(f"Empty plasma interval for {teacher.shot_id}")

    proposals_by_index: dict[int, object] = {}
    for proposal in proposals:
        plasma_index = int(proposal.index) * downsample
        if not 0 <= plasma_index < len(time_plasma):
            continue
        previous = proposals_by_index.get(plasma_index)
        if previous is None or float(proposal.score) > float(previous.score):
            proposals_by_index[plasma_index] = proposal

    teacher_times = np.asarray([event.time_s for event in teacher.events], dtype=float)
    is_not_saw = teacher.category.casefold() == "notsaw"
    if not is_not_saw and len(teacher_times) == 0:
        raise ValueError(f"No teacher events for sawtooth shot {teacher.shot_id}")
    labeled: list[LabeledCandidate] = []
    ignored_count = 0

    for plasma_index in sorted(proposals_by_index):
        proposal = proposals_by_index[plasma_index]
        time_s = float(time_plasma[plasma_index])

        if is_not_saw:
            label = 0
            closest_time = None
            distance = None
        else:
            distances = np.abs(teacher_times - time_s)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            closest_time = float(teacher_times[nearest])
            if distance <= config.positive_tolerance_s:
                label = 1
            elif distance >= config.negative_exclusion_s:
                label = 0
            else:
                ignored_count += 1
                continue

        labeled.append(
            LabeledCandidate(
                plasma_index=plasma_index,
                time_s=time_s,
                relative_time_s=float(time_s - time_plasma[0]),
                label=label,
                proposer_score=float(proposal.score),
                direction=str(proposal.direction),
                method=str(proposal.method or "wavelet_posr"),
                closest_teacher_time_s=closest_time,
                distance_to_teacher_s=distance,
                metadata=_json_safe(dict(proposal.meta)),
            )
        )

    if is_not_saw or len(teacher_times) == 0:
        matched_teacher_count = 0
        proposal_recall = None
    else:
        proposal_times = np.asarray(
            [float(time_plasma[index]) for index in proposals_by_index], dtype=float
        )
        matched_teacher_count = sum(
            bool(len(proposal_times))
            and bool(np.min(np.abs(proposal_times - teacher_time)) <= config.positive_tolerance_s)
            for teacher_time in teacher_times
        )
        proposal_recall = matched_teacher_count / len(teacher_times)

    return CandidateShot(
        shot_id=teacher.shot_id,
        category=teacher.category,
        relative_path=teacher.relative_path,
        status="candidates_built",
        channel=teacher.channel,
        dt_s=float(prepared.dt),
        estimated_period_s=period,
        teacher_event_count=len(teacher.events),
        matched_teacher_event_count=matched_teacher_count,
        proposal_recall=proposal_recall,
        ignored_candidate_count=ignored_count,
        candidates=tuple(labeled),
    )


def _build_split(
    teacher_shots: tuple[PseudoLabeledShot, ...],
    source_dir: str | Path,
    logger,
    config: FeatureMLCandidateConfig,
    prepare_shot: PrepareShotFn,
    detector_factory: DetectorFactory,
    debug: bool,
) -> tuple[CandidateShot, ...]:
    result: list[CandidateShot] = []
    for teacher in teacher_shots:
        is_not_saw = teacher.category.casefold() == "notsaw"
        if not is_not_saw and (teacher.status != "labeled" or not teacher.events):
            result.append(_empty_candidate_shot(teacher, "skipped_no_teacher_labels"))
            continue

        sources = "POSR + raw CPD" if config.use_cpd else "POSR"
        logger.info(f"Generating {sources} candidates for {teacher.relative_path}")
        try:
            prepared = prepare_shot(
                source_dir=str(source_dir),
                filename=teacher.relative_path,
                logger=logger,
                channel_name=teacher.channel,
                channels=[teacher.channel],
                require_sawtooth=False,
            )
            if prepared is None:
                result.append(_empty_candidate_shot(teacher, "skipped_preprocessing"))
                continue
            result.append(
                label_candidates(
                    teacher,
                    prepared,
                    config,
                    detector_factory=detector_factory,
                    debug=debug,
                )
            )
        except Exception as exc:
            logger.error(f"Could not build candidates for {teacher.relative_path}: {exc}")
            result.append(_empty_candidate_shot(teacher, "error", str(exc)))
    return tuple(result)


def build_candidate_dataset(
    pseudo_labels: PseudoLabelDataset,
    logger,
    config: FeatureMLCandidateConfig | None = None,
    prepare_shot: PrepareShotFn = _default_prepare_shot,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> CandidateDataset:
    config = config or FeatureMLCandidateConfig(downsample=pseudo_labels.config.downsample)
    train = _build_split(
        pseudo_labels.train,
        pseudo_labels.source_dir,
        logger,
        config,
        prepare_shot,
        detector_factory,
        debug,
    )
    validation = _build_split(
        pseudo_labels.validation,
        pseudo_labels.source_dir,
        logger,
        config,
        prepare_shot,
        detector_factory,
        debug,
    )
    return CandidateDataset(
        source_dir=pseudo_labels.source_dir,
        teacher_schema_version=pseudo_labels.schema_version,
        config=config,
        train=train,
        validation=validation,
    )


def save_candidate_dataset(dataset: CandidateDataset, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_safe(asdict(dataset)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_candidate_dataset(input_path: str | Path) -> CandidateDataset:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != CANDIDATE_DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported candidate dataset schema version: {schema_version}; "
            f"expected {CANDIDATE_DATASET_SCHEMA_VERSION}"
        )

    def parse_shot(item: dict[str, Any]) -> CandidateShot:
        candidates = tuple(LabeledCandidate(**candidate) for candidate in item.get("candidates", []))
        shot_data = dict(item)
        shot_data["candidates"] = candidates
        return CandidateShot(**shot_data)

    return CandidateDataset(
        source_dir=payload["source_dir"],
        teacher_schema_version=int(payload["teacher_schema_version"]),
        config=FeatureMLCandidateConfig(**payload["config"]),
        train=tuple(parse_shot(item) for item in payload["train"]),
        validation=tuple(parse_shot(item) for item in payload["validation"]),
        schema_version=schema_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled POSR + raw CPD candidates from wavelet pseudo-labels"
    )
    parser.add_argument("--pseudo-labels", default="data/dataset/wavelet_pseudo_labels.json")
    parser.add_argument("--output", default="data/dataset/feature_ml_candidates.json")
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--no-cpd", action="store_true")
    parser.add_argument("--cpd-penalty", type=float, default=3.0)
    parser.add_argument("--cpd-model", default="rbf")
    parser.add_argument("--positive-tolerance-ms", type=float, default=0.3)
    parser.add_argument("--negative-exclusion-ms", type=float, default=0.6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from src.logger import setup_logger

    logger = setup_logger(log_to_file=False)
    pseudo_labels = load_pseudo_labels(args.pseudo_labels)
    downsample = args.downsample or pseudo_labels.config.downsample
    config = FeatureMLCandidateConfig(
        downsample=downsample,
        threshold=args.threshold,
        score_threshold=args.score_threshold,
        use_cpd=not args.no_cpd,
        cpd_penalty=args.cpd_penalty,
        cpd_model=args.cpd_model,
        positive_tolerance_s=args.positive_tolerance_ms * 1e-3,
        negative_exclusion_s=args.negative_exclusion_ms * 1e-3,
    )
    dataset = build_candidate_dataset(
        pseudo_labels,
        logger=logger,
        config=config,
        debug=args.debug,
    )
    output = save_candidate_dataset(dataset, args.output)
    logger.info(f"Candidate dataset saved to {output.resolve()}")
    logger.info(f"Summary: {dataset.summary()}")


if __name__ == "__main__":
    main()
