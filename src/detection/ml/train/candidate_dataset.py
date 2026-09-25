import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from config.path import (
    DEFAULT_EXPERT_LABELS_PATH,
    DEFAULT_PSEUDO_LABELS_PATH,
    SXR_CHANNELS,
    feature_ml_artifact_paths,
)
from src.detection.ml.multisxr_proposals import collect_multisxr_candidates
from src.detection.ml.train.expert_labels import (
    ExpertEvent,
    ExpertLabeledShot,
    ExpertLabelDataset,
    load_expert_labels,
)
from src.detection.ml.train.pseudo_labels import (
    PseudoLabelDataset,
    PseudoLabeledShot,
    load_pseudo_labels,
)


CANDIDATE_DATASET_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class FeatureMLCandidateConfig:
    downsample: int = 10
    threshold: float = 3.0
    min_distance_factor: float = 0.45
    proposal_source: str = "posr"
    cpd_penalty: float = 3.0
    cpd_model: str = "rbf"
    positive_tolerance_s: float = 0.3e-3
    negative_exclusion_s: float = 0.6e-3
    proposal_channels: tuple[str, ...] = tuple(SXR_CHANNELS)
    coincidence_window_s: float = 0.15e-3
    plateau_level_fraction: float = 0.8
    plateau_hold_s: float = 0.5e-3

    def __post_init__(self):
        if self.downsample < 1:
            raise ValueError("downsample must be at least 1")
        proposal_source = str(self.proposal_source).lower()
        if proposal_source not in {"posr", "cpd", "hybrid"}:
            raise ValueError("proposal_source must be one of: posr, cpd, hybrid")
        object.__setattr__(self, "proposal_source", proposal_source)
        if self.positive_tolerance_s < 0:
            raise ValueError("positive_tolerance_s must be non-negative")
        if self.negative_exclusion_s <= self.positive_tolerance_s:
            raise ValueError("negative_exclusion_s must exceed positive_tolerance_s")
        proposal_channels = tuple(dict.fromkeys(self.proposal_channels))
        if not proposal_channels:
            raise ValueError("proposal_channels must not be empty")
        object.__setattr__(self, "proposal_channels", proposal_channels)
        if not np.isfinite(self.coincidence_window_s) or self.coincidence_window_s <= 0:
            raise ValueError("coincidence_window_s must be positive and finite")
        if not 0 < self.plateau_level_fraction < 1:
            raise ValueError("plateau_level_fraction must be between 0 and 1")
        if not np.isfinite(self.plateau_hold_s) or self.plateau_hold_s <= 0:
            raise ValueError("plateau_hold_s must be positive and finite")


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
    label_source: str
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
    expert_event_count: int
    matched_expert_event_count: int
    expert_proposal_recall: Optional[float]
    injected_expert_candidate_count: int
    ignored_candidate_count: int
    candidates: tuple[LabeledCandidate, ...]
    error: Optional[str] = None


@dataclass(frozen=True)
class CandidateDataset:
    source_dir: str
    teacher_schema_version: int
    expert_label_schema_version: Optional[int]
    config: FeatureMLCandidateConfig
    train: tuple[CandidateShot, ...]
    validation: tuple[CandidateShot, ...]
    schema_version: int = CANDIDATE_DATASET_SCHEMA_VERSION

    def summary(self) -> dict[str, Any]:
        shots = self.train + self.validation
        candidates = [candidate for shot in shots for candidate in shot.candidates]
        teacher_events = sum(shot.teacher_event_count for shot in shots)
        matched_teacher_events = sum(shot.matched_teacher_event_count for shot in shots)
        expert_events = sum(shot.expert_event_count for shot in shots)
        matched_expert_events = sum(shot.matched_expert_event_count for shot in shots)
        return {
            "train_shots": len(self.train),
            "validation_shots": len(self.validation),
            "candidates": len(candidates),
            "positive_candidates": sum(candidate.label == 1 for candidate in candidates),
            "negative_candidates": sum(candidate.label == 0 for candidate in candidates),
            "ignored_candidates": sum(shot.ignored_candidate_count for shot in shots),
            "injected_expert_candidates": sum(
                shot.injected_expert_candidate_count for shot in shots
            ),
            "candidate_label_sources": dict(
                sorted(Counter(candidate.label_source for candidate in candidates).items())
            ),
            "teacher_events": teacher_events,
            "matched_teacher_events": matched_teacher_events,
            "proposal_recall": (
                matched_teacher_events / teacher_events if teacher_events else None
            ),
            "expert_events": expert_events,
            "matched_expert_events": matched_expert_events,
            "expert_proposal_recall": (
                matched_expert_events / expert_events if expert_events else None
            ),
            "statuses": dict(sorted(Counter(shot.status for shot in shots).items())),
        }


DetectorFactory = Callable[[float, float, FeatureMLCandidateConfig], object]
PrepareShotFn = Callable[..., object]


def _default_detector_factory(dt: float, period: float, config: FeatureMLCandidateConfig):
    from src.detection.ml.models.hybrid_proposal_detector import FeatureMLProposalDetector
    from src.detection.models.cpd_detector import CPDDetector
    from src.detection.models.posr_detector import WaveletPOSRDetector

    sigma = float(np.clip(0.03 * period, 30e-6, 200e-6))
    posr_detector = WaveletPOSRDetector(
        dt=dt,
        sigma=sigma,
        threshold=config.threshold,
        min_distance_factor=config.min_distance_factor,
    )
    return FeatureMLProposalDetector(
        source=config.proposal_source,
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
    expert: ExpertLabeledShot | None = None,
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
        expert_event_count=len(expert.positive_events) if expert else 0,
        matched_expert_event_count=0,
        expert_proposal_recall=None,
        injected_expert_candidate_count=0,
        ignored_candidate_count=0,
        candidates=(),
        error=error,
    )


def label_candidates(
    teacher: PseudoLabeledShot,
    prepared,
    config: FeatureMLCandidateConfig,
    expert: ExpertLabeledShot | None = None,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> CandidateShot:
    """Build proposals; expert labels override wavelet labels in reviewed intervals."""

    downsample = config.downsample
    period = float(prepared.estimated_period)
    proposals = collect_multisxr_candidates(
        prepared,
        downsample=downsample,
        channels=config.proposal_channels,
        coincidence_window_s=config.coincidence_window_s,
        detector_factory=lambda dt, estimated_period: detector_factory(
            dt, estimated_period, config
        ),
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

    pseudo_events = tuple(
        event
        for event in teacher.events
        if expert is None or not expert.is_reviewed(event.time_s)
    )
    pseudo_uncertain_events = tuple(
        event
        for event in teacher.uncertain_events
        if expert is None or not expert.is_reviewed(event.time_s)
    )
    expert_events = expert.positive_events if expert else ()
    expert_uncertain_events = expert.uncertain_events if expert else ()
    effective_events: tuple[tuple[float, float], ...] = tuple(
        (event.time_s, config.positive_tolerance_s) for event in pseudo_events
    ) + tuple(
        (event.time_s, event.effective_tolerance(config.positive_tolerance_s))
        for event in expert_events
    )
    teacher_times = np.asarray([event.time_s for event in pseudo_events], dtype=float)
    uncertain_times = np.asarray(
        [event.time_s for event in pseudo_uncertain_events], dtype=float
    )
    is_not_saw = teacher.category.casefold() == "notsaw"
    if not is_not_saw and not effective_events and expert is None:
        raise ValueError(f"No teacher events for sawtooth shot {teacher.shot_id}")
    labeled: list[LabeledCandidate] = []
    ignored_count = 0

    def closest_expert_event(
        time_s: float,
        events: tuple[ExpertEvent, ...],
    ) -> tuple[ExpertEvent | None, float | None]:
        if not events:
            return None, None
        event = min(events, key=lambda item: abs(item.time_s - time_s))
        return event, abs(event.time_s - time_s)

    for plasma_index in sorted(proposals_by_index):
        proposal = proposals_by_index[plasma_index]
        time_s = float(time_plasma[plasma_index])

        expert_positive_nearby = (
            expert is not None
            and any(
                abs(event.time_s - time_s)
                <= event.effective_tolerance(config.positive_tolerance_s)
                for event in expert_events
            )
        )
        expert_uncertain_nearby = (
            expert is not None
            and any(
                abs(event.time_s - time_s)
                < event.effective_tolerance(config.negative_exclusion_s)
                for event in expert_uncertain_events
            )
        )
        if expert is not None and (
            expert.is_reviewed(time_s)
            or expert_positive_nearby
            or expert_uncertain_nearby
        ):
            label_source = "expert"
            nearest_event, distance = closest_expert_event(time_s, expert_events)
            closest_time = nearest_event.time_s if nearest_event else None
            if (
                nearest_event is not None
                and distance is not None
                and distance
                <= nearest_event.effective_tolerance(config.positive_tolerance_s)
            ):
                label = 1
            else:
                uncertain_event, uncertain_distance = closest_expert_event(
                    time_s, expert_uncertain_events
                )
                near_uncertain = (
                    uncertain_event is not None
                    and uncertain_distance is not None
                    and uncertain_distance
                    < uncertain_event.effective_tolerance(config.negative_exclusion_s)
                )
                if near_uncertain or (
                    distance is not None and distance < config.negative_exclusion_s
                ):
                    ignored_count += 1
                    continue
                label = 0
        elif is_not_saw:
            label = 0
            closest_time = None
            distance = None
            label_source = "wavelet"
        else:
            if not len(teacher_times):
                ignored_count += 1
                continue
            distances = np.abs(teacher_times - time_s)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            closest_time = float(teacher_times[nearest])
            label_source = "wavelet"
            if distance <= config.positive_tolerance_s:
                label = 1
            elif len(uncertain_times) and np.min(np.abs(uncertain_times - time_s)) < config.negative_exclusion_s:
                ignored_count += 1
                continue
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
                label_source=label_source,
                metadata=_json_safe(dict(proposal.meta)),
            )
        )

    proposal_times = np.asarray(
        [float(time_plasma[index]) for index in proposals_by_index], dtype=float
    )

    def matched_event_count(events: tuple[tuple[float, float], ...]) -> int:
        return sum(
            bool(len(proposal_times))
            and bool(np.min(np.abs(proposal_times - time_s)) <= tolerance_s)
            for time_s, tolerance_s in events
        )

    if not effective_events:
        matched_teacher_count = 0
        proposal_recall = None
    else:
        matched_teacher_count = matched_event_count(effective_events)
        proposal_recall = matched_teacher_count / len(effective_events)

    expert_targets = tuple(
        (event.time_s, event.effective_tolerance(config.positive_tolerance_s))
        for event in expert_events
    )
    matched_expert_count = matched_event_count(expert_targets)
    expert_proposal_recall = (
        matched_expert_count / len(expert_targets) if expert_targets else None
    )

    injected_count = 0
    for event in expert_events:
        tolerance = event.effective_tolerance(config.positive_tolerance_s)
        if len(proposal_times) and np.min(np.abs(proposal_times - event.time_s)) <= tolerance:
            continue
        nearest_sample = int(np.argmin(np.abs(time_plasma - event.time_s)))
        plasma_index = int(round(nearest_sample / downsample)) * downsample
        plasma_index = min(max(plasma_index, 0), ((len(time_plasma) - 1) // downsample) * downsample)
        actual_time = float(time_plasma[plasma_index])
        if not expert.is_reviewed(actual_time):
            raise ValueError(
                f"Injected expert event at {event.time_s} s in {teacher.shot_id} "
                "falls outside its reviewed interval after grid alignment"
            )
        if abs(actual_time - event.time_s) > tolerance:
            raise ValueError(
                f"Expert event at {event.time_s} s in {teacher.shot_id} cannot be "
                "aligned to the candidate grid within its tolerance"
            )
        synthetic_channel = event.reference_sxr or teacher.channel
        synthetic_method = (
            "cpd_raw" if config.proposal_source == "cpd" else "wavelet_posr_raw"
        )
        synthetic_score = float(config.threshold)
        labeled.append(
            LabeledCandidate(
                plasma_index=plasma_index,
                time_s=actual_time,
                relative_time_s=float(actual_time - time_plasma[0]),
                label=1,
                proposer_score=synthetic_score,
                direction="unknown",
                method="expert_manual_injection",
                closest_teacher_time_s=event.time_s,
                distance_to_teacher_s=abs(actual_time - event.time_s),
                label_source="expert",
                metadata={
                    "source_channels": [synthetic_channel],
                    "source_methods": [synthetic_method],
                    "source_channel_methods": {
                        synthetic_channel: [synthetic_method]
                    },
                    "source_scores": [synthetic_score],
                    "source_count": 1,
                    "proposal_count": 1,
                    "time_spread_s": 0.0,
                    "expert_injected": True,
                },
            )
        )
        injected_count += 1

    labeled.sort(key=lambda candidate: candidate.plasma_index)

    return CandidateShot(
        shot_id=teacher.shot_id,
        category=teacher.category,
        relative_path=teacher.relative_path,
        status="candidates_built",
        channel=teacher.channel,
        dt_s=float(prepared.dt),
        estimated_period_s=period,
        teacher_event_count=len(effective_events),
        matched_teacher_event_count=matched_teacher_count,
        proposal_recall=proposal_recall,
        expert_event_count=len(expert_events),
        matched_expert_event_count=matched_expert_count,
        expert_proposal_recall=expert_proposal_recall,
        injected_expert_candidate_count=injected_count,
        ignored_candidate_count=ignored_count,
        candidates=tuple(labeled),
    )


def _build_split(
    teacher_shots: tuple[PseudoLabeledShot, ...],
    source_dir: str | Path,
    logger,
    config: FeatureMLCandidateConfig,
    expert_by_shot_id: dict[str, ExpertLabeledShot],
    prepare_shot: PrepareShotFn,
    detector_factory: DetectorFactory,
    debug: bool,
) -> tuple[CandidateShot, ...]:
    result: list[CandidateShot] = []
    for teacher in teacher_shots:
        expert = expert_by_shot_id.get(teacher.shot_id)
        is_not_saw = teacher.category.casefold() == "notsaw"
        has_expert_supervision = expert is not None and bool(expert.reviewed_intervals)
        if (
            not is_not_saw
            and not has_expert_supervision
            and (teacher.status != "labeled" or not teacher.events)
        ):
            result.append(
                _empty_candidate_shot(teacher, "skipped_no_teacher_labels", expert)
            )
            continue

        logger.info(
            f"Generating raw {config.proposal_source} candidates for {teacher.relative_path}"
        )
        try:
            prepared = prepare_shot(
                source_dir=str(source_dir),
                filename=teacher.relative_path,
                logger=logger,
                channel_name=teacher.channel,
                channels=list(dict.fromkeys(config.proposal_channels + (teacher.channel,))),
                require_sawtooth=False,
                enable_plasma_plateau_gate=True,
                plateau_level_fraction=config.plateau_level_fraction,
                plateau_hold_s=config.plateau_hold_s,
            )
            if prepared is None:
                result.append(
                    _empty_candidate_shot(teacher, "skipped_preprocessing", expert)
                )
                continue
            result.append(
                label_candidates(
                    teacher,
                    prepared,
                    config,
                    expert=expert,
                    detector_factory=detector_factory,
                    debug=debug,
                )
            )
        except Exception as exc:
            logger.error(f"Could not build candidates for {teacher.relative_path}: {exc}")
            result.append(_empty_candidate_shot(teacher, "error", expert, str(exc)))
    return tuple(result)


def build_candidate_dataset(
    pseudo_labels: PseudoLabelDataset,
    logger,
    config: FeatureMLCandidateConfig | None = None,
    expert_labels: ExpertLabelDataset | None = None,
    prepare_shot: PrepareShotFn = _default_prepare_shot,
    detector_factory: DetectorFactory = _default_detector_factory,
    debug: bool = False,
) -> CandidateDataset:
    config = config or FeatureMLCandidateConfig(downsample=pseudo_labels.config.downsample)
    expert_by_shot_id = expert_labels.by_shot_id() if expert_labels else {}
    known_shot_ids = {
        shot.shot_id for shot in pseudo_labels.train + pseudo_labels.validation
    }
    unknown_expert_shots = sorted(expert_by_shot_id.keys() - known_shot_ids)
    if unknown_expert_shots:
        raise ValueError(
            "Expert labels contain shots absent from the pseudo-label split: "
            f"{unknown_expert_shots}"
        )
    train = _build_split(
        pseudo_labels.train,
        pseudo_labels.source_dir,
        logger,
        config,
        expert_by_shot_id,
        prepare_shot,
        detector_factory,
        debug,
    )
    validation = _build_split(
        pseudo_labels.validation,
        pseudo_labels.source_dir,
        logger,
        config,
        expert_by_shot_id,
        prepare_shot,
        detector_factory,
        debug,
    )
    return CandidateDataset(
        source_dir=pseudo_labels.source_dir,
        teacher_schema_version=pseudo_labels.schema_version,
        expert_label_schema_version=(
            expert_labels.schema_version if expert_labels is not None else None
        ),
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
        expert_label_schema_version=(
            None
            if payload.get("expert_label_schema_version") is None
            else int(payload["expert_label_schema_version"])
        ),
        config=FeatureMLCandidateConfig(**payload["config"]),
        train=tuple(parse_shot(item) for item in payload["train"]),
        validation=tuple(parse_shot(item) for item in payload["validation"]),
        schema_version=schema_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled ML candidates from wavelet and expert labels"
    )
    parser.add_argument("--pseudo-labels", default=DEFAULT_PSEUDO_LABELS_PATH)
    parser.add_argument("--expert-labels", default=DEFAULT_EXPERT_LABELS_PATH)
    parser.add_argument(
        "--output",
        default=None,
        help="Default: feature_ml_candidates_<proposal-source>.json",
    )
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument(
        "--proposal-source",
        choices=("posr", "cpd", "hybrid"),
        default="posr",
    )
    parser.add_argument("--cpd-penalty", type=float, default=3.0)
    parser.add_argument("--cpd-model", default="rbf")
    parser.add_argument("--proposal-channels", nargs="+", default=SXR_CHANNELS)
    parser.add_argument("--coincidence-window-ms", type=float, default=0.15)
    parser.add_argument("--positive-tolerance-ms", type=float, default=0.3)
    parser.add_argument("--negative-exclusion-ms", type=float, default=0.6)
    parser.add_argument("--plateau-level-fraction", type=float, default=0.8)
    parser.add_argument("--plateau-hold-ms", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from src.logger import setup_logger

    logger = setup_logger(log_to_file=False)
    pseudo_labels = load_pseudo_labels(args.pseudo_labels)
    expert_labels = load_expert_labels(args.expert_labels)
    downsample = args.downsample or pseudo_labels.config.downsample
    config = FeatureMLCandidateConfig(
        downsample=downsample,
        threshold=args.threshold,
        proposal_source=args.proposal_source,
        cpd_penalty=args.cpd_penalty,
        cpd_model=args.cpd_model,
        positive_tolerance_s=args.positive_tolerance_ms * 1e-3,
        negative_exclusion_s=args.negative_exclusion_ms * 1e-3,
        proposal_channels=tuple(args.proposal_channels),
        coincidence_window_s=args.coincidence_window_ms * 1e-3,
        plateau_level_fraction=args.plateau_level_fraction,
        plateau_hold_s=args.plateau_hold_ms * 1e-3,
    )
    dataset = build_candidate_dataset(
        pseudo_labels,
        logger=logger,
        config=config,
        expert_labels=expert_labels,
        debug=args.debug,
    )
    output_path = args.output or feature_ml_artifact_paths(args.proposal_source)["candidates"]
    output = save_candidate_dataset(dataset, output_path)
    logger.info(f"Candidate dataset saved to {output.resolve()}")
    logger.info(f"Summary: {dataset.summary()}")


if __name__ == "__main__":
    main()
