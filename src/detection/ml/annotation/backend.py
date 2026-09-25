from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config.path import IP_CHANNEL, SXR_CHANNELS
from config.channels import REFERENCE_SXR_FEATURE_PROFILE, get_feature_map_profile
from src.detection.ml.models.feature_ml_detector import (
    FeatureMLCrashDetector,
    load_feature_ml_payload,
)
from src.detection.ml.multisxr_proposals import collect_multisxr_candidates
from src.detection.ml.train.candidate_dataset import (
    FeatureMLCandidateConfig,
    _default_detector_factory,
)
from src.detection.ml.train.expert_labels import ExpertLabelDataset
from src.detection.ml.train.pseudo_labels import PseudoLabelDataset
from src.detection.utils.pipeline_utils import prepare_shot_for_detection
from src.io.loader import SHTLoader


@dataclass(frozen=True)
class AnnotationShot:
    shot_id: str
    category: str
    relative_path: str
    split: str
    teacher_event_count: int
    is_annotated: bool


@dataclass(frozen=True)
class AutomaticSuggestion:
    source: str
    time_s: float
    score: float
    reference_sxr: str | None
    source_channels: tuple[str, ...] = ()


def build_shot_catalog(
    pseudo_labels: PseudoLabelDataset,
    expert_labels: ExpertLabelDataset,
) -> tuple[AnnotationShot, ...]:
    annotated = set(expert_labels.by_shot_id())
    records = []
    for split, shots in (
        ("train", pseudo_labels.train),
        ("validation", pseudo_labels.validation),
    ):
        records.extend(
            AnnotationShot(
                shot_id=shot.shot_id,
                category=shot.category,
                relative_path=shot.relative_path,
                split=split,
                teacher_event_count=len(shot.events),
                is_annotated=shot.shot_id in annotated,
            )
            for shot in shots
        )
    return tuple(sorted(records, key=lambda item: (item.category.casefold(), item.shot_id)))


def default_display_channels(available: Iterable[str]) -> list[str]:
    available_set = set(available)
    return [
        name for name in [*SXR_CHANNELS, IP_CHANNEL]
        if name in available_set
    ]


def merge_reviewed_interval(
    intervals: Iterable[Iterable[float]],
    start_s: float,
    end_s: float,
) -> list[list[float]]:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise ValueError("Reviewed interval must satisfy start_s < end_s")
    ordered = sorted(
        [[float(a), float(b)] for a, b in intervals] + [[float(start_s), float(end_s)]],
        key=lambda item: item[0],
    )
    merged: list[list[float]] = []
    for current_start, current_end in ordered:
        if not merged or current_start > merged[-1][1]:
            merged.append([current_start, current_end])
        else:
            merged[-1][1] = max(merged[-1][1], current_end)
    return merged


def load_channels(
    source_dir: str | Path,
    relative_path: str,
    channels: Iterable[str],
    logger,
):
    loader = SHTLoader(str(source_dir), logger=logger)
    return loader.load_multirate_shot(relative_path, channels=list(channels))


def decimate_minmax(
    time: np.ndarray,
    values: np.ndarray,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve local extrema while reducing a long trace for browser display."""

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or len(time) != len(values):
        raise ValueError("time and values must be same-length 1D arrays")
    if max_points < 4:
        raise ValueError("max_points must be at least 4")
    if len(time) <= max_points:
        return time.copy(), values.copy()
    bin_count = max(1, max_points // 2)
    boundaries = np.linspace(0, len(time), bin_count + 1, dtype=int)
    indices: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segment = values[start:end]
        finite = np.flatnonzero(np.isfinite(segment))
        if not len(finite):
            indices.append(start)
            continue
        finite_values = segment[finite]
        low = start + int(finite[int(np.argmin(finite_values))])
        high = start + int(finite[int(np.argmax(finite_values))])
        indices.extend(sorted({low, high}))
    selected = np.asarray(sorted(set([0, *indices, len(time) - 1])), dtype=int)
    return time[selected], values[selected]


def compute_automatic_suggestions(
    source_dir: str | Path,
    relative_path: str,
    logger,
    include_posr: bool = True,
    include_cpd: bool = True,
    model_path: str | Path | None = None,
) -> tuple[AutomaticSuggestion, ...]:
    """Compute raw proposal overlays and optional accepted feature-ML events."""

    payload = None
    if model_path is not None and Path(model_path).is_file():
        payload = load_feature_ml_payload(model_path)
        feature_profile = get_feature_map_profile(payload["feature_map_profile"])
        proposal_config = payload["proposal_config"]
        plateau_level = float(proposal_config["plateau_level_fraction"])
        plateau_hold = float(proposal_config["plateau_hold_s"])
    else:
        feature_profile = REFERENCE_SXR_FEATURE_PROFILE
        plateau_level = 0.8
        plateau_hold = 0.5e-3
    prepared = prepare_shot_for_detection(
        source_dir=str(source_dir),
        filename=relative_path,
        logger=logger,
        channel_name="SXR 50 mkm",
        channels=SXR_CHANNELS,
        require_sawtooth=False,
        auto_reference_sxr=True,
        minimum_snr=2.0,
        feature_channel_profile=feature_profile,
        enable_plasma_plateau_gate=True,
        plateau_level_fraction=plateau_level,
        plateau_hold_s=plateau_hold,
    )
    if prepared is None:
        raise ValueError(f"Could not prepare {relative_path} for suggestions")

    result: list[AutomaticSuggestion] = []
    for source, enabled in (("posr", include_posr), ("cpd", include_cpd)):
        if not enabled:
            continue
        config = FeatureMLCandidateConfig(proposal_source=source)
        candidates = collect_multisxr_candidates(
            prepared,
            downsample=config.downsample,
            channels=config.proposal_channels,
            coincidence_window_s=config.coincidence_window_s,
            detector_factory=lambda dt, period, config=config: _default_detector_factory(
                dt, period, config
            ),
        )
        result.extend(
            AutomaticSuggestion(
                source=source,
                time_s=float(candidate.time),
                score=float(candidate.score),
                reference_sxr=prepared.channel_name,
                source_channels=tuple(candidate.meta.get("source_channels", ())),
            )
            for candidate in candidates
        )

    if payload is not None:
        downsample = int(payload["downsample"])
        detector = FeatureMLCrashDetector(
            dt=float(prepared.dt) * downsample,
            downsample=downsample,
            model_path=model_path,
            payload=payload,
        )
        result.extend(
            AutomaticSuggestion(
                source="feature_ml",
                time_s=float(candidate.time),
                score=float(candidate.meta.get("ml_probability", candidate.score)),
                reference_sxr=prepared.channel_name,
                source_channels=tuple(candidate.meta.get("source_channels", ())),
            )
            for candidate in detector.detect_candidates(prepared)
        )
    return tuple(sorted(result, key=lambda item: (item.time_s, item.source)))
