from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config.path import IP_CHANNEL, SXR_CHANNELS
from src.detection.ml.train.expert_labels import ExpertLabelDataset
from src.detection.ml.train.pseudo_labels import PseudoLabelDataset
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
