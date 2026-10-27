from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional

EXPERT_LABEL_SCHEMA_VERSION = 1
EXPERT_EVENT_LABELS = ("sawtooth", "uncertain", "false_positive")
EXPERT_REVIEW_STATUSES = ("partial", "complete")
EXPERT_SHOT_LABELS = ("sawtooth", "no_sawtooth", "uncertain")


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _check_keys(
    payload: dict[str, Any],
    field: str,
    required: set[str],
    optional: set[str],
) -> None:
    missing = required - payload.keys()
    unknown = payload.keys() - required - optional
    if missing:
        raise ValueError(f"{field} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ReviewedInterval:
    start_s: float
    end_s: float

    def contains(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s


@dataclass(frozen=True)
class ExpertEvent:
    time_s: float
    label: str

    @property
    def is_positive(self) -> bool:
        return self.label == "sawtooth"

@dataclass(frozen=True)
class ExpertLabeledShot:
    shot_id: str
    review_status: str
    shot_label: str
    reviewed_intervals: tuple[ReviewedInterval, ...]
    events: tuple[ExpertEvent, ...]
    relative_path: Optional[str] = None
    note: str = ""

    @property
    def positive_events(self) -> tuple[ExpertEvent, ...]:
        return tuple(event for event in self.events if event.is_positive)

    @property
    def uncertain_events(self) -> tuple[ExpertEvent, ...]:
        return tuple(event for event in self.events if event.label == "uncertain")

    def is_reviewed(self, time_s: float) -> bool:
        return any(interval.contains(time_s) for interval in self.reviewed_intervals)


@dataclass(frozen=True)
class ExpertLabelDataset:
    shots: tuple[ExpertLabeledShot, ...]
    schema_version: int = EXPERT_LABEL_SCHEMA_VERSION

    def by_shot_id(self) -> dict[str, ExpertLabeledShot]:
        return {shot.shot_id: shot for shot in self.shots}


def _parse_interval(value: Any, field: str) -> ReviewedInterval:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be [start_s, end_s]")
    start_s = _finite_float(value[0], f"{field}[0]")
    end_s = _finite_float(value[1], f"{field}[1]")
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"{field} must satisfy 0 <= start_s < end_s")
    return ReviewedInterval(start_s, end_s)


def _parse_event(value: Any, field: str) -> ExpertEvent:
    payload = _require_object(value, field)
    _check_keys(
        payload,
        field,
        required={"time_s", "label"},
        optional=set(),
    )
    time_s = _finite_float(payload["time_s"], f"{field}.time_s")
    if time_s < 0:
        raise ValueError(f"{field}.time_s must be non-negative")
    label = str(payload["label"]).strip().casefold()
    if label not in EXPERT_EVENT_LABELS:
        raise ValueError(f"{field}.label must be one of {EXPERT_EVENT_LABELS}")
    return ExpertEvent(time_s=time_s, label=label)


def _parse_shot(value: Any, index: int) -> ExpertLabeledShot:
    field = f"shots[{index}]"
    payload = _require_object(value, field)
    _check_keys(
        payload,
        field,
        required={
            "shot_id", "review_status", "shot_label", "reviewed_intervals_s",
            "events",
        },
        optional={"relative_path", "note"},
    )
    shot_id = str(payload["shot_id"]).strip().casefold()
    if not shot_id:
        raise ValueError(f"{field}.shot_id must not be empty")
    if shot_id.isdigit():
        shot_id = f"sht{shot_id}"
    review_status = str(payload["review_status"]).strip().casefold()
    if review_status not in EXPERT_REVIEW_STATUSES:
        raise ValueError(
            f"{field}.review_status must be one of {EXPERT_REVIEW_STATUSES}"
        )
    shot_label = str(payload["shot_label"]).strip().casefold()
    if shot_label not in EXPERT_SHOT_LABELS:
        raise ValueError(f"{field}.shot_label must be one of {EXPERT_SHOT_LABELS}")
    raw_intervals = payload["reviewed_intervals_s"]
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError(f"{field}.reviewed_intervals_s must not be empty")
    intervals = tuple(
        sorted(
            (
                _parse_interval(item, f"{field}.reviewed_intervals_s[{i}]")
                for i, item in enumerate(raw_intervals)
            ),
            key=lambda interval: interval.start_s,
        )
    )
    for previous, current in zip(intervals, intervals[1:]):
        if current.start_s <= previous.end_s:
            raise ValueError(f"{field}.reviewed_intervals_s must not overlap")
    raw_events = payload["events"]
    if not isinstance(raw_events, list):
        raise ValueError(f"{field}.events must be a list")
    events = tuple(
        sorted(
            (_parse_event(item, f"{field}.events[{i}]") for i, item in enumerate(raw_events)),
            key=lambda event: event.time_s,
        )
    )
    for event in events:
        if not any(interval.contains(event.time_s) for interval in intervals):
            raise ValueError(
                f"Expert event at {event.time_s} s in {shot_id} is outside reviewed intervals"
            )
    if shot_label == "no_sawtooth":
        if review_status != "complete":
            raise ValueError(
                f"{field}: no_sawtooth requires review_status='complete'"
            )
        if any(event.is_positive for event in events):
            raise ValueError(
                f"{field}: no_sawtooth cannot contain sawtooth events"
            )
    if any(event.is_positive for event in events) and shot_label != "sawtooth":
        raise ValueError(
            f"{field}: sawtooth events require shot_label='sawtooth'"
        )
    relative_path = payload.get("relative_path")
    if relative_path is not None:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"{field}.relative_path must be a non-empty string")
        relative_path = relative_path.strip()
        path = Path(relative_path)
        if path.suffix.casefold() != ".sht" or path.stem.casefold() != shot_id:
            raise ValueError(
                f"{field}.relative_path must point to the same shot_id and have .SHT suffix"
            )
    note = payload.get("note", "")
    if not isinstance(note, str):
        raise ValueError(f"{field}.note must be a string")
    return ExpertLabeledShot(
        shot_id=shot_id,
        review_status=review_status,
        shot_label=shot_label,
        reviewed_intervals=intervals,
        events=events,
        relative_path=relative_path,
        note=note,
    )


def parse_expert_labels(payload: Any) -> ExpertLabelDataset:
    payload = _require_object(payload, "root")
    _check_keys(payload, "root", required={"schema_version", "shots"}, optional=set())
    schema_version = int(payload["schema_version"])
    if schema_version != EXPERT_LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported expert-label schema version: {schema_version}; "
            f"expected {EXPERT_LABEL_SCHEMA_VERSION}"
        )
    if not isinstance(payload["shots"], list):
        raise ValueError("root.shots must be a list")
    shots = tuple(
        _parse_shot(item, i) for i, item in enumerate(payload["shots"])
    )
    duplicates = sorted(
        shot_id for shot_id in {shot.shot_id for shot in shots}
        if sum(shot.shot_id == shot_id for shot in shots) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate expert shot_id values: {duplicates}")
    return ExpertLabelDataset(shots=shots, schema_version=schema_version)


def load_expert_labels(input_path: str | Path) -> ExpertLabelDataset:
    path = Path(input_path)
    return parse_expert_labels(json.loads(path.read_text(encoding="utf-8")))
