from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import fcntl

from src.detection.ml.train.expert_labels import (
    EXPERT_LABEL_SCHEMA_VERSION,
    parse_expert_labels,
)


class AnnotationConflictError(RuntimeError):
    """The annotation file changed after the browser loaded its draft."""


@dataclass(frozen=True)
class AnnotationDocument:
    payload: dict[str, Any]
    revision: str


def _empty_payload() -> dict[str, Any]:
    return {"schema_version": EXPERT_LABEL_SCHEMA_VERSION, "shots": []}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_annotation_document(path: str | Path) -> AnnotationDocument:
    annotation_path = Path(path)
    if annotation_path.exists():
        data = annotation_path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    else:
        payload = _empty_payload()
        data = _canonical_bytes(payload)
    parse_expert_labels(payload)
    return AnnotationDocument(payload=payload, revision=_revision(data))


def upsert_shot_annotation(
    path: str | Path,
    shot_payload: dict[str, Any],
    expected_revision: str | None = None,
) -> AnnotationDocument:
    """Validate and atomically upsert one shot, detecting concurrent edits."""

    annotation_path = Path(path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = annotation_path.with_suffix(annotation_path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load_annotation_document(annotation_path)
        if expected_revision is not None and current.revision != expected_revision:
            raise AnnotationConflictError(
                "Annotation file changed in another session; reload before saving"
            )
        candidate = {
            "schema_version": EXPERT_LABEL_SCHEMA_VERSION,
            "shots": [
                item
                for item in current.payload["shots"]
                if str(item.get("shot_id", "")).casefold()
                != str(shot_payload.get("shot_id", "")).casefold()
            ] + [shot_payload],
        }
        parse_expert_labels(candidate)
        candidate["shots"] = sorted(
            candidate["shots"], key=lambda item: str(item["shot_id"]).casefold()
        )
        data = _canonical_bytes(candidate)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=annotation_path.parent,
                prefix=f".{annotation_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, annotation_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return AnnotationDocument(payload=candidate, revision=_revision(data))
