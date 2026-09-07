from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Sequence


EXPERT_CATEGORIES = ("easy", "medium", "hard", "notSaw")


@dataclass(frozen=True)
class ShotRecord:
    shot_id: str
    category: str
    relative_path: str


@dataclass(frozen=True)
class ShotSplit:
    train: tuple[ShotRecord, ...]
    validation: tuple[ShotRecord, ...]


def _shot_id(path: Path) -> str:
    return path.stem.casefold()


def build_shot_manifest(
    source_dir: str | Path,
    categories: Sequence[str] = EXPERT_CATEGORIES,
) -> list[ShotRecord]:

    root = Path(source_dir)
    print(f"Building shot manifest from {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[ShotRecord] = []
    seen: dict[str, ShotRecord] = {}

    for category in categories:
        category_dir = root / category
        if not category_dir.exists():
            continue
        if not category_dir.is_dir():
            raise NotADirectoryError(f"Category path is not a directory: {category_dir}")

        files = sorted(
            (path for path in category_dir.iterdir() if path.is_file() and path.suffix.casefold() == ".sht"),
            key=lambda path: path.name.casefold(),
        )
        for path in files:
            shot_id = _shot_id(path)
            record = ShotRecord(
                shot_id=shot_id,
                category=category,
                relative_path=path.relative_to(root).as_posix(),
            )

            if shot_id in seen:
                previous = seen[shot_id]
                raise ValueError(
                    f"Duplicate shot_id {shot_id!r}: "
                    f"{previous.relative_path!r} and {record.relative_path!r}"
                )

            seen[shot_id] = record
            records.append(record)

    return sorted(records, key=lambda record: (record.category.casefold(), record.shot_id))


def stratified_shot_split(
    records: Iterable[ShotRecord],
    validation_ratio: float = 0.2,
    random_state: int = 42,
) -> ShotSplit:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in the [0, 1) interval")

    grouped: dict[str, list[ShotRecord]] = {}
    seen_ids: set[str] = set()
    for record in records:
        if record.shot_id in seen_ids:
            raise ValueError(f"Duplicate shot_id in manifest: {record.shot_id!r}")
        seen_ids.add(record.shot_id)
        grouped.setdefault(record.category, []).append(record)

    rng = random.Random(random_state)
    train: list[ShotRecord] = []
    validation: list[ShotRecord] = []

    for category in sorted(grouped, key=str.casefold):
        category_records = sorted(grouped[category], key=lambda record: record.shot_id)
        rng.shuffle(category_records)

        if validation_ratio == 0.0 or len(category_records) < 2:
            validation_count = 0
        else:
            validation_count = max(1, round(len(category_records) * validation_ratio))
            validation_count = min(validation_count, len(category_records) - 1)

        validation.extend(category_records[:validation_count])
        train.extend(category_records[validation_count:])

    sort_key = lambda record: (record.category.casefold(), record.shot_id)
    return ShotSplit(
        train=tuple(sorted(train, key=sort_key)),
        validation=tuple(sorted(validation, key=sort_key)),
    )
