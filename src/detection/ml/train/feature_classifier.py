import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.detection.ml.train.candidate_dataset import (
    CandidateDataset,
    CandidateShot,
    LabeledCandidate,
    load_candidate_dataset,
)
from src.detection.ml.models.feature_ml_detector import (
    FEATURE_NAMES,
    FeatureMLMetadata,
)
from src.detection.utils.features import extract_candidate_features


@dataclass(frozen=True)
class FeatureRows:
    X: np.ndarray
    y: np.ndarray
    shot_ids: np.ndarray
    categories: np.ndarray
    candidate_times_s: np.ndarray
    teacher_event_keys: np.ndarray
    teacher_event_count: int
    teacher_event_counts_by_category: dict[str, int]
    teacher_event_counts_by_shot: dict[str, int]


PrepareShotFn = Callable[..., object]
FeatureExtractor = Callable[[np.ndarray, int, float, Optional[float]], np.ndarray]


def _make_model(backend: str = "auto", random_state: int = 42, catboost=None):
    backend = backend.lower()
    if backend in {"auto", "catboost"}:
        try:
            from catboost import CatBoostClassifier

            return "catboost", CatBoostClassifier(
                iterations=300,
                depth=5,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=random_state,
                auto_class_weights="Balanced",
            )
        except Exception:
            if backend == "catboost":
                raise

    if backend in {"auto", "xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier

            return "xgboost", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
            )
        except Exception:
            if backend in {"xgboost", "xgb"}:
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    return "sklearn_hgb", HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=random_state,
        class_weight="balanced",
    )


class FeatureMLTrainer:
    """Train a candidate classifier from known/pseudo crash indices."""

    def __init__(self, dt: float, period: Optional[float] = None, backend: str = "auto"):
        self.dt = float(dt)
        self.period = period
        self.backend_requested = backend
        self.backend: Optional[str] = None
        self.model = None

    def build_dataset_from_indices(
        self,
        signal: np.ndarray,
        crash_indices: Sequence[int],
        negative_indices: Optional[Sequence[int]] = None,
        n_negative_per_positive: int = 4,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        n = len(signal)
        positives = np.array(sorted(set(map(int, crash_indices))), dtype=int)
        positives = positives[(positives > 3) & (positives < n - 3)]

        if negative_indices is None:
            if self.period is None or not np.isfinite(self.period) or self.period <= 0:
                guard = max(10, int(0.5e-3 / self.dt))
            else:
                guard = max(10, int(0.35 * self.period / self.dt))
            possible = np.arange(guard, max(guard + 1, n - guard))
            for positive in positives:
                possible = possible[np.abs(possible - positive) > guard]
            count = min(
                len(possible),
                max(len(positives) * n_negative_per_positive, 1),
            )
            negatives = (
                rng.choice(possible, size=count, replace=False)
                if count > 0
                else np.array([], dtype=int)
            )
        else:
            negatives = np.array(sorted(set(map(int, negative_indices))), dtype=int)

        features = []
        labels = []
        for index in positives:
            features.append(
                extract_candidate_features(signal, int(index), self.dt, self.period)
            )
            labels.append(1)
        for index in negatives:
            features.append(
                extract_candidate_features(signal, int(index), self.dt, self.period)
            )
            labels.append(0)

        if not features:
            raise ValueError("No training samples were generated")
        return np.vstack(features), np.asarray(labels, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.backend, self.model = _make_model(self.backend_requested)
        self.model.fit(X, y)
        return self

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": FeatureMLMetadata(
                    backend=self.backend or self.backend_requested,
                    dt=self.dt,
                    period=self.period,
                    feature_names=FEATURE_NAMES,
                ),
            },
            path,
        )


def _default_prepare_shot(**kwargs):
    from src.detection.utils.pipeline_utils import prepare_shot_for_detection

    return prepare_shot_for_detection(**kwargs)


def _teacher_event_key(shot_id: str, candidate: LabeledCandidate) -> str:
    if candidate.label != 1 or candidate.closest_teacher_time_s is None:
        return ""
    return f"{shot_id}:{candidate.closest_teacher_time_s:.12f}"


def extract_feature_rows(
    shots: tuple[CandidateShot, ...],
    source_dir: str | Path,
    downsample: int,
    logger,
    prepare_shot: PrepareShotFn = _default_prepare_shot,
    feature_extractor: FeatureExtractor = extract_candidate_features,
) -> FeatureRows:
    features: list[np.ndarray] = []
    labels: list[int] = []
    shot_ids: list[str] = []
    categories: list[str] = []
    candidate_times: list[float] = []
    teacher_event_keys: list[str] = []
    teacher_event_count = sum(shot.teacher_event_count for shot in shots)
    teacher_event_counts_by_category = dict(
        Counter(
            shot.category
            for shot in shots
            for _ in range(shot.teacher_event_count)
        )
    )
    teacher_event_counts_by_shot = {
        shot.shot_id: shot.teacher_event_count for shot in shots
    }

    for shot in shots:
        if shot.status != "candidates_built" or not shot.candidates:
            continue

        logger.info(f"Extracting ML features for {shot.relative_path}")
        prepared = prepare_shot(
            source_dir=str(source_dir),
            filename=shot.relative_path,
            logger=logger,
            channel_name=shot.channel,
            channels=[shot.channel],
            require_sawtooth=False,
        )
        if prepared is None:
            raise RuntimeError(f"Could not prepare {shot.relative_path} for feature extraction")

        signal = np.asarray(prepared.reference_signal, dtype=float)[::downsample]
        effective_dt = float(prepared.dt) * downsample
        period = float(shot.estimated_period_s or prepared.estimated_period)

        for candidate in shot.candidates:
            if candidate.plasma_index % downsample != 0:
                raise ValueError(
                    f"Candidate index {candidate.plasma_index} in {shot.shot_id} "
                    f"is not aligned to downsample={downsample}"
                )
            candidate_index = candidate.plasma_index // downsample
            if not 0 <= candidate_index < len(signal):
                raise IndexError(
                    f"Candidate index {candidate_index} is outside {shot.shot_id} signal"
                )

            vector = np.asarray(
                feature_extractor(signal, candidate_index, effective_dt, period),
                dtype=float,
            )
            if vector.shape != (len(FEATURE_NAMES),):
                raise ValueError(
                    f"Expected {len(FEATURE_NAMES)} features, got {vector.shape} "
                    f"for {shot.shot_id}"
                )

            features.append(vector)
            labels.append(int(candidate.label))
            shot_ids.append(shot.shot_id)
            categories.append(shot.category)
            candidate_times.append(float(candidate.time_s))
            teacher_event_keys.append(_teacher_event_key(shot.shot_id, candidate))

    if not features:
        raise ValueError("No candidate features were extracted")

    return FeatureRows(
        X=np.vstack(features),
        y=np.asarray(labels, dtype=int),
        shot_ids=np.asarray(shot_ids),
        categories=np.asarray(categories),
        candidate_times_s=np.asarray(candidate_times, dtype=float),
        teacher_event_keys=np.asarray(teacher_event_keys),
        teacher_event_count=teacher_event_count,
        teacher_event_counts_by_category=teacher_event_counts_by_category,
        teacher_event_counts_by_shot=teacher_event_counts_by_shot,
    )


def _positive_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def classification_metrics(
    rows: FeatureRows,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(rows.y, predicted, labels=[0, 1]).ravel()
    positive_recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    if positive_recall is not None and specificity is not None:
        balanced_accuracy = 0.5 * (positive_recall + specificity)
    else:
        balanced_accuracy = positive_recall if positive_recall is not None else specificity
    predicted_teacher_keys = {
        key
        for key, target, prediction in zip(rows.teacher_event_keys, rows.y, predicted)
        if key and target == 1 and prediction == 1
    }
    metrics = {
        "threshold": float(threshold),
        "samples": int(len(rows.y)),
        "positive_samples": int(np.sum(rows.y == 1)),
        "negative_samples": int(np.sum(rows.y == 0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "accuracy": float(accuracy_score(rows.y, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy) if balanced_accuracy is not None else None
        ),
        "precision": float(precision_score(rows.y, predicted, zero_division=0)),
        "candidate_recall": float(recall_score(rows.y, predicted, zero_division=0)),
        "specificity": float(specificity) if specificity is not None else None,
        "false_positive_rate": (
            float(1.0 - specificity) if specificity is not None else None
        ),
        "f1": float(f1_score(rows.y, predicted, zero_division=0)),
        "teacher_event_recall": (
            len(predicted_teacher_keys) / rows.teacher_event_count
            if rows.teacher_event_count
            else None
        ),
    }
    if len(np.unique(rows.y)) == 2:
        metrics["average_precision"] = float(average_precision_score(rows.y, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(rows.y, probabilities))
    else:
        metrics["average_precision"] = None
        metrics["roc_auc"] = None
    return metrics


def select_f1_threshold(rows: FeatureRows, probabilities: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                np.asarray(probabilities, dtype=float),
            ]
        )
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        f1 = f1_score(rows.y, predicted, zero_division=0)
        recall = recall_score(rows.y, predicted, zero_division=0)
        key = (float(f1), float(recall), -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _metrics_by_category(
    rows: FeatureRows,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    result = {}
    for category in sorted(set(rows.categories), key=str.casefold):
        mask = rows.categories == category
        subset = FeatureRows(
            X=rows.X[mask],
            y=rows.y[mask],
            shot_ids=rows.shot_ids[mask],
            categories=rows.categories[mask],
            candidate_times_s=rows.candidate_times_s[mask],
            teacher_event_keys=rows.teacher_event_keys[mask],
            teacher_event_count=rows.teacher_event_counts_by_category.get(str(category), 0),
            teacher_event_counts_by_category={
                str(category): rows.teacher_event_counts_by_category.get(str(category), 0)
            },
            teacher_event_counts_by_shot={
                str(shot_id): count
                for shot_id, count in rows.teacher_event_counts_by_shot.items()
                if shot_id in set(rows.shot_ids[mask])
            },
        )
        result[str(category)] = classification_metrics(subset, probabilities[mask], threshold)
    return result


def _metrics_by_shot(
    rows: FeatureRows,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, dict]:
    result = {}
    for shot_id in sorted(set(rows.shot_ids)):
        mask = rows.shot_ids == shot_id
        category = str(rows.categories[mask][0])
        teacher_event_count = rows.teacher_event_counts_by_shot.get(str(shot_id), 0)
        subset = FeatureRows(
            X=rows.X[mask],
            y=rows.y[mask],
            shot_ids=rows.shot_ids[mask],
            categories=rows.categories[mask],
            candidate_times_s=rows.candidate_times_s[mask],
            teacher_event_keys=rows.teacher_event_keys[mask],
            teacher_event_count=teacher_event_count,
            teacher_event_counts_by_category={category: teacher_event_count},
            teacher_event_counts_by_shot={str(shot_id): teacher_event_count},
        )
        result[str(shot_id)] = {
            "category": category,
            **classification_metrics(subset, probabilities[mask], threshold),
        }
    return result


def save_feature_dataset(
    train_rows: FeatureRows,
    validation_rows: FeatureRows,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X_train=train_rows.X,
        y_train=train_rows.y,
        train_shot_ids=train_rows.shot_ids,
        train_categories=train_rows.categories,
        train_candidate_times_s=train_rows.candidate_times_s,
        X_validation=validation_rows.X,
        y_validation=validation_rows.y,
        validation_shot_ids=validation_rows.shot_ids,
        validation_categories=validation_rows.categories,
        validation_candidate_times_s=validation_rows.candidate_times_s,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    return output


def train_feature_classifier(
    dataset: CandidateDataset,
    logger,
    backend: str = "sklearn_hgb",
    random_state: int = 42,
    prepare_shot: PrepareShotFn = _default_prepare_shot,
    feature_extractor: FeatureExtractor = extract_candidate_features,
):
    downsample = dataset.config.downsample
    train_rows = extract_feature_rows(
        dataset.train,
        dataset.source_dir,
        downsample,
        logger,
        prepare_shot,
        feature_extractor,
    )
    validation_rows = extract_feature_rows(
        dataset.validation,
        dataset.source_dir,
        downsample,
        logger,
        prepare_shot,
        feature_extractor,
    )

    if len(np.unique(train_rows.y)) != 2:
        raise ValueError("Training data must contain both positive and negative candidates")
    if len(np.unique(validation_rows.y)) != 2:
        raise ValueError("Validation data must contain both positive and negative candidates")

    backend_name, model = _make_model(backend=backend, random_state=random_state)
    model.fit(train_rows.X, train_rows.y)
    train_probability = _positive_probability(model, train_rows.X)
    validation_probability = _positive_probability(model, validation_rows.X)
    selected_threshold = select_f1_threshold(validation_rows, validation_probability)

    report = {
        "schema_version": 1,
        "backend": backend_name,
        "random_state": random_state,
        "feature_names": FEATURE_NAMES,
        "candidate_config": asdict(dataset.config),
        "selected_threshold": selected_threshold,
        "metrics_at_0_5": {
            "train": classification_metrics(train_rows, train_probability, 0.5),
            "validation": classification_metrics(validation_rows, validation_probability, 0.5),
        },
        "metrics_at_selected_threshold": {
            "train": classification_metrics(train_rows, train_probability, selected_threshold),
            "validation": classification_metrics(
                validation_rows, validation_probability, selected_threshold
            ),
        },
        "validation_by_category": _metrics_by_category(
            validation_rows, validation_probability, selected_threshold
        ),
        "validation_by_shot": _metrics_by_shot(
            validation_rows, validation_probability, selected_threshold
        ),
    }
    return model, train_rows, validation_rows, report


def save_model(model, report: dict, output_path: str | Path, effective_dt: float) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "metadata": FeatureMLMetadata(
                backend=report["backend"],
                dt=effective_dt,
                period=None,
                feature_names=FEATURE_NAMES,
            ),
            "training": {
                "selected_threshold": report["selected_threshold"],
                "candidate_config": report["candidate_config"],
                "random_state": report["random_state"],
            },
        },
        output,
    )
    return output


def save_report(report: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the feature-ML candidate classifier")
    parser.add_argument("--candidates", default="data/dataset/feature_ml_candidates.json")
    parser.add_argument("--model-output", default="data/model_data/feature_ml.joblib")
    parser.add_argument("--features-output", default="data/dataset/feature_candidates.npz")
    parser.add_argument("--report-output", default="data/model_data/feature_ml_metrics.json")
    parser.add_argument("--backend", default="sklearn_hgb")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    from src.logger import setup_logger

    logger = setup_logger(log_to_file=False)
    dataset = load_candidate_dataset(args.candidates)
    model, train_rows, validation_rows, report = train_feature_classifier(
        dataset,
        logger=logger,
        backend=args.backend,
        random_state=args.random_state,
    )
    features_path = save_feature_dataset(train_rows, validation_rows, args.features_output)
    effective_dt = dataset.config.downsample * next(
        shot.dt_s
        for shot in dataset.train + dataset.validation
        if shot.dt_s is not None
    )
    model_path = save_model(model, report, args.model_output, effective_dt)
    report_path = save_report(report, args.report_output)

    logger.info(f"Feature dataset saved to {features_path.resolve()}")
    logger.info(f"Model saved to {model_path.resolve()}")
    logger.info(f"Metrics saved to {report_path.resolve()}")
    logger.info(
        f"Validation metrics: {report['metrics_at_selected_threshold']['validation']}"
    )


if __name__ == "__main__":
    main()
