from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from src.detection.ml.train.candidate_dataset import (
    CandidateDataset,
    FeatureMLCandidateConfig,
    build_candidate_dataset,
    save_candidate_dataset,
)
from src.detection.ml.train.pseudo_labels import load_pseudo_labels
from src.detection.ml.train.feature_classifier import (
    save_feature_dataset,
    save_model,
    save_report,
    train_feature_classifier,
)


@dataclass(frozen=True)
class FeatureMLTrainingArtifacts:
    candidates_path: Path
    features_path: Path
    model_path: Path
    metrics_path: Path
    dataset: CandidateDataset
    report: dict


def train_feature_ml_from_pseudo_labels(
    logger,
    pseudo_labels_path: str | Path = "data/dataset/wavelet_pseudo_labels.json",
    source_dir: Optional[str | Path] = None,
    candidates_output: str | Path = "data/dataset/feature_ml_candidates.json",
    features_output: str | Path = "data/dataset/feature_candidates.npz",
    model_output: str | Path = "data/model_data/feature_ml.joblib",
    metrics_output: str | Path = "data/model_data/feature_ml_metrics.json",
    backend: str = "sklearn_hgb",
    random_state: int = 42,
    downsample: Optional[int] = None,
    posr_threshold: float = 3.0,
    posr_score_threshold: float = 0.5,
    use_cpd: bool = True,
    cpd_penalty: float = 3.0,
    cpd_model: str = "rbf",
    positive_tolerance_s: float = 0.3e-3,
    negative_exclusion_s: float = 0.6e-3,
    debug: bool = False,
) -> FeatureMLTrainingArtifacts:
    pseudo_labels = load_pseudo_labels(pseudo_labels_path)
    if source_dir is not None:
        pseudo_labels = replace(pseudo_labels, source_dir=str(source_dir))

    effective_downsample = downsample or pseudo_labels.config.downsample
    config = FeatureMLCandidateConfig(
        downsample=effective_downsample,
        threshold=posr_threshold,
        score_threshold=posr_score_threshold,
        use_cpd=use_cpd,
        cpd_penalty=cpd_penalty,
        cpd_model=cpd_model,
        positive_tolerance_s=positive_tolerance_s,
        negative_exclusion_s=negative_exclusion_s,
    )
    dataset = build_candidate_dataset(
        pseudo_labels,
        logger=logger,
        config=config,
        debug=debug,
    )
    candidates_path = save_candidate_dataset(dataset, candidates_output)
    logger.info(f"Candidate dataset saved to {candidates_path.resolve()}")
    logger.info(f"Candidate summary: {dataset.summary()}")

    model, train_rows, validation_rows, report = train_feature_classifier(
        dataset,
        logger=logger,
        backend=backend,
        random_state=random_state,
    )
    features_path = save_feature_dataset(train_rows, validation_rows, features_output)
    effective_dt = dataset.config.downsample * next(
        shot.dt_s
        for shot in dataset.train + dataset.validation
        if shot.dt_s is not None
    )
    model_path = save_model(model, report, model_output, effective_dt)
    metrics_path = save_report(report, metrics_output)

    logger.info(f"Feature dataset saved to {features_path.resolve()}")
    logger.info(f"Model saved to {model_path.resolve()}")
    logger.info(f"Metrics saved to {metrics_path.resolve()}")
    logger.info(
        "Validation metrics: "
        f"{report['metrics_at_selected_threshold']['validation']}"
    )
    return FeatureMLTrainingArtifacts(
        candidates_path=candidates_path,
        features_path=features_path,
        model_path=model_path,
        metrics_path=metrics_path,
        dataset=dataset,
        report=report,
    )
