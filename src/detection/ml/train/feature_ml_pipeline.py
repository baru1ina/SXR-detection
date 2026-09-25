from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from config.path import (
    DEFAULT_EXPERT_LABELS_PATH,
    DEFAULT_PSEUDO_LABELS_PATH,
    feature_ml_artifact_paths,
)
from config.channels import get_feature_map_profile
from src.detection.ml.multichannel_features import MultichannelFeatureBuilder
from src.detection.ml.train.candidate_dataset import (
    CandidateDataset,
    FeatureMLCandidateConfig,
    build_candidate_dataset,
    save_candidate_dataset,
)
from src.detection.ml.train.pseudo_labels import load_pseudo_labels
from src.detection.ml.train.expert_labels import load_expert_labels
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
    pseudo_labels_path: str | Path = DEFAULT_PSEUDO_LABELS_PATH,
    expert_labels_path: str | Path | None = DEFAULT_EXPERT_LABELS_PATH,
    source_dir: Optional[str | Path] = None,
    candidates_output: Optional[str | Path] = None,
    features_output: Optional[str | Path] = None,
    model_output: Optional[str | Path] = None,
    metrics_output: Optional[str | Path] = None,
    backend: str = "sklearn_hgb",
    random_state: int = 42,
    downsample: Optional[int] = None,
    posr_threshold: float = 3.0,
    proposal_source: str = "posr",
    cpd_penalty: float = 3.0,
    cpd_model: str = "rbf",
    positive_tolerance_s: float = 0.3e-3,
    negative_exclusion_s: float = 0.6e-3,
    proposal_coincidence_window_s: float = 0.15e-3,
    feature_profile: str = "full",
    plateau_level_fraction: float = 0.8,
    plateau_hold_s: float = 0.5e-3,
    debug: bool = False,
) -> FeatureMLTrainingArtifacts:
    artifact_paths = feature_ml_artifact_paths(
        proposal_source,
        backend,
        feature_profile,
    )
    candidates_output = candidates_output or artifact_paths["candidates"]
    features_output = features_output or artifact_paths["features"]

    pseudo_labels = load_pseudo_labels(pseudo_labels_path)
    expert_labels = (
        load_expert_labels(expert_labels_path)
        if expert_labels_path is not None
        else None
    )
    if expert_labels is not None:
        logger.info(
            f"Loaded expert labels for {len(expert_labels.shots)} shots from "
            f"{Path(expert_labels_path).resolve()}"
        )
    if source_dir is not None:
        pseudo_labels = replace(pseudo_labels, source_dir=str(source_dir))

    effective_downsample = downsample or pseudo_labels.config.downsample
    config = FeatureMLCandidateConfig(
        downsample=effective_downsample,
        threshold=posr_threshold,
        proposal_source=proposal_source,
        cpd_penalty=cpd_penalty,
        cpd_model=cpd_model,
        positive_tolerance_s=positive_tolerance_s,
        negative_exclusion_s=negative_exclusion_s,
        coincidence_window_s=proposal_coincidence_window_s,
        plateau_level_fraction=plateau_level_fraction,
        plateau_hold_s=plateau_hold_s,
    )
    dataset = build_candidate_dataset(
        pseudo_labels,
        logger=logger,
        config=config,
        expert_labels=expert_labels,
        debug=debug,
    )
    candidates_path = save_candidate_dataset(dataset, candidates_output)
    logger.info(f"Candidate dataset saved to {candidates_path.resolve()}")
    logger.info(f"Candidate summary: {dataset.summary()}")

    feature_builder = MultichannelFeatureBuilder(
        get_feature_map_profile(feature_profile)
    )
    model, train_rows, validation_rows, report = train_feature_classifier(
        dataset,
        logger=logger,
        backend=backend,
        random_state=random_state,
        feature_builder=feature_builder,
    )
    model_artifact_paths = feature_ml_artifact_paths(
        proposal_source,
        report["backend"],
        feature_profile,
    )
    model_output = model_output or model_artifact_paths["model"]
    metrics_output = metrics_output or model_artifact_paths["metrics"]
    features_path = save_feature_dataset(
        train_rows,
        validation_rows,
        features_output,
        feature_builder=feature_builder,
    )
    effective_dt = dataset.config.downsample * next(
        shot.dt_s
        for shot in dataset.train + dataset.validation
        if shot.dt_s is not None
    )
    model_path = save_model(
        model,
        report,
        model_output,
        effective_dt,
        feature_builder=feature_builder,
    )
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
