# source_dir="data/raw/test"
# source_dir="data/raw/easy"
source_dir="data/raw/medium"
# source_dir="data/raw/hard"
# source_dir="data/raw/notSaw"

SXR_CHANNELS = [
    "SXR 15 мкм",
    "SXR 27 мкм",
    "SXR 50 mkm",
    "SXR 80 mkm",
    "SXR 127 мкм"
]

IP_CHANNEL = "Ip внутр.(Пр2ВК) (инт.18)"

ALL_CHANNELS = SXR_CHANNELS + [IP_CHANNEL]

PATH_TO_RES_AUTOCORR = 'data/process/autocorr'
PATH_TO_RES_TEMP = 'data/process/temp_res'

# Feature-ML artifacts.  Pseudo-labels do not depend on the proposal source;
# downstream artifacts include it so experimental models cannot be confused.
DEFAULT_PSEUDO_LABELS_PATH = "data/dataset/wavelet_pseudo_labels.json"
DEFAULT_EXPERT_LABELS_PATH = "data/annotations/expert_events.json"
FEATURE_ML_PROPOSAL_SOURCES = ("posr", "cpd", "hybrid")
DEFAULT_FEATURE_ML_PROPOSAL_SOURCE = "posr"
DEFAULT_FEATURE_ML_BACKEND = "sklearn_hgb"
DEFAULT_FEATURE_ML_PROFILE = "full"


def feature_ml_artifact_paths(
    proposal_source=DEFAULT_FEATURE_ML_PROPOSAL_SOURCE,
    model_name=DEFAULT_FEATURE_ML_BACKEND,
    feature_profile=DEFAULT_FEATURE_ML_PROFILE,
):
    proposal_source = str(proposal_source).lower()
    if proposal_source not in FEATURE_ML_PROPOSAL_SOURCES:
        raise ValueError(
            f"Unsupported feature-ML proposal source {proposal_source!r}; "
            f"expected one of {FEATURE_ML_PROPOSAL_SOURCES}"
        )
    model_name = str(model_name).lower().strip()
    if not model_name or not all(char.isalnum() or char == "_" for char in model_name):
        raise ValueError(
            "Feature-ML model name must contain only letters, digits, and underscores"
        )
    feature_profile = str(feature_profile).lower().strip()
    if feature_profile not in {"reference_sxr", "core_diagnostics", "full"}:
        raise ValueError(
            "Feature-ML profile must be one of: reference_sxr, "
            "core_diagnostics, full"
        )
    return {
        "candidates": f"data/dataset/feature_ml_candidates_{proposal_source}.json",
        "features": (
            f"data/dataset/feature_candidates_{proposal_source}_{feature_profile}.npz"
        ),
        "model": (
            f"data/model_data/feature_ml_{proposal_source}_{model_name}_"
            f"{feature_profile}.joblib"
        ),
        "metrics": (
            f"data/model_data/feature_ml_metrics_{proposal_source}_{model_name}_"
            f"{feature_profile}.json"
        ),
    }


_DEFAULT_FEATURE_ML_PATHS = feature_ml_artifact_paths()
DEFAULT_CANDIDATES_PATH = _DEFAULT_FEATURE_ML_PATHS["candidates"]
DEFAULT_FEATURES_PATH = _DEFAULT_FEATURE_ML_PATHS["features"]
DEFAULT_MODEL_PATH = _DEFAULT_FEATURE_ML_PATHS["model"]
DEFAULT_METRICS_PATH = _DEFAULT_FEATURE_ML_PATHS["metrics"]
