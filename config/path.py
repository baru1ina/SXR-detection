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

# Stable feature-ML artifact paths. Override them with CLI arguments when needed.
DEFAULT_PSEUDO_LABELS_PATH = "data/dataset/wavelet_pseudo_labels.json"
DEFAULT_CANDIDATES_PATH = "data/dataset/feature_ml_candidates.json"
DEFAULT_FEATURES_PATH = "data/dataset/feature_candidates.npz"
DEFAULT_MODEL_PATH = "data/model_data/feature_ml.joblib"
DEFAULT_METRICS_PATH = "data/model_data/feature_ml_metrics.json"
