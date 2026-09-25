# ML package structure

`models/` contains runtime code used while detecting crashes:

- `feature_ml_detector.py` — loads and applies the feature classifier;
- `hybrid_proposal_detector.py` — selects raw POSR, raw CPD, or their union.

`train/` contains code used only to prepare datasets or train models:

- `manifest.py` — discovers expert-classified shots and splits them by shot;
- `pseudo_labels.py` — generates wavelet teacher labels;
- `expert_labels.py` — validates versioned expert event annotations;
- `candidate_dataset.py` — builds labeled proposals for the selected source mode;
- `feature_classifier.py` — extracts features, trains and evaluates the classifier;
- `feature_ml_pipeline.py` — runs the complete feature-ML training sequence.

`annotation/` contains the browser-based expert labeling workflow:

- `app.py` — Streamlit UI for signals, reviewed intervals, and event labels;
- `backend.py` — shot catalog, multirate signal loading, decimation, and hints;
- `store.py` — validated atomic writes with concurrent-edit detection.

`multichannel_features.py` defines the `reference_sxr` (16),
`core_diagnostics` (154), and `full` (218) schemas used in training and runtime.
`multisxr_proposals.py` unions nearby raw POSR/CPD proposals
from the configured SXR channels, retaining single-channel proposals and adding
source-evidence features. The selected source mode (`posr`, `cpd`, or `hybrid`)
runs independently on each available SXR. POSR uses its raw ML path: no physics
score filter, sawtooth mask, jump filter, or period NMS is applied before ML.
One merged event contributes one training row, with shape/noise features
for each SXR channel, including which method proposed that channel. The model
stores the schema, selected diagnostic channel profile, proposal settings,
downsample, and effective time step. Incompatible old `.joblib` artifacts must
be retrained. Schema versions are stored inside artifacts, not in their names.
Default output names are stable and each run overwrites its matching artifact;
CLI path arguments can be used to keep a specially named copy.
`--multichannel` now asks for at least `--min_channels` SXR sources after ML
classification; without it, all candidates remain eligible.

Wavelet pseudo-label generation evaluates each available SXR independently,
requiring per-channel SNR >= 2 but not ACF/FFT acceptance. Events supported by
at least two channels become teacher labels; single-channel events are marked
uncertain and are excluded from negative training labels nearby. The ML reference
SXR is selected by SNR per shot rather than fixed to 50 mkm. This is still
pseudo-labeling and requires expert review on hard shots. Teacher wavelets use an
independent fixed period seed (3 ms), then refine from each channel's events.
The teacher-label file defaults to `wavelet_pseudo_labels.json`. Expert labels
default to `data/annotations/expert_events.json`; inside explicit reviewed time
intervals they override wavelet labels, while unreviewed regions retain wavelet
supervision. Missing expert events are injected into the training candidates,
but proposal recall is measured before injection.
Expert schema 2 also stores the shot-level `sawtooth`, `no_sawtooth`, or
`uncertain` decision. Incorrect automatic hints are stored as
`false_positive` events with their source and become negatives only inside
explicit reviewed intervals.
Teacher events and ML proposals are gated until the smoothed plasma current has
remained above 80% of its robust plateau level for 0.5 ms. The gate stays open
during later ramp-down.

Canonical commands from the repository root:

```bash
python -m pip install -r requirements-ui.txt
MPLCONFIGDIR=/tmp/mpl streamlit run src/detection/ml/annotation/app.py
python -m src.detection.ml.train.pseudo_labels --source-dir data/raw
python main.py --mode train --method feature_ml --ml_proposal_source posr --ml_feature_profile full
python main.py --mode detect --method feature_ml --file easy/sht46699.SHT --source_dir data/raw
```

The default artifact paths are defined in `config/path.py`:
`data/dataset/wavelet_pseudo_labels.json`,
`data/annotations/expert_events.json`,
`data/dataset/feature_ml_candidates_<source>.json`,
`data/dataset/feature_candidates_<source>_<profile>.npz`,
`data/model_data/feature_ml_<source>_<model>_<profile>.joblib`, and
`data/model_data/feature_ml_metrics_<source>_<model>_<profile>.json`. Here `<source>` is
`posr`, `cpd`, or `hybrid`, while `<model>` is the actual backend such as
`sklearn_hgb`, `catboost`, or `xgboost`; `<profile>` is `reference_sxr`,
`core_diagnostics`, or `full`. To save a model under another name,
pass `--model_path` to training and detection; other output paths have matching
CLI parameters.
Detection gets the proposal source, thresholds, CPD settings, SXR list,
coincidence window, and downsample from the selected model rather than CLI.
