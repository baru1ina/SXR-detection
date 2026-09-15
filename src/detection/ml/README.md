# ML package structure

`models/` contains runtime code used while detecting crashes:

- `feature_ml_detector.py` — loads and applies the feature classifier;
- `hybrid_proposal_detector.py` — combines POSR with raw CPD proposals.

`train/` contains code used only to prepare datasets or train models:

- `manifest.py` — discovers expert-classified shots and splits them by shot;
- `pseudo_labels.py` — generates wavelet teacher labels;
- `candidate_dataset.py` — builds labeled POSR + raw CPD proposals;
- `feature_classifier.py` — extracts features, trains and evaluates the classifier;
- `feature_ml_pipeline.py` — runs the complete feature-ML training sequence.

`multichannel_features.py` defines the shared 163-feature schema used in both
training and runtime. `multisxr_proposals.py` unions nearby POSR/CPD proposals
from the configured SXR channels, retaining single-channel proposals and adding
source-evidence features. The v3 model stores the schema, selected diagnostic
channel profile, proposal settings, downsample, and effective time step. Older
v1/v2 `.joblib` artifacts cannot be used by the refactored detector.
Training writes to `_v3` artifact names by default, leaving old files intact.
`--multichannel` now asks for at least `--min_channels` SXR sources after ML
classification; without it, all candidates remain eligible.

Canonical commands from the repository root:

```bash
python -m src.detection.ml.train.pseudo_labels --source-dir data/raw
python main.py --mode train --method feature_ml
python main.py --mode detect --method feature_ml --model_path data/model_data/feature_ml_v3.joblib --file easy/sht46699.SHT --source_dir data/raw --downsample 10
```
