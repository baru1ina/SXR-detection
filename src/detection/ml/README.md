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

Canonical commands from the repository root:

```powershell
python -m src.detection.ml.train.pseudo_labels --source-dir data/raw
python main.py --mode train --method feature_ml
python main.py --mode detect --method feature_ml --model_path data/model_data/feature_ml.joblib --file easy/sht46699.SHT --source_dir data/raw
```
