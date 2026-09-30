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
- `backend.py` — shot catalog, multirate signal loading, and decimation;
- `store.py` — validated atomic writes with concurrent-edit detection.

`store.py` is shared unchanged by Windows and Linux and uses the
cross-platform `filelock` dependency from `requirements-ui.txt`.

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

On Windows PowerShell, run the annotation UI with the virtual environment's
Python directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/detection/ml/annotation/app.py
```

The annotation screen uses Plotly WebGL traces. Action buttons and all
marker/event interaction live in a right-hand workspace. The four action
buttons embed the PNG assets from `annotation/media` and retain full action
names as hover tooltips. Their order is add, mark false, select interval, and
inspect/edit. Free graph clicks
accumulate in a persistent queue only in add-event and interval modes. Inspect
responds only to existing teacher/manual markers, while reject responds only to
teacher markers. Source paths have their own collapsible section in the left
sidebar. A separate file-navigation section groups file search, category/status
filters, and the file selector together. The channel
section uses a case-insensitive name search and a fixed-height scrollable
alphabetically sorted `st.pills` list. Selection is shown by each pill's
highlight, without a second summary area. Filtering preserves hidden selected
channels, and clearing the query restores the complete list.
Display settings and a collapsible UI log also remain in the left sidebar;
file-level decisions live in the right workspace.
The graph-view section mirrors the live Plotly viewport into a two-column X/Y
bounds grid. X is shared, while Y ranges are retained separately per selected
channel. A small Streamlit v2 browser listener captures `plotly_relayout` after
zoom or pan and writes the actual ranges back to session state; this requires
Streamlit 1.64 or newer. A DOM observer and periodic fallback reattach the
listener whenever Streamlit replaces or reinitializes the Plotly node.
Current-crop bounds are also applied as Plotly
`minallowed`/`maxallowed` constraints. Reset restores the permitted X interval
and every channel's natural Y range.
The shot-level `no_sawtooth` decision is selected only through the file-content
field; there is no second shortcut that mutates the status, intervals, or events.
The reviewed-interval list is not exposed as a separate editor control. Event
actions maintain local trusted intervals internally, and selecting a complete
review records the full signal range automatically.
A refresh button below the file decision reloads the annotation document and
catalog, allowing an unannotated-only workflow to advance to the next file
after saving.
Annotation-rail markers also draw source-colored vertical lines through every
signal panel so their intersections with the diagnostics remain visible.
The marker rail is a taller bottom row with three non-duplicating levels:
rejected, teacher, and manual events. An
interval tool uses two graph clicks as boundaries and collects every manual and
saved teacher marker between them immediately. Its two editable boundary fields
recompute the selection automatically; there is no separate collect button.
Signal-row height is adjustable from the sidebar. The figure also reserves a
48-pixel vertical gap per subplot boundary so channel titles do not overlap,
without taking that space out of the requested signal-row height.
The right workspace is a wider bordered panel occupying roughly one third of
the main area; no width control is placed in the left sidebar. Native Streamlit
provides only its single left `st.sidebar` and no draggable `st.columns` divider.
Signal panels use a single axis border without an additional outer rectangle or
translucent fill. A `Grid` checkbox in the graph-view controls toggles grid lines
for all signal panels while leaving the annotation rail clear. The larger marker
legend is separated from the upper signal panel by extra spacing. The marker
rail has visible bottom and left axes, no mirrored frame, and a horizontal guide
for each of its three event levels. These guides are independent of the signal
panel grid toggle.
Manual events with either `label=uncertain` or `confidence=uncertain` share one
visual rule in every workflow: a yellow open circle and a yellow dashed line.
The UI is annotation-only: it does not run raw POSR, raw CPD, feature-ML, model
training, or candidate regeneration. It displays the already persisted wavelet
teacher as context. Ordinary clicks on a signal stage times for one batch
confirmation only in add-event and interval modes. The left sidebar contains files, search,
filters, channels, display settings, and a collapsible UI log; shot decisions
and saving are in the right workspace.
The pending queue is edited through the removal dropdown in the right workspace,
without a duplicated gray text list, and highlighted in Plotly: existing markers
get a cyan outline, while selected times get a dashed line
through all signal panels and a translucent tolerance band. The annotation chart uses a stable
`uirevision` and a single selection callback, while normal Streamlit widget
identity ensures updated markers are actually rendered.
Single-event editing hides the wavelet source/score and exposes only event time
and reference SXR. Uniform buttons mark an item as a sawtooth, possible
sawtooth, or false sawtooth. Tolerance is fixed at 0.3 ms, confidence follows
the chosen action, and wavelet `source_channels` populate channel metadata
automatically; type, visible-channel, confidence, and comment fields are not
delegated to the annotator.
Channel selection, source directory, per-signal height, and current X/Y bounds
are included in a separate view revision, ensuring those controls rebuild the
chart when changed. The graph-view section stores the viewport per shot.
An optional current crop reuses `detect_plasma_interval` with the legacy
`Ip > 20% * max(Ip)` rule and initializes those bounds to the detected plasma
interval. Explicit bounds are reapplied after every rerun, providing a stable
view while annotations change; this display-only crop does not alter labels or
training data.
Inspect mode is strictly single-event: every marker click replaces the current
selection, and even an accumulated Plotly Shift selection is collapsed to one
new marker. Teacher events expose sawtooth, possible-sawtooth, and false-
sawtooth actions. Saved manual events expose sawtooth, possible-sawtooth, and
manual deletion; inapplicable actions remain visible but disabled.
The obsolete separate click-scale rail has been removed.

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
