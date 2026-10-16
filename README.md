# EEG-Based Depression Detection

This project tries to tell, from a few minutes of EEG (brain electrical activity recorded from the scalp), whether a person's brain activity looks more similar to a group of people diagnosed with major depressive disorder (MDD) or a group of healthy controls (HC). It does **not** diagnose depression — it is a research pipeline that trains and evaluates a machine learning model on a public research dataset, and reports how well that model does using proper methodology.

**A quick honesty note before anything else:** any accuracy numbers below (or in `results/`, `results_v2/`) come directly from actually running this pipeline — nothing is estimated, rounded up, or picked from a "best" run. See [Results](#results) and `report.md` for what happened when we ran it.

> **Two versions live in this repo.** The original pipeline (**v1**: eyes-closed EEG only, CWT images, ResNet50, 5-fold CV — `configs/default.yaml`, `scripts/run_pipeline.py`) is unchanged and still fully reproducible; its results are in `results/`. A newer **v2 comparison study** (`configs/channel_study.yaml`, `scripts/run_experiment_grid.py`) extends it to compare EEG channels, CNN backbones, and recording conditions against each other, using SPWVD instead of CWT and 10-fold CV; its results are in `results_v2/`. See [`report.md`](report.md) for the full writeup of what changed and why, and [`research_paper.tex`](research_paper.tex) for the conference-paper version.

---

## What EEG is, briefly

EEG (electroencephalography) records the tiny electrical signals your brain produces, using electrodes placed on the scalp in a standard layout (here, 19 electrodes in the "10-20 system", a naming scheme for electrode positions like `Fp1`, `Cz`, `O2`). This project uses **resting-state, eyes-closed** recordings — the person is just sitting quietly with their eyes closed, not doing a task.

## Dataset

**Mumtaz EEG dataset** (also known as the MDD Patients and Healthy Controls EEG Data, Mumtaz et al., 2017), recorded at Hospital Universiti Sains Malaysia:
- 34 people diagnosed with MDD, 30 healthy controls
- 19-channel EEG, 10-20 system, 256 Hz sampling rate
- Resting-state recordings, both eyes-closed (EC) and eyes-open (EO), plus a cognitive task condition
- **v1 (`configs/default.yaml`) uses eyes-closed (EC) recordings only. v2 (`configs/channel_study.yaml` + `scripts/run_experiment_grid.py`) uses all three conditions — EC, EO, and TASK — and compares them against each other.**

The dataset is not included in this repository (research EEG datasets typically carry usage/licensing terms that don't permit redistribution). If you're trying to reproduce this project, you'll need to request/download it yourself from the original source and place the `.edf` files in `data/raw/`.

**Known data-availability quirks, discovered while building this pipeline:**
- The dataset nominally has 34 MDD + 30 HC subjects, but not every subject has a recording for every condition. Usable cohorts, per condition: **EC 58 (30 MDD, 28 HC)**, **EO 60 (32 MDD, 28 HC)**, **TASK 61 (33 MDD, 28 HC)**. The code discovers whatever files are actually present rather than assuming a fixed count — see [Limitations](#known-limitations).
- Two EO files (`6921143_H S15 EO.edf`, `6921959_H S15 EO.edf`) don't follow the dataset's `<group> S<n> <condition>.edf` naming pattern and are intentionally skipped by the filename parser rather than guessed at.
- v2's 4-channel montage (Fp1, Fp2, F3, F4 — see below) causes ICA to fail on one additional EC subject, **MDD_S23** (too little signal variance across only 4 channels for the ICA solver to extract more than one component), leaving **57 of 58 EC subjects** usable for v2's EC runs specifically. This doesn't affect v1, which still uses all 19 channels.

---

## The pipeline, step by step

```
raw EEG (.edf) -> cleaning -> fixed-length windows ("epochs") -> time-frequency image -> CNN -> Normal / Depressed
```

1. **Loading.** Each recording is loaded and the 19 standard channels are picked out by name (files have a few extra non-standard channels that get dropped).

2. **Filtering.** A "bandpass filter" keeps only the frequencies that carry meaningful brain-activity information (0.5–45 Hz) and discards very slow drifts and very fast noise. A "notch filter" removes the 50 Hz electrical hum picked up from the building's power supply (50 Hz because this dataset was recorded in Malaysia, where mains electricity runs at 50 Hz — this is configurable if you plug in a different dataset from a 60 Hz country).

3. **Artifact removal.** Eye blinks and muscle tension show up in EEG as large, non-brain signals that can fool a model. This pipeline uses **ICA** (Independent Component Analysis), a technique that separates the recording into distinct underlying signal sources and lets us discard the ones that look like eye or muscle artifacts, keeping the rest. *Assumption:* this dataset has no dedicated eye-movement sensor channel, so the pipeline uses the two forehead electrodes (`Fp1`, `Fp2`) as a stand-in signal for spotting eye-blink artifacts, rather than a true dedicated channel. This is a common, reasonable workaround, but it's a real assumption worth knowing about.

4. **Re-referencing.** EEG voltage is always measured relative to *something*; this pipeline switches every channel to be measured relative to the average of all channels ("average reference"), a standard, neutral choice.

5. **Epoching.** The continuous ~5-minute recording is chopped into short fixed-length windows called "epochs" — 4 seconds each by default, with 50% overlap between consecutive windows (both are adjustable in `configs/default.yaml`). The overlap is what lets a ~58-subject dataset produce thousands of training examples. Every epoch keeps a record of which subject it came from — this bookkeeping is what makes the train/test split below possible to do correctly. Epochs with unusually large voltage swings (likely a leftover artifact) are automatically dropped.

6. **Time-frequency image.** A raw EEG epoch is a wiggly voltage-over-time signal. To use an image-based deep learning model, each epoch is converted into a 2D image that shows *which frequencies were present, and when* — brighter regions mean more of that frequency at that moment. Two ways of building this image are implemented:
   - **CWT (Continuous Wavelet Transform)** — v1's default. Good general-purpose time-frequency picture, computed with the `PyWavelets` library using a Morlet wavelet.
   - **SPWVD (Smoothed Pseudo Wigner-Ville Distribution)** — **v2's default.** Sharper time detail, implemented directly in this repo rather than via the usual `tftb` library — confirmed uninstallable alongside this project's stack (see [Assumptions](#assumptions-made-explicit)), not just assumed to be. v2's SPWVD follows the window recipe (Kaiser windows, 63/31 taps) of Khare, Bajaj & Acharya's PDCNNet paper (2021), the methodological reference for v2 — see `report.md` §7.

   *Multi-channel handling — v1:* all 19 channels are averaged into a single image. Avoids arbitrarily picking "the important" electrodes, but discards information about which side/region of the brain a pattern came from (e.g. the well-known "frontal asymmetry" depression marker).

   *Multi-channel handling — v2:* restricted to 4 frontal channels (Fp1, Fp2, F3, F4 — chosen explicitly to capture frontal asymmetry; see `report.md` §5), v2 generates **5 image variants per epoch** — each individual channel's own image, plus a combined (4-channel-averaged) variant — and compares all 5 against each other, directly answering whether averaging actually helps.

7. **CNN classifier.** The image is resized to 224x224 and fed into a convolutional neural network (CNN) pretrained on ImageNet. **v1** uses ResNet50 by default (VGG16 also available). **v2** compares **five** backbones head-to-head: ResNet50, ResNet18, VGG16, DenseNet121, MobileNetV2 (`src/model.py`). Reusing a network that already knows how to recognize visual patterns, and only teaching it the new task, is called **transfer learning**: the early layers (generic edges/textures) are frozen, a late block is fine-tuned on the EEG images, and the original 1000-category output layer is replaced with a small new layer that outputs one number: how "MDD-like" this image looks.

8. **Prediction.** That output is turned into a Normal/Depressed prediction per epoch.

---

## Why the train/test split is done "by subject", and why that matters

This is one of the most common mistakes in EEG machine-learning papers, so it's worth spelling out.

Because each 5-minute recording is chopped into many overlapping 4-second epochs, a single person contributes *dozens* of highly similar training examples. If you split those epochs randomly into train and test sets, epochs from the *same person* can end up on both sides of the split. The model can then partly "cheat" — instead of learning general patterns that distinguish depression from health, it can partly just learn to recognize *that specific person's* EEG signature, which it saw during training. Accuracy numbers from this kind of split look great but don't reflect how the model would perform on someone it has never seen before, which is the only thing that actually matters.

This project instead splits **by subject**: for every fold, every epoch from a given person goes entirely into either the training set or the test set, never both. This is called a **subject-independent split**, and it's enforced here with scikit-learn's `StratifiedGroupKFold` (grouped by subject ID) plus a runtime check in the code that raises an error if a subject ever ends up on both sides.

Because there are only 58 usable subjects (v1), a single train/test split would leave very little data — and the result would depend heavily on which few subjects happened to land in the test set. Instead, this project uses **subject-wise k-fold cross-validation**: **v1 uses 5 folds**, **v2 uses 10 folds** (both configurable). The data is divided into k subject groups, and the model is trained and tested k times, each time with a different group held out as the test set. The final result is reported as the **mean and standard deviation across all folds**, not a single number — a single fold's accuracy can look unusually good or bad just by chance, given how few subjects are involved.

---

## Repository layout

```
configs/default.yaml         - v1 config: EC only, 19ch averaged, CWT, ResNet50, 5-fold
configs/channel_study.yaml   - v2 config: Fp1/Fp2/F3/F4, SPWVD, 5 backbones, 10-fold
data/raw/                    - put the dataset's .edf files here (not committed to git)
data/processed/              - cached epochs + time-frequency images (auto-generated, not committed)
                                v1 caches flat (epochs/, tfd_metadata_cwt.csv);
                                v2 caches per condition (EC_study/, EO_study/, TASK_study/)
src/config.py                - loads a YAML config
src/preprocessing.py         - EDF loading, filtering, ICA, re-referencing, epoching
src/tfd_transform.py         - CWT + SPWVD image generation, multi-channel / single-channel variants
src/dataset.py               - PyTorch Dataset + subject-wise k-fold splitter
src/model.py                 - 5 transfer-learning backbones: ResNet50/18, VGG16, DenseNet121, MobileNetV2
src/train.py                  - per-fold training loop + checkpoint saving (src/train.save_checkpoint)
src/evaluate.py                - metrics, confusion matrices, cross-validation reporting
scripts/run_pipeline.py         - v1: runs one (condition, channel-set, model) config end-to-end
scripts/run_experiment_grid.py  - v2: runs the channel x condition x model comparison grid
scripts/predict.py               - run a saved checkpoint on a NEW .edf file (see below)
notebooks/demo_pipeline.ipynb    - visual walkthrough of preprocessing + image generation on 2 subjects
results/                          - v1 output: metrics.json, per-fold confusion matrices (auto-generated)
results_v2/                        - v2 output: pilot/ and full_grid/ subfolders, leaderboard.csv, figures/
report.md                           - full writeup of the v2 update: what changed, why, how to run it
research_paper.tex                   - conference-paper version of the same
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If a pinned version in `requirements.txt` doesn't resolve on your machine, install unpinned (`pip install mne PyWavelets numpy scipy scikit-learn scikit-image pandas matplotlib PyYAML torch torchvision tqdm`) and re-freeze with `pip freeze`.

Place the dataset's `.edf` files directly in `data/raw/` (filenames like `H S1 EC.edf`, `MDD S12 EC.edf`, matching how the original dataset names them).

## Running it

**v1 — the original, single-configuration pipeline** (already-reported results in `results/`):
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```
Runs, in order: preprocessing (cached to `data/processed/epochs/`), CWT image generation (cached to `data/processed/tfd_images/`), then subject-wise 5-fold training and evaluation on all 19 channels averaged into one image, ResNet50. Re-running reuses caches automatically; pass `--force-preprocess` / `--force-tfd` to regenerate. By default also saves every fold's model weights under `results/checkpoints/` (pass `--no-checkpoints` to skip). Output: `results/metrics.json`, `results/per_fold_metrics.csv`, confusion-matrix plots.

**v2 — the channel x condition x CNN-backbone comparison study**:
```bash
# one combination, fast — a good first check:
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --conditions EC --channel-variants combined --models mobilenet_v2

# full grid (ALL conditions x ALL channels x ALL 5 models, 10-fold — plan for
# hours on GPU, much longer on CPU-only):
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --conditions EC,EO,TASK --channel-variants Fp1,Fp2,F3,F4,combined \
    --models resnet50,resnet18,vgg16,densenet121,mobilenet_v2
```
See **`report.md` §10–11** for exactly what's been run so far (a real, reduced-scale pilot — `results_v2/pilot/`), full-grid time estimates, and practical tips for running it (start with the cheap backbones, run one condition at a time, use a GPU if you have one).

To just look at what the intermediate data looks like (without a full training run), open `notebooks/demo_pipeline.ipynb`.

## Testing a trained model on a new EEG recording

```bash
python scripts/predict.py --checkpoint results/checkpoints/best_model.pt --edf "path/to/new_recording.edf"
```
Loads the checkpoint (which stores everything needed to reproduce its exact training pipeline — channels, transform, filter/ICA/epoching settings), preprocesses the new `.edf` file the same way, generates the same image variant per epoch, and reports both a mean-probability and a majority-vote MDD/HC verdict across the recording's epochs. See `report.md` §12 for detail and caveats.

## Configuration

Everything tunable lives in the config YAML — window length, overlap, notch frequency, channels, transform (CWT/SPWVD), backbone, number of folds, training hyperparameters. Nothing is hardcoded in the source files; copy `configs/default.yaml` or `configs/channel_study.yaml` and pass `--config your_config.yaml` to experiment.

---

## Results

### v1 (EC, 19ch averaged, CWT, ResNet50, 5-fold) — complete

Mean accuracy **90.15% ± 5.16%** across 5 subject-independent folds (58 subjects, 8,293 epochs). Precision 90.74% ± 6.97%, recall 90.03% ± 10.64%, F1 89.80% ± 5.78%. Full per-fold numbers in `results/metrics.json` / `results/per_fold_metrics.csv`, confusion matrices in `results/confusion_matrix_*.png`.

### v2 (channel x condition x model comparison study) — pilot complete, full grid not yet run

A real, reduced-scale **pilot** has been run on the **EC condition only** (EO/TASK not yet run) to validate the new pipeline end-to-end, as two one-axis-at-a-time sweeps (9 combos, not the full 5×5 cross product): a **backbone sweep** (5 models, channel fixed to `combined`) and a **channel sweep** (5 channels, backbone fixed to MobileNetV2), both 3-fold, 2 training epochs, ≤7 epochs/subject:

| Backbone (combined channel) | Accuracy (mean ± std) | F1 |
|---|---|---|
| **VGG16** | **83.96% ± 0.94%** | 84.05% |
| ResNet18 | 71.93% ± 14.22% | 62.35% |
| MobileNetV2 | 60.65% ± 7.68% | 71.47% |
| ResNet50 | 58.65% ± 11.37% | 48.16% |
| DenseNet121 | 54.64% ± 5.70% | 56.79% |

| Channel (MobileNetV2) | Accuracy (mean ± std) | F1 |
|---|---|---|
| **Fp2** | **65.41% ± 5.92%** | 65.73% |
| F3 | 62.41% ± 5.86% | 68.34% |
| F4 | 61.65% ± 4.91% | 65.46% |
| Combined (Fp1+Fp2+F3+F4) | 60.65% ± 7.68% | 71.47% |
| Fp1 | 59.40% ± 2.21% | 58.82% |

VGG16 leads the backbone sweep (and with much the lowest variance); Fp2 leads the channel sweep. **A real reproducibility bug was found and fixed while producing these numbers** — the shared MobileNetV2×combined cell scored 59.90% the first time this was run and 65.66% the second time, purely from run-order-dependent RNG seeding (now fixed; re-verified to match exactly, 60.65%, regardless of order). The channel ranking above is a complete reversal of the pre-fix run. **Treat both tables as a plumbing/correctness validation, not a performance verdict** — 3 folds, 2 training epochs, a 399-epoch subsample, and (importantly) only a partial, non-cross-product slice of the full grid are each individually known to affect these numbers. Full story: `report.md` §10. Raw numbers and charts (now annotated with each bar's sample count `n`, since the two tables above aren't a fair 5×5 comparison): `results_v2/pilot/leaderboard.csv`, `results_v2/pilot/figures/`.

**The full 75-combination (3 conditions x 5 channels x 5 models), 10-fold grid has not been run** — it's a multi-hour-to-multi-day job on CPU (see `report.md` §10 for measured per-backbone speeds), implemented and ready to run via `scripts/run_experiment_grid.py` (see [Running it](#running-it)), ideally on a GPU machine. Once `results_v2/full_grid/leaderboard.csv` exists, its top row and the three `leaderboard_by_*.png` charts answer the project's three questions directly: best channel, best model, best condition.

---

## Assumptions made explicit

The spec for this project deliberately left a few implementation choices open, asking that they be documented rather than silently picked. Collecting them here (each is also noted at its point of use in the code):

1. **Mains notch frequency: 50 Hz.** The Mumtaz dataset was recorded in Malaysia, which uses 50 Hz mains electricity. Configurable (`preprocessing.notch_hz`) for other datasets.
2. **EOG/EMG artifact detection uses proxy channels, not dedicated sensors.** No true EOG/EMG channel exists in this dataset. Eye-blink ICA components are identified via correlation with the `Fp1`/`Fp2` forehead channels; muscle components via MNE's built-in high-frequency-power heuristic.
3. **v1's multi-channel time-frequency images are combined by averaging** all 19 channels, not by stacking channels or picking a subset. **v2 instead compares 5 variants** (4 individual channels + 1 combined) directly, rather than assuming averaging is the right call — see `report.md` §6.
4. **SPWVD is implemented directly in `src/tfd_transform.py`, not via the `tftb` package — confirmed, not just assumed, to be uninstallable here.** `tftb`'s only PyPI release pins `numpy==1.26.4` exactly, and attempting that install fails outright (`NumPy requires GCC >= 8.4`) on top of conflicting with this project's numpy≥2 stack — both verified by actually attempting the install. SPWVD is implemented from its standard mathematical definition, using a Kaiser window (v2; v1 used Hamming) matching the recipe in Khare, Bajaj & Acharya's PDCNNet paper (2021) — see `report.md` §7.
5. **"Replace final FC layer with binary classification head" is interpreted as replacing the entire original classifier**, not just its last layer — for all five backbones (v2) so they're handled identically. See `src/model.py` for detail.
6. **Peak-to-peak epoch rejection threshold (150 µV) is a simple heuristic**, not adaptive per subject — a reasonable default, not a tuned value.
7. **v2's channel set is Fp1, Fp2, F3, F4 — not the literally-requested "F1, F2"**, because F1/F2 don't exist in this dataset's 19-channel montage (verified by reading the raw channel labels). F3/F4 were chosen as the closest available electrodes, and because they're the standard electrodes for frontal alpha asymmetry, a literature-backed depression marker — see `report.md` §5 for the full reasoning, confirmed with the project team before implementation.
8. **Kaiser window shape parameter β = 5.0 (v2 SPWVD)** is not specified by the PDCNNet reference paper being followed; used as a documented, moderate default rather than a silent guess.

## Known limitations

- **Small dataset.** 58 usable EC subjects (30 MDD, 28 HC) is small for deep learning; results should be read as "how well this approach can do on this dataset", not as clinically validated performance. v2's 4-channel EC runs specifically have 57 usable subjects (see the ICA quirk above).
- **v1 is resting-state, eyes-closed only.** v2 adds eyes-open and the cognitive-task condition, but see the next point.
- **v2's full comparison grid has not been run to completion** (only a reduced-scale pilot has) — see [Results](#results) and `report.md` §10–11. Treat `results_v2/pilot/` as a correctness check, not a performance claim; `results_v2/full_grid/` doesn't exist yet.
- **v1's averaged multi-channel image discards spatial information** — a known EEG depression marker (frontal alpha asymmetry) can't be captured by a single averaged image. v2 addresses this directly by comparing single-channel images against the combined/averaged one, but only across 4 frontal channels, not all 19.
- **EOG/EMG artifact removal uses proxy channels**, not dedicated eye/muscle sensors (see Assumptions above) — likely slightly less clean than a dataset recorded with dedicated EOG electrodes. With only 4 channels (v2), this additionally causes ICA to fail outright on one subject (MDD_S23) rather than just being "slightly less clean."
- **CPU vs GPU.** Training is meaningfully slower on CPU, especially ResNet50/VGG16/ResNet18; if you have a CUDA GPU available, PyTorch will use it automatically. This is the main practical blocker to running v2's full grid quickly — see `report.md` §10–11 for measured CPU throughput per backbone.
- **No claim beyond what was actually run.** Whatever is in `results/` or `results_v2/` reflects the specific configs/combinations actually executed — other settings might perform differently, and that hasn't been exhaustively explored.
