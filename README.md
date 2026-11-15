# EEG-Based Depression Classification

This project evaluates machine-learning methods for distinguishing EEG recordings from participants with Major Depressive Disorder (MDD) and healthy controls (HC). It compares frontal-channel inputs, recording conditions, and convolutional neural network (CNN) backbones using time-frequency images and subject-independent cross-validation.

This is a research project, not a medical device or a clinical diagnostic tool.

## Pipeline

```text
EDF recordings
  -> channel selection and EEG preprocessing
  -> 4-second overlapping epochs
  -> SPWVD time-frequency images
  -> pretrained CNN classification
  -> subject-grouped 10-fold evaluation
```

### Dataset

The experiments use the Mumtaz MDD and healthy-control EEG dataset, recorded at 256 Hz with a 19-channel 10-20 montage. EC (eyes closed), EO (eyes open), and TASK recordings are evaluated separately. The dataset files are not included here; obtain them from the dataset provider and place the EDF files in `data/raw/`.

### Preprocessing and image generation

- Select Fp1, Fp2, F3, and F4 from each recording.
- Apply a 0.5-45 Hz band-pass filter, a 50 Hz notch filter, ICA-based artifact attenuation, and average reference.
- Segment recordings into 4-second epochs with 50% overlap; reject epochs exceeding the configured peak-to-peak threshold.
- Convert each channel to an SPWVD image using 64 frequency bins and 64 time bins. The SPWVD uses Kaiser windows of lengths 63 and 31 with beta=5.0.
- Evaluate five image variants: Fp1, Fp2, F3, F4, and the average of the four channel-wise SPWVD maps.

The data contain no dedicated EOG channel, so Fp1 and Fp2 are used as eye-artifact proxy channels during ICA. F1 and F2 are not present in the dataset montage; F3 and F4 are used instead.

### Classifiers and evaluation

The comparison uses ImageNet-pretrained ResNet50, ResNet18, VGG16, DenseNet121, and MobileNetV2 models. Each is adapted with a binary MDD/HC classification head.

Each condition/channel/model configuration is evaluated with stratified 10-fold cross-validation grouped by subject. All epochs from a subject stay together in each train/test split. A subject-wise validation split from the training portion is used for early stopping. Accuracy, precision, recall, and F1 are reported as mean and standard deviation across folds.

## Results

The repository includes the experiment outputs under `results/`. The full comparison contains 75 configurations: 3 recording conditions x 5 channel variants x 5 CNN backbones, with 10 folds per configuration.

| Condition | Highest-scoring channel/model | Accuracy (mean +/- std) |
|---|---|---:|
| EC | F3 / ResNet18 | 91.04% +/- 10.48% |
| EO | Combined / ResNet50 | 88.78% +/- 10.33% |
| TASK | Fp1 / ResNet18 | 88.05% +/- 8.19% |

These are the highest-scoring configurations within the saved cross-validation comparison. Per-configuration metrics and fold-level results are available under `results/full_grid/`. The `results/pilot/` directory contains the project's pilot run artifacts.

Paper figures are stored in `results/figures/` and `results/full_grid/EC/f3/resnet18/`. See [`research_paper.tex`](research_paper.tex) for the IEEE-style paper and its detailed result tables.

## Setup

Python 3.10 or later is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the dataset EDF files in `data/raw/` using names that identify the group, subject, and condition, for example:

```text
H S1 EC.edf
MDD S12 EO.edf
```

Raw EEG data, processed caches, model weights, and local secrets are excluded by `.gitignore`. Experiment results and paper figures are not ignored.

## Run the comparison

Run the complete condition x channel x model grid:

```powershell
python scripts/run_experiment_grid.py --config configs/channel_study.yaml `
  --conditions EC,EO,TASK `
  --channel-variants Fp1,Fp2,F3,F4,combined `
  --models resnet50,resnet18,vgg16,densenet121,mobilenet_v2
```

The configuration sets the output root to `results_v2/`; the script writes the grid to `results_v2/full_grid/`. Use `--conditions`, `--channel-variants`, and `--models` to run a selected subset. For a quick end-to-end check, add `--pilot`.

Processed epochs and time-frequency images are cached under `data/processed/`. Use `--force-preprocess` or `--force-tfd` to rebuild the corresponding cache.

## Predict from a trained checkpoint

To save fold checkpoints during training, add `--save-checkpoints` to the grid command. Then run:

```powershell
python scripts/predict.py --checkpoint path\to\fold0.pt --edf path\to\recording.edf
```

The checkpoint stores the model and preprocessing settings used during training. Inference reports epoch-level outputs and a recording-level summary.

## Repository layout

```text
configs/channel_study.yaml      Comparison-study settings
data/raw/                       User-provided EDF recordings (not committed)
data/processed/                 Generated EEG and image caches (not committed)
src/preprocessing.py            EDF loading, filtering, ICA, epoching
src/tfd_transform.py            CWT and SPWVD image generation
src/model.py                    CNN backbones
src/train.py                    Fold training and checkpoint saving
src/evaluate.py                 Metrics and confusion matrices
scripts/run_experiment_grid.py  Channel/condition/model comparison
scripts/predict.py              Inference from a saved checkpoint
results/                        Saved experiment outputs and paper figures
results_v2/                     Output directory configured for new runs
research_paper.tex              IEEE-style research paper
```
