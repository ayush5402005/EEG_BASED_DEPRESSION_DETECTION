# EEG-Based Depression Detection — Project Report

**Group 24** — Aatha Pant (2311401208), Ayush Kumar (2311401231), Adhya Singh Chandel (2311401272), Shweta Pal (2311401277)
**Supervisor** — Dr. Varun Bajaj
Department of Electronics & Communication Engineering, MANIT Bhopal

This report documents the **v2 (channel / condition / model comparison) pipeline**, which extends the project's original single-configuration pipeline (EC-only, CWT, ResNet50, 5-fold — still intact and reproducible via `configs/default.yaml`). See [research_paper.tex](research_paper.tex) for the condensed, conference-style writeup; this document goes into more implementation detail and is the place to look for *why* each decision was made.

---

## 1. Introduction

Major Depressive Disorder (MDD) affects an estimated 280 million people worldwide and is currently diagnosed through subjective clinical interviews and questionnaires. EEG offers an objective, non-invasive signal that correlates with MDD in the neuroscience literature (notably, frontal alpha asymmetry). This project builds a pipeline that converts raw EEG recordings into time-frequency images and classifies them with transfer-learned CNNs — and, in this v2 iteration, systematically compares *which channel*, *which CNN*, and *which recording condition* work best, rather than assuming one configuration.

## 2. Problem Statement

Clinical depression diagnosis has no objective, EEG-based marker in routine use. A prior version of this pipeline showed that an EEG-image + CNN approach can distinguish MDD from healthy controls with ~90% subject-independent accuracy on one specific configuration (eyes-closed EEG, all 19 channels averaged into one image, ResNet50). That leaves open whether that specific configuration was actually the best choice, or simply a reasonable starting point.

## 3. Research Gap

1. Prior EEG-depression studies (including this project's own first version) typically commit to **one** channel-handling strategy, **one** CNN architecture, and **one** recording condition, without reporting how much any of those three choices actually matters.
2. Multi-channel EEG is often collapsed into a single averaged image, which is simple and avoids arbitrary channel selection, but is documented (in this project's own README) to discard inter-channel information — specifically frontal alpha asymmetry, the most literature-backed EEG depression biomarker.
3. The Mumtaz dataset used here contains three recording conditions (eyes-closed, eyes-open, task), but almost all prior work on it (including this project's v1) uses only one.

## 4. Objectives

1. Generate SPWVD time-frequency images (replacing CWT) for individual EEG channels (Fp1, Fp2, F3, F4) and a combined/averaged variant, from the same preprocessed recording.
2. Compare five ImageNet-pretrained CNN backbones (ResNet50, ResNet18, VGG16, DenseNet121, MobileNetV2) under an identical evaluation protocol.
3. Compare all three recording conditions (EC, EO, TASK) under the same protocol.
4. Answer, with real numbers: **best channel**, **best CNN model**, and **best condition** — and the combination that works best overall.
5. Keep evaluation leakage-free: strict subject-independent, stratified, subject-wise 10-fold cross-validation throughout.

## 5. Why F3/F4, Not F1/F2

The project brief named Fp1, Fp2, plus "either F1, F2 or O1, O2" as the channel set to study. **F1 and F2 do not exist in this dataset.** The Mumtaz dataset's montage is the basic 19-channel 10-20 system:

```
Fp1, F3, C3, P3, O1, F7, T3, T5, Fz, Fp2, F4, C4, P4, O2, F8, T4, T6, Cz, Pz
```

F1/F2 belong to the denser 10-10 extended system and were never recorded. Verified directly by reading the channel labels out of a raw `.edf` file rather than assumed from the dataset's documentation.

Decision: use **F3, F4** — the closest electrodes this dataset actually has to "F1/F2," and, more importantly, the *standard* electrodes used in the EEG literature to measure **frontal alpha asymmetry**, which is specifically the kind of inter-hemispheric signal the original pipeline's "average all 19 channels into one image" step was already documented to discard. O1/O2 (the literal fallback named in the brief) are occipital electrodes tied to visual-cortex alpha rhythm, not a depression-specific marker — available, but a weaker scientific fit. This tradeoff was surfaced to, and confirmed with, the project team before implementation.

Final channel set: **Fp1, Fp2, F3, F4** — all four frontal, consistent with Fp1/Fp2's existing role as the EOG-artifact-proxy channels.

## 6. Channel Variants ("Best Channel Out of These 5")

From the 4 preprocessed channels, **5 image variants** are generated per epoch, resolving the brief's "best channel out of these 5":

| Variant | Channels averaged | What it answers |
|---|---|---|
| `fp1` | Fp1 only | Does this single channel alone carry a usable signal? |
| `fp2` | Fp2 only | ″ |
| `f3`  | F3 only | ″ |
| `f4`  | F4 only | ″ |
| `combined` | Fp1+Fp2+F3+F4 averaged | Does averaging help or hurt, vs. the best single channel? |

All 5 are generated from the *same* preprocessed 4-channel epoch (no extra preprocessing per variant), so the comparison is fair and the extra compute cost is just image generation (milliseconds per epoch), not re-running ICA five times.

## 7. SPWVD: Why a Library Wasn't Used (Verified, Not Assumed)

The original pipeline already avoided the standard `tftb` SPWVD package because its PyPI metadata pins `numpy<2.0`, which conflicts with this project's PyTorch/numpy≥2 stack. For this update, that claim was **actually tested**, not just re-stated:

```
$ pip install tftb==0.2.0 --dry-run
...
Project version: 1.26.4                    # tftb pulls numpy==1.26.4 EXACTLY
...
ERROR: NumPy requires GCC >= 8.4            # and that numpy version won't even build here
```

So `tftb` is confirmed uninstallable in this environment, for two independent reasons. SPWVD continues to be implemented directly in `src/tfd_transform.py`, from its standard discrete definition (Boashash), now matching the window choice of the directly relevant reference paper below.

### Reference paper followed for this update

**Khare, S. K., Bajaj, V., & Acharya, U. R. (2021). "PDCNNet: An Automatic Framework for the Detection of Parkinson's Disease Using EEG Signals." *IEEE Sensors Journal*, 21(15), 17017–17024.**

This paper (co-authored by this project's own supervisor, Dr. Varun Bajaj) couples SPWVD time-frequency imaging with a CNN for EEG-based Parkinson's detection, reporting 99.97% accuracy under 10-fold cross-validation on one of its two datasets. This project's SPWVD implementation now matches its specific recipe:

- **Window type**: Kaiser (previously Hamming in this project's v1 SPWVD implementation) for both the time- and lag-smoothing windows.
- **Window lengths**: 63 (frequency/lag) and 31 (time), following the `2^m − 1` rule for fast evaluation — this project's defaults already matched these exact values before this update, which is a reassuring consistency check.
- **Kaiser β**: not specified in the reference; `β = 5.0` is used here as a documented, moderate default, not a silent guess.
- **10-fold cross-validation**: adopted directly from the reference (this project's v1 used 5-fold).

What was **not** copied: PDCNNet's own compact, from-scratch, 4-conv-layer CNN and its 227×227 input size (an AlexNet-era convention). This project instead compares 5 *pretrained*, transfer-learned backbones at 224×224 (the input size torchvision's pretrained weights expect), since depression-EEG epoch counts per class here are smaller than PDCNNet's two Parkinson's datasets, and transfer learning is expected to need less task-specific data.

## 8. The Five CNN Backbones

| Backbone | Total params | Trainable | Unfrozen block |
|---|---|---|---|
| ResNet50 | 23.8M | 15.2M | `layer4` |
| ResNet18 | 11.2M | 8.5M | `layer4` |
| VGG16 | 17.9M | 10.3M | conv block 5 (fixed, index-based) |
| DenseNet121 | 7.1M | 2.3M | `denseblock4` |
| MobileNetV2 | 2.4M | 1.8M | last 5 inverted-residual blocks (fixed, index-based) |

Every backbone follows the same recipe as the original pipeline's ResNet50/VGG16: keep the pretrained convolutional features, freeze everything except a late slice, replace the entire original classifier (not just its last layer) with a small binary head trained from scratch. See `src/model.py` for the full implementation, including why VGG16 and MobileNetV2 can't use a named `freeze_until` the way ResNet/DenseNet can (they expose their features as flat, unnamed sequences).

Measured CPU training speed (this project's CPU-only development machine, batch size 16–32, forward+backward):

| Backbone | Images/sec (CPU) |
|---|---|
| MobileNetV2 | ~50 |
| DenseNet121 | ~18 |
| ResNet18 | ~6 |
| ResNet50 / VGG16 | not separately benchmarked; expected slower than ResNet18 given parameter count |

This is why the pilot run (Section 10) uses MobileNetV2.

## 9. Evaluation Protocol

Unchanged in spirit from the original pipeline, upgraded from 5-fold to **10-fold**: `StratifiedGroupKFold` grouped by `subject_id`, so every epoch from a given subject lands entirely on one side of any fold's split, with a runtime assertion that no subject ever appears in both. Within each fold's training subjects, a further subject-wise split carves out an early-stopping validation set; the fold's held-out test subjects are never used for model selection. Metrics (accuracy, precision, recall, F1, confusion matrix) are logged per fold and reported as mean ± std across all folds, never a single cherry-picked fold.

## 10. What Was Actually Run (Pilot — Real Numbers, Reduced Scale)

The full grid is **3 conditions × 5 channel variants × 5 backbones = 75 combinations × 10 folds = 750 individual model trainings**. Measured on this project's CPU-only development machine, that is a multi-hour-to-multi-day job — not something to run inside an interactive session. Rather than claim full-grid numbers that were never produced, a **real, reduced-scale pilot** was run to validate the entire new pipeline end-to-end:

- Condition: EC only
- Folds: 3 (not 10)
- Training epochs per fold: 2 (not 20)
- Epoch subsample: up to 7 epochs/subject, seeded random sample (399 of 8,379 available EC epochs)
- Two one-axis-at-a-time sweeps, not the full 5×5 cross product (25 combos): a **channel sweep** (all 5 channel variants, backbone fixed to MobileNetV2 — the fastest) and a **backbone sweep** (all 5 backbones, channel fixed to `combined`) — 9 combinations total, sharing the MobileNetV2 × combined cell.

Commands used:
```bash
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --pilot --conditions EC --channel-variants Fp1,Fp2,F3,F4,combined \
    --models mobilenet_v2 --pilot-max-epochs-per-subject 7 \
    --pilot-folds 3 --pilot-training-epochs 2

python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --pilot --conditions EC --channel-variants combined \
    --models resnet50,resnet18,vgg16,densenet121,mobilenet_v2 \
    --pilot-max-epochs-per-subject 7 --pilot-folds 3 --pilot-training-epochs 2
```

**Result — backbone sweep (real numbers, EC, combined channel, 3-fold):**

| Backbone | Accuracy (mean ± std) | F1 |
|---|---|---|
| **VGG16** | **83.96% ± 0.94%** | 84.05% |
| ResNet18 | 71.93% ± 14.22% | 62.35% |
| MobileNetV2 | 60.65% ± 7.68% | 71.47% |
| ResNet50 | 58.65% ± 11.37% | 48.16% |
| DenseNet121 | 54.64% ± 5.70% | 56.79% |

**Result — channel sweep (real numbers, EC, MobileNetV2, 3-fold):**

| Channel variant | Accuracy (mean ± std) | F1 |
|---|---|---|
| **Fp2** | **65.41% ± 5.92%** | 65.73% |
| F3 | 62.41% ± 5.86% | 68.34% |
| F4 | 61.65% ± 4.91% | 65.46% |
| Combined (Fp1+Fp2+F3+F4) | 60.65% ± 7.68% | 71.47% |
| Fp1 | 59.40% ± 2.21% | 58.82% |

**VGG16 comes out clearly on top** of the backbone sweep — and, notably, with by far the lowest variance of the five, suggesting it's not just luckier but more consistently fitting in 2 epochs than the others (plausibly because its larger, more expressive frozen-feature representation needs less fine-tuning to become useful — see the paper's Discussion for the full reasoning). ResNet50, the *original* pipeline's own default backbone, comes in 4th here under this reduced 2-epoch budget — a reminder that a backbone's full-training performance (ResNet50 reached 90%+ accuracy in the original, 20-epoch, 5-fold v1 run) doesn't necessarily predict its performance under a much smaller training budget.

**Fp2 comes out on top** of the channel sweep. **Important:** this directly *contradicts* an earlier, buggy run of the same sweep, which had ranked Fp1 first and Fp2 last. Here's why, and it's worth reading in full before trusting either ranking too much:

> **A real bug was found and fixed while producing these numbers.** The first time this channel sweep was run, `mobilenet_v2`/`combined` scored 59.90%. Run again (as part of a separate backbone-sweep invocation, otherwise identically configured), it scored 65.66%. The cause: `src/train.py`'s model initialization draws from PyTorch's *global* random-number stream, and `scripts/run_experiment_grid.py` was only seeding it once, at the very start of the whole script — not before each individual combination. So a combination's result silently depended on *which other combinations happened to run before it* in the same process, not just on its own configuration. This is now fixed (the RNG is reseeded immediately before every fold), and verified: re-running both sweeps now gives the *exact same* value (60.65% ± 7.68%, matched to 4 decimal places) for that shared cell regardless of which sweep it ran as part of. **All numbers in this report are from the fixed version** — but the fact that fixing a seeding bug flipped the entire channel ranking is itself the clearest evidence in this whole pilot that 3-fold/2-epoch/399-epoch numbers are not stable enough to call a "best channel" yet.

A second, related code fix: the leaderboard used to be fully overwritten by each new invocation of the grid script, rather than merged with what was already there — fixed so repeated/partial runs accumulate into one leaderboard instead of clobbering each other.

**Take both tables as a plumbing validation, not a verdict** — see the caveats directly below, and note specifically that the two tables above are *not* a fair 5×5 comparison: VGG16 was only ever tested on the `combined` channel, and the channel ranking was only ever tested with MobileNetV2. `scripts/run_experiment_grid.py`'s chart generation now detects and flags this automatically (prints a warning, and annotates each bar in `leaderboard_by_*.png` with its sample count `n`) whenever a comparison mixes groups with unequal coverage, specifically so a sparse, partial grid like this one's charts can't be misread as a fair comparison.

This also surfaced a real finding: reducing the channel count from 19 to 4 causes ICA to fail on one EC subject (**MDD_S23**) — with only 4 channels, variance concentrates so heavily in 1 principal component that `ica_n_components=0.99` can't resolve more than one, and MNE's `find_bads_eog`/`find_bads_muscle` need more than that. The subject is automatically skipped (as the pipeline already does for any file that fails to process), leaving **57 of 58 EC subjects** usable for the 4-channel study, one fewer than the original 19-channel pipeline's cohort. This is reported, not hidden.

**What the pilot does and doesn't tell you:** it proves the entire pipeline — preprocessing, 5-variant SPWVD generation, 5-backbone architecture, checkpoint saving, leaderboard aggregation — actually runs correctly end-to-end and produces genuine numbers. It is *not* a reliable estimate of the full study's eventual accuracy: 3 folds instead of 10, 2 training epochs instead of 20, and a 7-epochs/subject cap each independently reduce achievable accuracy and increase variance relative to the full configuration.

## 11. How to Train the Model

### 11.1 The original single-configuration pipeline (unchanged, already-reported results)

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```
Runs EC-only, CWT, ResNet50, 5-fold, on all 19 channels averaged into one image — exactly the configuration `results/metrics.json` and the original sections of the paper report. By default this now also saves every fold's model weights to `results/checkpoints/foldN.pt` and the best-accuracy fold to `results/checkpoints/best_model.pt` (pass `--no-checkpoints` to skip this).

### 11.2 The new comparison grid

```bash
# One combination (fast, a good first check that everything works on your machine):
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --conditions EC --channel-variants combined --models mobilenet_v2

# A real but bounded pilot (minutes, not hours — same as Section 10 above):
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --pilot --conditions EC --channel-variants Fp1,Fp2,F3,F4,combined \
    --models mobilenet_v2 --pilot-max-epochs-per-subject 7 \
    --pilot-folds 3 --pilot-training-epochs 2

# The FULL grid (all 3 conditions x 5 channels x 5 models x 10-fold — plan for
# hours on GPU, much longer on CPU; run overnight or on a cloud GPU instance):
python scripts/run_experiment_grid.py --config configs/channel_study.yaml \
    --conditions EC,EO,TASK \
    --channel-variants Fp1,Fp2,F3,F4,combined \
    --models resnet50,resnet18,vgg16,densenet121,mobilenet_v2 \
    --save-checkpoints
```

**Practical tips for running the full grid:**
- **Use a GPU if at all possible.** PyTorch uses CUDA automatically if `torch.cuda.is_available()`; on a modern GPU, expect roughly 10–50x the CPU throughput measured in Section 8, making the full grid realistically an overnight run rather than a multi-day one.
- **Run conditions one at a time** (`--conditions EC`, then `--conditions EO`, then `--conditions TASK`) if you want intermediate results sooner, or want to run them on different machines in parallel — each condition's preprocessing cache (`data/processed/<condition>_study/`) is independent.
- **Preprocessing and image generation are cached** — re-running the same `--conditions`/`--channel-variants` combination skips straight to training. Pass `--force-preprocess` / `--force-tfd` only if you change `configs/channel_study.yaml`'s `eeg`, `preprocessing`, `epoching`, or `tfd` sections.
- **Start with the cheap backbones** (`mobilenet_v2`, `densenet121`) to get a channel/condition leaderboard quickly, then add `resnet18`, `resnet50`, `vgg16` for the full comparison.
- **Leaderboard**: `results_v2/full_grid/leaderboard.csv`, plus `results_v2/full_grid/figures/leaderboard_by_{channel,model,condition}.png`, and a per-combination subfolder `results_v2/full_grid/<condition>/<channel_variant>/<model>/` with that combination's own `metrics.json`, per-fold confusion matrices, and (if `--save-checkpoints` was passed) model weights.

### 11.3 Running this on a different laptop

This project is self-contained aside from the dataset itself:
1. Copy the whole `EEG_Depression_Project/` folder (or clone/transfer it) to the new machine, including the `.edf` files under `data/raw/` (not committed to git — see README's Setup section).
2. `python -m venv .venv && .venv\Scripts\activate` (Windows) and `pip install -r requirements.txt`. If a dependency version in `requirements.txt` doesn't resolve on that machine, install unpinned (`pip install mne PyWavelets numpy scipy scikit-learn scikit-image pandas matplotlib PyYAML torch torchvision tqdm`) and update `requirements.txt` with what actually got installed via `pip freeze`.
3. `data/processed/` (the preprocessing/image caches) does **not** need to be copied — it will be regenerated on first run — but copying it over does save re-running preprocessing/ICA if you already have it.
4. Run either `scripts/run_pipeline.py` (original) or `scripts/run_experiment_grid.py` (new) as above.

## 12. How to Test the Model on a Different/New EEG (.edf) File

A new script, **`scripts/predict.py`**, runs a saved checkpoint on any `.edf` file not used for training:

```bash
python scripts/predict.py --checkpoint results/checkpoints/best_model.pt --edf "data/raw/H S2 EC.edf"

# or a checkpoint from the comparison grid:
python scripts/predict.py \
    --checkpoint results_v2/full_grid/EC/fp1/resnet18/checkpoints/best_model.pt \
    --edf path/to/some_new_recording.edf
```

What it does:
1. Loads the checkpoint, which stores not just the trained weights but **everything needed to reproduce the exact pipeline it was trained with** — which channels, which TF transform, image size, filter/notch/ICA settings, epoch window/overlap, and which channel variant (single channel or combined) its images were built from. This means `predict.py` never has to guess a config; it reconstructs the training pipeline exactly from the checkpoint file itself.
2. Runs that same preprocessing (filter, ICA, re-reference) and epoching on the new `.edf` file.
3. Generates the same SPWVD (or CWT) image variant for every resulting epoch.
4. Runs the model on every epoch, producing a per-epoch MDD probability.
5. Reports both:
   - **Mean-probability verdict**: average the per-epoch MDD probability across the whole recording, threshold at 0.5.
   - **Majority-vote verdict**: the fraction of epochs individually predicted MDD vs. HC.
6. Prints a reminder that this is a research pipeline's output, not a clinical diagnosis — same framing as the rest of this project.

**Important caveat**: the new `.edf` file should be from the same kind of recording the checkpoint was trained on (same channel montage containing at least the channels the checkpoint needs, same approximate recording condition) — the pipeline does not currently detect or warn about a condition mismatch (e.g. testing an EC-trained model on an EO recording), beyond whatever channel-name lookup failure would occur if the montage genuinely differs.

## 12.5 Bugs Found and Fixed While Running the Pilots (Summary)

Two real issues surfaced from actually running `scripts/run_experiment_grid.py` more than once — not from code review alone. Both are explained in full, with the exact before/after numbers, in §10 above; summarized here for reference:

1. **Leaderboard overwrite** — a second invocation used to replace `leaderboard.csv` entirely instead of merging with it. Fixed.
2. **Run-order-dependent results** — `set_seed()` ran once per script invocation, not per fold, so model init/shuffling drew from a shared global RNG stream and a combo's result depended on what ran before it. Fixed by reseeding before every fold; verified reproducible by re-running both sweeps and matching the shared cell exactly.

## 13. Known Limitations (This Update)

- **Full grid not executed.** `results_v2/full_grid/` does not exist yet — only `results_v2/pilot/` does. The paper and this report are explicit about this; no full-grid number is presented as fact.
- **Pilot is small.** 3-fold (not 10), 2 training epochs (not 20), 7-epochs/subject cap, one backbone, one condition. Treat pilot numbers as a correctness check, not a performance claim.
- **4-channel ICA is less robust than 19-channel ICA** — demonstrated directly by MDD_S23 failing preprocessing with the 4-channel montage (it processes fine with the original 19-channel `configs/default.yaml`).
- **CPU-only development environment.** No CUDA GPU was available while building this; all timing estimates in this report are CPU numbers and should be read as upper bounds on full-grid runtime, not lower bounds.
- **Checkpoints are large and numerous if `--save-checkpoints` is used on the full grid** (750 individual fold models) — consider enabling it only for the specific combinations you intend to actually deploy/inspect afterward, not the whole grid.
- Limitations already documented in the original pipeline (small overall dataset, EOG/EMG proxy-channel artifact detection, resting-state assumptions) still apply equally to this update — see `README.md`'s own Limitations section.

## 14. References

See `research_paper.tex`'s bibliography for full citations, including the new primary reference for this update:

> S. K. Khare, V. Bajaj, and U. R. Acharya, "PDCNNet: An Automatic Framework for the Detection of Parkinson's Disease Using EEG Signals," *IEEE Sensors Journal*, vol. 21, no. 15, pp. 17017–17024, Aug. 2021.
