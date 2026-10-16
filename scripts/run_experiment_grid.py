#!/usr/bin/env python
"""Runs the channel x recording-condition x CNN-backbone comparison study and
answers, with numbers: which single channel works best, which of the 5 CNN
architectures works best, and which recording condition (EC/EO/TASK) works
best -- and with which combination.

For each (condition, channel-variant) pair, preprocessing + SPWVD image
generation is run ONCE and cached; every one of the 5 CNN backbones is then
trained and evaluated with subject-wise k-fold CV on that same cached image
set, so the grid is (conditions x channel-variants) preprocessing runs, times
(models) training runs -- not a full re-preprocessing per model.

Channel variants, from the 4 channels in the config (Fp1, Fp2, F3, F4 by
default): one image variant per individual channel (no averaging -- a
single-channel "average" is just that channel), plus one "combined" variant
(all 4 channels averaged, the same strategy the original pipeline used for
all 19 channels). That gives 5 channel variants to compare, matching the
project brief's "best channel out of these 5".

This is a LOT of training runs (conditions x variants x models x k-folds).
On a CPU-only machine, running the full grid to completion is a multi-hour-
to-multi-day job, not something to kick off casually inside a chat session --
see report.md's "How to train the model" section for realistic time
estimates and a recommendation to use a GPU (local or Colab) for the full
grid. Use --pilot for a fast, real (not fabricated) correctness check that
exercises the entire pipeline end-to-end on a small subsample.

Usage (full grid, EC condition only, two backbones, as an example):
    python scripts/run_experiment_grid.py --config configs/channel_study.yaml \\
        --conditions EC --models resnet18,mobilenet_v2

Usage (fast pilot, everything wired up, small numbers, clearly tagged):
    python scripts/run_experiment_grid.py --config configs/channel_study.yaml \\
        --pilot --conditions EC --models resnet18,mobilenet_v2 --channel-variants Fp1,combined

Usage (the real, full study -- expect this to take a long time):
    python scripts/run_experiment_grid.py --config configs/channel_study.yaml \\
        --conditions EC,EO,TASK --models resnet50,resnet18,vgg16,densenet121,mobilenet_v2 \\
        --channel-variants Fp1,Fp2,F3,F4,combined
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.dataset import make_subject_kfold_splits
from src.evaluate import aggregate_and_report, evaluate_fold
from src.model import SUPPORTED_BACKBONES
from src.preprocessing import build_epoch_dataset
from src.tfd_transform import build_tfd_dataset
from src.train import get_device, save_checkpoint, set_seed, train_one_fold


def _subsample_for_pilot(epoch_metadata: pd.DataFrame, max_per_subject: int, seed: int) -> pd.DataFrame:
    """Keeps at most `max_per_subject` epochs per subject (random, seeded), preserving every
    subject (so the subject-wise k-fold split still has every subject's group to work with).
    This is what keeps --pilot fast: it shrinks the dataset, not the pipeline's correctness.

    Written as an explicit per-group loop + concat rather than groupby().apply(), which on
    this project's pandas version silently drops the grouping column from the result.
    """
    parts = [
        group.sample(n=min(len(group), max_per_subject), random_state=seed)
        for _, group in epoch_metadata.groupby("subject_id")
    ]
    return pd.concat(parts, ignore_index=True)


def _channel_variants(cfg, requested: list[str]) -> list[tuple[str, list[int] | None]]:
    """Returns [(variant_tag, channel_indices), ...] for the requested variant names.
    A named channel (e.g. "Fp1") -> that single channel's own index, no averaging.
    "combined" -> None (average every channel in cfg.eeg.channels, as in the original pipeline).
    """
    channels = list(cfg.eeg.channels)
    variants = []
    for name in requested:
        if name.lower() == "combined":
            variants.append(("combined", None))
        else:
            if name not in channels:
                raise ValueError(f"Channel {name!r} is not in cfg.eeg.channels {channels}")
            variants.append((name.lower(), [channels.index(name)]))
    return variants


def run_combo(base_cfg, condition: str, variant_tag: str, channel_indices, model_name: str,
              epoch_metadata: pd.DataFrame, k_folds: int, epochs: int, device, results_root: Path,
              save_checkpoints: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.data.condition = condition
    cfg.data.cache_subdir = f"{condition}_study"
    cfg.model.backbone = model_name
    cfg.training.k_folds = k_folds
    cfg.training.epochs = epochs
    combo_dir = results_root / condition / variant_tag / model_name
    cfg.output.results_dir = str(combo_dir)

    tfd_metadata = build_tfd_dataset(
        cfg, epoch_metadata, channel_indices=channel_indices, variant_tag=variant_tag
    )

    fold_metrics = []
    for fold_idx, (train_meta, test_meta) in enumerate(
        make_subject_kfold_splits(tfd_metadata, k_folds, cfg.training.seed)
    ):
        # Re-seed before every fold, not just once in main(): otherwise model init /
        # DataLoader shuffling draw from torch's global RNG stream, so a combo's result
        # silently depends on which other combos happened to run before it in the same
        # process -- not reproducible from the config alone. Confirmed this was happening
        # (mobilenet_v2/combined scored 59.9% solo vs 65.7% when run after 4 other models).
        set_seed(cfg.training.seed + fold_idx)
        model = train_one_fold(cfg, train_meta, fold_idx, device)
        fold_metrics.append(evaluate_fold(model, test_meta, cfg, device))
        if save_checkpoints:
            save_checkpoint(model, cfg, combo_dir / "checkpoints" / f"fold{fold_idx}.pt",
                             extra={"fold_idx": fold_idx, "condition": condition, "channel_variant": variant_tag})

    summary = aggregate_and_report(fold_metrics, cfg)
    row = {
        "condition": condition,
        "channel_variant": variant_tag,
        "model": model_name,
        "n_folds": k_folds,
        "n_epochs_trained": epochs,
        "n_test_epochs_total": sum(m["n_test_epochs"] for m in fold_metrics),
    }
    row.update(summary["summary_mean_std"])
    return row


def _leaderboard_chart(df: pd.DataFrame, group_col: str, out_path: Path, title: str) -> None:
    """Bar chart of mean accuracy per group. Also shows each bar's n (how many
    (condition, channel, model) combinations fed it) -- on anything less than the
    full grid, groups can have different, non-comparable n (e.g. one channel run
    across all 5 models, the rest run with only 1), which would otherwise silently
    make the chart compare unlike things. A WARNING is printed when n is uneven.
    """
    grouped = df.groupby(group_col)["accuracy_mean"]
    agg = grouped.mean().sort_values(ascending=False)
    counts = grouped.count()
    if counts.nunique() > 1:
        print(f"[grid] WARNING: '{group_col}' comparison is from an uneven number of combos per "
              f"group (not a full grid yet) -- counts: {counts.to_dict()}. Treat this chart as "
              f"indicative, not a fair like-for-like comparison, until every group has equal n.")

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(agg.index.astype(str), agg.values, color="#2F6F6B")
    ax.set_ylabel("Mean accuracy")
    uneven_note = " (uneven n per bar -- see labels)" if counts.nunique() > 1 else ""
    ax.set_title(title + uneven_note, fontsize=10)
    ax.set_ylim(0, 1.0)
    for i, group_key in enumerate(agg.index):
        ax.text(i, agg.values[i] + 0.015, f"{agg.values[i]:.1%}", ha="center", fontsize=9)
        ax.text(i, 0.02, f"n={counts[group_key]}", ha="center", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Channel x condition x CNN-backbone comparison study.")
    parser.add_argument("--config", default="configs/channel_study.yaml")
    parser.add_argument("--conditions", default="EC,EO,TASK")
    parser.add_argument("--channel-variants", default="Fp1,Fp2,F3,F4,combined")
    parser.add_argument("--models", default=",".join(SUPPORTED_BACKBONES))
    parser.add_argument("--k-folds", type=int, default=None, help="Overrides training.k_folds from the config.")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides training.epochs from the config.")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-tfd", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true",
                         help="Save every fold's model weights. Off by default for the full grid "
                              "(disk use scales with conditions x variants x models x k_folds).")
    parser.add_argument("--pilot", action="store_true",
                         help="Fast, real, end-to-end correctness check: subsamples epochs per "
                              "subject and reduces folds/training epochs. Results are genuine "
                              "numbers from an actual (small) run, not placeholders -- but are "
                              "NOT a substitute for the full grid, and results are tagged 'pilot'.")
    parser.add_argument("--pilot-max-epochs-per-subject", type=int, default=8)
    parser.add_argument("--pilot-folds", type=int, default=3)
    parser.add_argument("--pilot-training-epochs", type=int, default=2)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    set_seed(base_cfg.training.seed)
    device = get_device()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unknown model {m!r}, expected one of {SUPPORTED_BACKBONES}")
    requested_variants = [v.strip() for v in args.channel_variants.split(",") if v.strip()]

    k_folds = args.pilot_folds if args.pilot else (args.k_folds or base_cfg.training.k_folds)
    epochs = args.pilot_training_epochs if args.pilot else (args.epochs or base_cfg.training.epochs)

    results_root = Path(base_cfg.output.results_dir) / ("pilot" if args.pilot else "full_grid")
    results_root.mkdir(parents=True, exist_ok=True)

    print(f"[grid] {'PILOT' if args.pilot else 'FULL'} run: conditions={conditions} "
          f"channel_variants={requested_variants} models={models} k_folds={k_folds} "
          f"training_epochs={epochs} device={device}")
    print(f"[grid] total combinations: {len(conditions) * len(requested_variants) * len(models)}")

    rows = []
    t_start = time.time()
    for condition in conditions:
        cond_cfg = copy.deepcopy(base_cfg)
        cond_cfg.data.condition = condition
        cond_cfg.data.cache_subdir = f"{condition}_study"
        print(f"\n[grid] === Condition {condition}: preprocessing (cached after first run) ===")
        epoch_metadata = build_epoch_dataset(cond_cfg, force=args.force_preprocess)
        print(f"[grid] {condition}: {len(epoch_metadata)} epochs, "
              f"{epoch_metadata['subject_id'].nunique()} subjects")

        if args.pilot:
            epoch_metadata = _subsample_for_pilot(
                epoch_metadata, args.pilot_max_epochs_per_subject, base_cfg.training.seed
            )
            print(f"[grid] pilot subsample: {len(epoch_metadata)} epochs "
                  f"(<= {args.pilot_max_epochs_per_subject}/subject)")

        for variant_tag, channel_indices in _channel_variants(cond_cfg, requested_variants):
            for model_name in models:
                combo_label = f"{condition} / {variant_tag} / {model_name}"
                print(f"\n[grid] --- {combo_label} ---")
                t0 = time.time()
                row = run_combo(
                    cond_cfg, condition, variant_tag, channel_indices, model_name,
                    epoch_metadata, k_folds, epochs, device, results_root, args.save_checkpoints,
                )
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(f"[grid] {combo_label}: accuracy={row['accuracy_mean']:.4f} "
                      f"+/- {row['accuracy_std']:.4f} ({row['seconds']:.0f}s)")

    leaderboard = pd.DataFrame(rows)
    leaderboard_path = results_root / "leaderboard.csv"
    if leaderboard_path.exists():
        # Merge with whatever's already there instead of clobbering it -- a second
        # invocation (e.g. a different --models or --channel-variants subset) used to
        # silently overwrite the first run's rows entirely. Keep this run's row for any
        # (condition, channel_variant, model) combo that appears in both.
        previous = pd.read_csv(leaderboard_path)
        key = ["condition", "channel_variant", "model"]
        previous = previous[~previous.set_index(key).index.isin(leaderboard.set_index(key).index)]
        leaderboard = pd.concat([previous, leaderboard], ignore_index=True)
    leaderboard = leaderboard.sort_values("accuracy_mean", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(leaderboard_path, index=False)

    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if leaderboard["channel_variant"].nunique() > 1:
        _leaderboard_chart(leaderboard, "channel_variant", figures_dir / "leaderboard_by_channel.png",
                            "Mean accuracy by channel")
    if leaderboard["model"].nunique() > 1:
        _leaderboard_chart(leaderboard, "model", figures_dir / "leaderboard_by_model.png",
                            "Mean accuracy by CNN backbone")
    if leaderboard["condition"].nunique() > 1:
        _leaderboard_chart(leaderboard, "condition", figures_dir / "leaderboard_by_condition.png",
                            "Mean accuracy by recording condition")

    print(f"\n[grid] === Done in {time.time() - t_start:.0f}s. Leaderboard -> {leaderboard_path} ===")
    print(leaderboard[["condition", "channel_variant", "model", "accuracy_mean", "accuracy_std", "f1_mean"]]
          .to_string(index=False))
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        print(f"\n[grid] BEST combination: condition={best['condition']} channel={best['channel_variant']} "
              f"model={best['model']} -> accuracy={best['accuracy_mean']:.4f} +/- {best['accuracy_std']:.4f}")
        print(f"[grid] best channel overall: "
              f"{leaderboard.groupby('channel_variant')['accuracy_mean'].mean().idxmax()}")
        print(f"[grid] best model overall: "
              f"{leaderboard.groupby('model')['accuracy_mean'].mean().idxmax()}")
        print(f"[grid] best condition overall: "
              f"{leaderboard.groupby('condition')['accuracy_mean'].mean().idxmax()}")


if __name__ == "__main__":
    main()
