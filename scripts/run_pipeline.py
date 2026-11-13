#!/usr/bin/env python
"""Orchestrates the full pipeline: preprocess -> epoch -> TFD images -> subject-wise
k-fold train/eval -> report.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import make_subject_kfold_splits
from src.evaluate import aggregate_and_report, evaluate_fold
from src.preprocessing import build_epoch_dataset
from src.tfd_transform import build_tfd_dataset
from src.train import get_device, save_checkpoint, set_seed, train_one_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EEG depression-detection pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-preprocess", action="store_true", help="Re-run preprocessing even if cached.")
    parser.add_argument("--force-tfd", action="store_true", help="Re-run TFD image generation even if cached.")
    parser.add_argument(
        "--no-checkpoints", action="store_true",
        help="Skip saving model weights (by default, every fold's model is saved under "
             "<results_dir>/checkpoints/, plus the best-accuracy fold copied to best_model.pt, "
             "for later inference with scripts/predict.py).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"[run_pipeline] using device: {device}")

    print("[run_pipeline] Stage 1/3: preprocessing (load, filter, ICA, epoch)...")
    epoch_metadata = build_epoch_dataset(cfg, force=args.force_preprocess)
    print(f"[run_pipeline] {len(epoch_metadata)} epochs across {epoch_metadata['subject_id'].nunique()} subjects")

    print(f"[run_pipeline] Stage 2/3: generating TFD images ({cfg.tfd.transform})...")
    tfd_metadata = build_tfd_dataset(cfg, epoch_metadata, force=args.force_tfd)

    print(f"[run_pipeline] Stage 3/3: subject-wise {cfg.training.k_folds}-fold training + evaluation...")
    fold_metrics = []
    for fold_idx, (train_meta, test_meta) in enumerate(
        make_subject_kfold_splits(tfd_metadata, cfg.training.k_folds, cfg.training.seed)
    ):
        print(
            f"\n--- Fold {fold_idx}: {train_meta['subject_id'].nunique()} train subjects, "
            f"{test_meta['subject_id'].nunique()} test subjects ---"
        )
        model = train_one_fold(cfg, train_meta, fold_idx, device)
        fold_metrics.append(evaluate_fold(model, test_meta, cfg, device))
        if not args.no_checkpoints:
            ckpt_path = Path(cfg.output.results_dir) / "checkpoints" / f"fold{fold_idx}.pt"
            save_checkpoint(model, cfg, ckpt_path, extra={"fold_idx": fold_idx})
            print(f"[run_pipeline] saved checkpoint -> {ckpt_path}")

    summary = aggregate_and_report(fold_metrics, cfg)

    if not args.no_checkpoints:
        best_fold = max(range(len(fold_metrics)), key=lambda i: fold_metrics[i]["accuracy"])
        best_src = Path(cfg.output.results_dir) / "checkpoints" / f"fold{best_fold}.pt"
        best_dst = Path(cfg.output.results_dir) / "checkpoints" / "best_model.pt"
        best_dst.write_bytes(best_src.read_bytes())
        print(
            f"[run_pipeline] best fold: {best_fold} (accuracy={fold_metrics[best_fold]['accuracy']:.4f}) "
            f"-> {best_dst}"
        )


if __name__ == "__main__":
    main()
