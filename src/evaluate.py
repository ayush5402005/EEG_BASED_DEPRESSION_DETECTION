"""Per-fold metrics + honest cross-validation reporting.

No result gets rounded up, cherry-picked, or presented as "the" accuracy —
every fold's numbers are logged individually, and the headline figure is
always mean +/- std across all folds.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .dataset import EEGImageDataset


def evaluate_fold(model: torch.nn.Module, test_meta: pd.DataFrame, cfg: SimpleNamespace, device: torch.device) -> dict:
    loader = DataLoader(
        EEGImageDataset(test_meta), batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    probs = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),  # rows=true [HC, MDD], cols=pred [HC, MDD]
        "n_test_epochs": int(len(labels)),
        "n_test_subjects": int(test_meta["subject_id"].nunique()),
    }


def _plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["HC", "MDD"])
    ax.set_yticks([0, 1], labels=["HC", "MDD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_and_report(fold_metrics: list[dict], cfg: SimpleNamespace) -> dict:
    """Prints every fold's metrics, saves them all to results/, and returns the summary dict."""
    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Per-fold results (subject-independent test set) ===")
    for i, m in enumerate(fold_metrics):
        print(
            f"Fold {i}: accuracy={m['accuracy']:.4f} precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} f1={m['f1']:.4f} "
            f"(n_epochs={m['n_test_epochs']}, n_subjects={m['n_test_subjects']})"
        )
        print(f"  confusion matrix [[TN,FP],[FN,TP]] = {m['confusion_matrix']}")
        _plot_confusion_matrix(
            np.array(m["confusion_matrix"]), f"Fold {i} confusion matrix",
            results_dir / f"confusion_matrix_fold{i}.png",
        )

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    total_cm = np.sum([np.array(m["confusion_matrix"]) for m in fold_metrics], axis=0)
    _plot_confusion_matrix(total_cm, "Summed confusion matrix (all folds)", results_dir / "confusion_matrix_overall.png")

    print("\n=== Cross-validation summary (mean +/- std across all folds) ===")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key}: {summary[f'{key}_mean']:.4f} +/- {summary[f'{key}_std']:.4f}")
    print(f"Summed confusion matrix [[TN,FP],[FN,TP]]:\n{total_cm}")

    output = {
        "config": {
            "transform": cfg.tfd.transform,
            "backbone": cfg.model.backbone,
            "k_folds": cfg.training.k_folds,
            "window_sec": cfg.epoching.window_sec,
            "overlap": cfg.epoching.overlap,
        },
        "per_fold": fold_metrics,
        "summary_mean_std": summary,
        "summed_confusion_matrix": total_cm.tolist(),
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    pd.DataFrame(fold_metrics).to_csv(results_dir / "per_fold_metrics.csv", index=False)
    return output
