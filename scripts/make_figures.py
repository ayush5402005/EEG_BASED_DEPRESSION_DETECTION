#!/usr/bin/env python
"""Generates the illustrative figures referenced by research_paper.tex / report.md:
  - results_v2/figures/pipeline_diagram.png  (schematic block diagram)
  - results_v2/figures/tfd_samples.png       (real SPWVD images: one HC + one MDD
    epoch, for each of the 5 channel variants, drawn from the actual cached
    EC_study epoch data -- not synthetic)

Run after at least one `python scripts/run_experiment_grid.py ... --conditions EC`
invocation has populated data/processed/EC_study/epochs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.tfd_transform import epoch_to_tfd_image

FIG_DIR = Path("results_v2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        "Raw EEG\n(.edf)\n4 ch, 256 Hz",
        "Filter + ICA\n+ re-reference\n(Fp1,Fp2,F3,F4)",
        "Epoching\n4s window\n50% overlap",
        "SPWVD image\nx5 variants\n(4 single + combined)",
        "CNN\n(5 backbones\ncompared)",
        "MDD / HC",
    ]
    n = len(stages)
    box_w, box_h, gap = 1.45, 1.35, 0.25
    total_w = n * box_w + (n - 1) * gap
    x0 = (10 - total_w) / 2
    y0 = (3 - box_h) / 2

    colors = ["#E8EEF7", "#E8EEF7", "#E8EEF7", "#FCEFE3", "#E3F2EF", "#2F6F6B"]
    for i, (stage, color) in enumerate(zip(stages, colors)):
        x = x0 + i * (box_w + gap)
        text_color = "white" if color == "#2F6F6B" else "#14213D"
        ax.add_patch(plt.Rectangle((x, y0), box_w, box_h, facecolor=color,
                                    edgecolor="#2F6F6B", linewidth=1.3, zorder=2))
        ax.text(x + box_w / 2, y0 + box_h / 2, stage, ha="center", va="center",
                 fontsize=8.3, color=text_color, zorder=3, linespacing=1.4)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, y0 + box_h / 2), xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#2F6F6B", lw=1.5), zorder=1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pipeline_diagram.png", dpi=180)
    plt.close(fig)
    print(f"[figures] wrote {FIG_DIR / 'pipeline_diagram.png'}")


def make_tfd_sample_grid() -> None:
    cfg = load_config("configs/channel_study.yaml")
    cfg.data.condition = "EC"
    cfg.data.cache_subdir = "EC_study"
    metadata_path = Path(cfg.data.processed_dir) / cfg.data.cache_subdir / "epochs_metadata.csv"
    if not metadata_path.exists():
        print(f"[figures] {metadata_path} not found -- run the EC_study grid/pilot first. Skipping TF sample grid.")
        return
    metadata = pd.read_csv(metadata_path)

    hc_row = metadata[metadata["group"] == "H"].iloc[0]
    mdd_row = metadata[metadata["group"] == "MDD"].iloc[0]
    channels = list(cfg.eeg.channels)  # Fp1, Fp2, F3, F4
    variants = [(ch, [channels.index(ch)]) for ch in channels] + [("Combined", None)]

    fig, axes = plt.subplots(len(variants), 2, figsize=(5, 2.1 * len(variants)))
    fs = cfg.eeg.sampling_rate_hz
    for row_idx, (label, row) in enumerate([("HC", hc_row), ("MDD", mdd_row)]):
        epochs = np.load(row["epoch_path"])
        epoch_uv = epochs[int(row["epoch_index"])]
        for col_idx, (variant_name, ch_idx) in enumerate(variants):
            image = epoch_to_tfd_image(epoch_uv, fs, cfg, channel_indices=ch_idx)
            ax = axes[col_idx, row_idx]
            ax.imshow(image, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_ylabel(variant_name, fontsize=9)
            if col_idx == 0:
                ax.set_title(f"{label} ({row['subject_id']})", fontsize=10)
    fig.suptitle("SPWVD images: channel variant x HC/MDD", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "tfd_samples.png", dpi=170)
    plt.close(fig)
    print(f"[figures] wrote {FIG_DIR / 'tfd_samples.png'} (HC={hc_row['subject_id']}, MDD={mdd_row['subject_id']})")


if __name__ == "__main__":
    make_pipeline_diagram()
    make_tfd_sample_grid()
