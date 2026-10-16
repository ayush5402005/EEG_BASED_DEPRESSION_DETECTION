#!/usr/bin/env python
"""Run a trained checkpoint on a NEW .edf recording (one not used for training).

This is research-prototype inference, not a diagnostic tool -- same caveat as
the rest of this project (see README.md's opening line). It answers "what
does the model say about this recording", nothing more.

The checkpoint (saved by src/train.save_checkpoint, from scripts/run_pipeline.py
or scripts/run_experiment_grid.py) stores not just weights but the exact
preprocessing/channel/transform settings it was trained with, so this script
reproduces that pipeline exactly -- it does not re-guess any config.

Usage:
    python scripts/predict.py --checkpoint results/checkpoints/best_model.pt \\
        --edf "data/raw/H S2 EC.edf"

    python scripts/predict.py --checkpoint results_v2/full_grid/EC/fp1/resnet18/checkpoints/best_model.pt \\
        --edf path/to/some_new_recording.edf --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model
from src.preprocessing import epoch_raw, load_and_clean_raw
from src.tfd_transform import epoch_to_tfd_image, image_to_rgb_tensor


def _cfg_from_checkpoint(ckpt: dict) -> SimpleNamespace:
    """Rebuilds the minimal cfg object load_and_clean_raw / epoch_raw / epoch_to_tfd_image
    need, from exactly what the checkpoint recorded -- not from any config file on disk,
    so inference reproduces the training run's pipeline even if configs/*.yaml changes later.
    """
    f = ckpt["eeg_filter"]
    ep = ckpt["epoching"]
    return SimpleNamespace(
        eeg=SimpleNamespace(channels=list(ckpt["channels"]), sampling_rate_hz=ckpt["sampling_rate_hz"]),
        preprocessing=SimpleNamespace(
            bandpass_low_hz=f["bandpass_low_hz"],
            bandpass_high_hz=f["bandpass_high_hz"],
            notch_hz=f["notch_hz"],
            ica_n_components=f["ica_n_components"],
            ica_method=f["ica_method"],
            ica_random_state=f["ica_random_state"],
            eog_proxy_channels=f["eog_proxy_channels"],
            reference=f["reference"],
        ),
        epoching=SimpleNamespace(
            window_sec=ep["window_sec"], overlap=ep["overlap"],
            reject_peak_to_peak_uv=ep["reject_peak_to_peak_uv"],
        ),
        tfd=SimpleNamespace(
            transform=ckpt["transform"], image_size=ckpt["image_size"],
            freq_min_hz=ckpt["freq_min_hz"], freq_max_hz=ckpt["freq_max_hz"],
            n_freq_bins=ckpt["n_freq_bins"], cwt_wavelet=ckpt.get("cwt_wavelet", "cmor1.5-1.0"),
            spwvd_time_bins=ckpt.get("spwvd_time_bins", 64),
        ),
        model=SimpleNamespace(
            backbone=ckpt["backbone"], pretrained=False, freeze_until=ckpt.get("freeze_until") or "",
            dropout=ckpt["dropout"], head_hidden_dim=ckpt["head_hidden_dim"],
        ),
    )


def _channel_indices(ckpt: dict) -> list[int] | None:
    """Which channel(s) of ckpt['channels'] this model's images were built from --
    mirrors scripts/run_experiment_grid.py's _channel_variants(). 'combined' (or no
    tag at all, e.g. a checkpoint from the original single-model pipeline, which
    always averaged every channel) means every channel, i.e. None.
    """
    variant = ckpt.get("channel_variant", "combined")
    if variant == "combined":
        return None
    channels_lower = [c.lower() for c in ckpt["channels"]]
    if variant.lower() not in channels_lower:
        raise ValueError(f"Checkpoint's channel_variant={variant!r} not found in its channels {ckpt['channels']}")
    return [channels_lower.index(variant.lower())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained EEG-depression checkpoint on a new .edf file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt file saved by src.train.save_checkpoint.")
    parser.add_argument("--edf", required=True, help="Path to the new .edf recording to classify.")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda'; default: cuda if available, else cpu.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(ckpt)
    channel_indices = _channel_indices(ckpt)

    print(f"[predict] checkpoint: backbone={ckpt['backbone']} transform={ckpt['transform']} "
          f"channels={ckpt['channels']} channel_variant={ckpt.get('channel_variant', 'combined')} "
          f"trained_condition={ckpt.get('condition', 'unknown')}")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[predict] loading + preprocessing {args.edf} (filter, ICA, re-reference -- this can take a minute)...")
    raw = load_and_clean_raw(Path(args.edf), cfg)
    epochs_uv = epoch_raw(raw, cfg)  # (n_epochs, n_channels, n_times), after bad-epoch rejection
    n_epochs = epochs_uv.shape[0]
    if n_epochs == 0:
        print("[predict] No usable epochs survived preprocessing/artifact-rejection for this file. Stopping.")
        return
    print(f"[predict] {n_epochs} usable epochs after preprocessing + artifact rejection")

    fs = cfg.eeg.sampling_rate_hz
    probs = []
    with torch.no_grad():
        for start in range(0, n_epochs, args.batch_size):
            batch_uv = epochs_uv[start:start + args.batch_size]
            images = []
            for epoch_uv in batch_uv:
                image = epoch_to_tfd_image(epoch_uv, fs, cfg, channel_indices=channel_indices)
                rgb = image_to_rgb_tensor(image)
                rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
                images.append(rgb.astype(np.float32))
            batch_tensor = torch.from_numpy(np.stack(images)).to(device)
            logits = model(batch_tensor)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    mean_prob = float(probs.mean())
    majority_label = int(round(preds.mean()))
    label_name = {0: "HC (healthy control)", 1: "MDD (depressed)"}

    print(f"\n[predict] per-epoch MDD probability: min={probs.min():.3f} max={probs.max():.3f} "
          f"mean={mean_prob:.3f} median={np.median(probs):.3f}")
    print(f"[predict] epochs predicted MDD: {int(preds.sum())}/{n_epochs} ({preds.mean():.1%})")
    print(f"\n[predict] SUBJECT-LEVEL VERDICT (mean epoch probability, threshold 0.5): "
          f"{label_name[int(round(mean_prob))]}  (mean prob = {mean_prob:.3f})")
    print(f"[predict] SUBJECT-LEVEL VERDICT (majority vote across epochs): "
          f"{label_name[majority_label]}  ({preds.mean():.1%} of epochs voted MDD)")
    print("\n[predict] Reminder: this is a research pipeline's output on one recording, not a "
          "clinical diagnosis. See README.md.")


if __name__ == "__main__":
    main()
