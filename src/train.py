"""Per-fold training loop.

Each fold's *training* subjects are further split (by subject, not epoch) into
a train/validation slice — validation is used only for early stopping. The
fold's held-out test subjects are never touched during training or early
stopping, so the reported test metrics in evaluate.py aren't inflated by
model-selection leakage.
"""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from .dataset import EEGImageDataset
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_train_val(train_meta: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    idx = np.arange(len(train_meta))
    tr_idx, val_idx = next(gss.split(idx, train_meta["label"], groups=train_meta["subject_id"]))
    return train_meta.iloc[tr_idx].reset_index(drop=True), train_meta.iloc[val_idx].reset_index(drop=True)


def _run_epoch(model, loader, criterion, optimizer, device) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(model: torch.nn.Module, cfg: SimpleNamespace, path: str | Path, extra: dict | None = None) -> None:
    """Save model weights plus the metadata scripts/predict.py needs to reproduce
    the exact preprocessing/image-generation pipeline this checkpoint expects:
    which backbone, which TF transform/image size, and -- critically -- which
    channels and in which order. Without this, a saved .pt file is just numbers;
    this makes it self-describing so inference on a new .edf file doesn't have
    to guess what the training config was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": cfg.model.backbone,
        "freeze_until": getattr(cfg.model, "freeze_until", None),
        "dropout": cfg.model.dropout,
        "head_hidden_dim": cfg.model.head_hidden_dim,
        "transform": cfg.tfd.transform,
        "image_size": cfg.tfd.image_size,
        "freq_min_hz": cfg.tfd.freq_min_hz,
        "freq_max_hz": cfg.tfd.freq_max_hz,
        "n_freq_bins": cfg.tfd.n_freq_bins,
        "cwt_wavelet": getattr(cfg.tfd, "cwt_wavelet", "cmor1.5-1.0"),
        "spwvd_time_bins": getattr(cfg.tfd, "spwvd_time_bins", 64),
        "sampling_rate_hz": cfg.eeg.sampling_rate_hz,
        "channels": list(cfg.eeg.channels),         # channels the model was preprocessed/epoched with
        "eeg_filter": {
            "bandpass_low_hz": cfg.preprocessing.bandpass_low_hz,
            "bandpass_high_hz": cfg.preprocessing.bandpass_high_hz,
            "notch_hz": cfg.preprocessing.notch_hz,
            "reference": cfg.preprocessing.reference,
            "ica_n_components": cfg.preprocessing.ica_n_components,
            "ica_method": cfg.preprocessing.ica_method,
            "ica_random_state": cfg.preprocessing.ica_random_state,
            "eog_proxy_channels": list(cfg.preprocessing.eog_proxy_channels),
        },
        "epoching": {
            "window_sec": cfg.epoching.window_sec,
            "overlap": cfg.epoching.overlap,
            "reject_peak_to_peak_uv": cfg.epoching.reject_peak_to_peak_uv,
        },
        "label_meaning": {"0": "HC (healthy control)", "1": "MDD (depressed)"},
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def train_one_fold(cfg: SimpleNamespace, fold_train_meta: pd.DataFrame, fold_idx: int, device: torch.device):
    """Trains one fold's model. Returns the trained model (best-val-loss weights restored)."""
    tr_meta, val_meta = _split_train_val(fold_train_meta, cfg.training.val_fraction, cfg.training.seed + fold_idx)

    train_loader = DataLoader(
        EEGImageDataset(tr_meta), batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        EEGImageDataset(val_meta), batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = build_model(cfg).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(cfg.training.epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        print(
            f"[fold {fold_idx}] epoch {epoch + 1}/{cfg.training.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.training.early_stopping_patience:
                print(f"[fold {fold_idx}] early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
