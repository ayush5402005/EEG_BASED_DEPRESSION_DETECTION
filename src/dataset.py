"""PyTorch Dataset over cached TFD images, and the subject-independent k-fold splitter.

The split is the part the spec calls out as critical: every epoch from a given
subject must end up entirely in train OR entirely in test, never both. That is
enforced here with sklearn's StratifiedGroupKFold (`groups=subject_id`), plus a
runtime assertion that double-checks no subject ever leaks across the split.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from . import tfd_transform

# Pretrained ResNet50/VGG16 backbones expect ImageNet-normalized RGB input.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EEGImageDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image = np.load(row["tfd_path"])  # (H, W) float32 in [0, 1]
        rgb = tfd_transform.image_to_rgb_tensor(image)  # (3, H, W)
        rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.from_numpy(rgb.astype(np.float32)), label


def make_subject_kfold_splits(
    metadata: pd.DataFrame, k_folds: int, seed: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yields (train_metadata, test_metadata) per fold, split by subject_id."""
    metadata = metadata.reset_index(drop=True)
    y = metadata["label"].to_numpy()
    groups = metadata["subject_id"].to_numpy()
    dummy_X = np.zeros(len(metadata))

    skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(dummy_X, y, groups):
        train_meta = metadata.iloc[train_idx].reset_index(drop=True)
        test_meta = metadata.iloc[test_idx].reset_index(drop=True)

        overlap = set(train_meta["subject_id"]) & set(test_meta["subject_id"])
        assert not overlap, f"Subject leakage across train/test: {overlap}"

        yield train_meta, test_meta
