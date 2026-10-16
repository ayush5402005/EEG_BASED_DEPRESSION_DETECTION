"""Raw EDF -> cleaned, epoched EEG.

Stages (see README.md for the plain-English version):
  1. Load an .edf recording and keep only the 19 standard 10-20 channels,
     matched by name (channel order/extra channels vary between files).
  2. Bandpass filter + mains notch filter.
  3. ICA-based artifact removal.
  4. Average re-reference.
  5. Fixed-length epoching (window/overlap from config) with subject_id and
     label tracked per epoch, plus simple amplitude-based bad-epoch rejection.

Assumption (documented per spec): this dataset has no dedicated EOG/EMG
channel, so ICA components are flagged as eye-blink artifacts using the
frontal channels (Fp1/Fp2) as an EOG proxy via MNE's find_bads_eog, and as
muscle artifacts via MNE's find_bads_muscle heuristic. This is a proxy-based
approach, not true EOG-channel regression.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

FILENAME_RE = re.compile(
    r"^(?P<group>MDD|H)\s+S(?P<subject>\d+)\s+(?P<condition>EC|EO|TASK)\.edf$",
    re.IGNORECASE,
)
# EDF channel labels look like "EEG Fp1-LE" or "EEG A2-A1" — pull out the
# electrode name between "EEG " and the reference suffix.
CHANNEL_LABEL_RE = re.compile(r"^EEG\s+([A-Za-z0-9]+)-", re.IGNORECASE)


@dataclass
class RecordingInfo:
    path: Path
    group: str        # "MDD" or "H"
    subject_num: int
    subject_id: str    # unique across groups, e.g. "MDD_S1", "H_S1"
    label: int          # 1 = MDD (depressed), 0 = healthy control
    condition: str


def discover_recordings(raw_dir, condition: str, mdd_prefix: str, hc_prefix: str) -> list[RecordingInfo]:
    """Find every EDF file for the given condition and parse subject/group from its filename."""
    raw_dir = Path(raw_dir)
    recordings = []
    for path in sorted(raw_dir.glob("*.edf")):
        m = FILENAME_RE.match(path.name)
        if not m or m.group("condition").upper() != condition.upper():
            continue
        group = m.group("group").upper()
        subject_num = int(m.group("subject"))
        label = 1 if group == mdd_prefix.upper() else 0
        recordings.append(
            RecordingInfo(
                path=path,
                group=group,
                subject_num=subject_num,
                subject_id=f"{group}_S{subject_num}",
                label=label,
                condition=condition.upper(),
            )
        )
    return recordings


def _pick_and_rename_channels(raw: mne.io.BaseRaw, target_channels: list[str]) -> mne.io.BaseRaw:
    target_lookup = {c.lower(): c for c in target_channels}
    rename_map = {}
    for ch in raw.ch_names:
        m = CHANNEL_LABEL_RE.match(ch)
        if not m:
            continue
        electrode = m.group(1).lower()
        if electrode in target_lookup:
            rename_map[ch] = target_lookup[electrode]

    missing = set(target_channels) - set(rename_map.values())
    if missing:
        raise ValueError(f"{raw.filenames[0]}: missing expected channels {sorted(missing)}")

    raw.pick(list(rename_map.keys()))
    raw.rename_channels(rename_map)
    raw.reorder_channels(target_channels)
    return raw


def load_and_clean_raw(path: Path, cfg: SimpleNamespace) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw = _pick_and_rename_channels(raw, list(cfg.eeg.channels))
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")

    raw.filter(
        l_freq=cfg.preprocessing.bandpass_low_hz,
        h_freq=cfg.preprocessing.bandpass_high_hz,
        fir_design="firwin",
        verbose=False,
    )
    raw.notch_filter(freqs=[cfg.preprocessing.notch_hz], verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = mne.preprocessing.ICA(
            n_components=cfg.preprocessing.ica_n_components,
            method=cfg.preprocessing.ica_method,
            random_state=cfg.preprocessing.ica_random_state,
            max_iter="auto",
        )
        raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica.fit(raw_hp)

        bad_components: set[int] = set()
        try:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=list(cfg.preprocessing.eog_proxy_channels))
            bad_components.update(eog_idx)
        except RuntimeError:
            pass  # no clear EOG-like component found; nothing to exclude
        try:
            muscle_idx, _ = ica.find_bads_muscle(raw)
            bad_components.update(muscle_idx)
        except RuntimeError:
            pass

        ica.exclude = sorted(bad_components)
        ica.apply(raw)

    raw.set_eeg_reference(cfg.preprocessing.reference, verbose=False)
    return raw


def epoch_raw(raw: mne.io.BaseRaw, cfg: SimpleNamespace) -> np.ndarray:
    """Fixed-length epochs -> array of shape (n_epochs, n_channels, n_times), in microvolts."""
    window_sec = float(cfg.epoching.window_sec)
    overlap_sec = window_sec * float(cfg.epoching.overlap)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=window_sec, overlap=overlap_sec, preload=True, verbose=False
    )
    epochs.drop_bad(reject=dict(eeg=cfg.epoching.reject_peak_to_peak_uv * 1e-6))
    return epochs.get_data() * 1e6  # volts -> microvolts


def build_epoch_dataset(cfg: SimpleNamespace, force: bool = False) -> pd.DataFrame:
    """Process every discovered recording, cache per-subject epoch arrays, and return metadata.

    Metadata columns: subject_id, group, label, epoch_index, epoch_path, n_channels, n_times.
    Skips a recording (with a logged warning) rather than aborting the whole run if it fails
    to load/process — a handful of edge-case files should not block the rest of the cohort.

    Caching note: `cfg.data.cache_subdir`, if set, namespaces the cache under
    `processed_dir/<cache_subdir>/...` instead of the flat `processed_dir/...`
    layout. This exists so the condition x channel-set comparison study (see
    scripts/run_experiment_grid.py) can cache EC/EO/TASK and different channel
    subsets side by side without overwriting each other or the original EC/
    19-channel cache this project's first results were computed from. Configs
    that don't set it (e.g. configs/default.yaml) are completely unaffected —
    they keep using the original flat paths exactly as before.
    """
    cache_subdir = getattr(cfg.data, "cache_subdir", "") or ""
    base_dir = Path(cfg.data.processed_dir) / cache_subdir if cache_subdir else Path(cfg.data.processed_dir)
    processed_dir = base_dir / "epochs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / "epochs_metadata.csv"

    if metadata_path.exists() and not force:
        return pd.read_csv(metadata_path)

    recordings = discover_recordings(
        cfg.data.raw_dir, cfg.data.condition, cfg.data.mdd_prefix, cfg.data.hc_prefix
    )
    if not recordings:
        raise RuntimeError(f"No '{cfg.data.condition}' EDF files found under {cfg.data.raw_dir}")

    rows = []
    for rec in recordings:
        try:
            raw = load_and_clean_raw(rec.path, cfg)
            data = epoch_raw(raw, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the run
            print(f"[preprocessing] SKIPPING {rec.path.name}: {exc}")
            continue

        out_path = processed_dir / f"{rec.subject_id}.npy"
        np.save(out_path, data.astype(np.float32))
        print(f"[preprocessing] {rec.subject_id}: {data.shape[0]} epochs kept -> {out_path}")

        for epoch_index in range(data.shape[0]):
            rows.append(
                {
                    "subject_id": rec.subject_id,
                    "group": rec.group,
                    "label": rec.label,
                    "epoch_index": epoch_index,
                    "epoch_path": str(out_path),
                    "n_channels": data.shape[1],
                    "n_times": data.shape[2],
                }
            )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_path, index=False)
    return metadata
