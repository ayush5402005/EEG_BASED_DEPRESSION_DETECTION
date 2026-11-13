"""Epoch (channels x time, microvolts) -> a single 2D time-frequency image.

Two transforms are implemented, selected via `cfg.tfd.transform`:
  - "cwt"   (default): Continuous Wavelet Transform, complex Morlet wavelet, via PyWavelets.
  - "spwvd": Smoothed Pseudo Wigner-Ville Distribution, implemented directly with
    scipy (see note below on why this isn't the `tftb` package).

Multi-channel handling (explicit decision — the spec leaves this open):
  A scalogram/SPWVD is computed per channel, then the 19 channel images are
  AVERAGED into one 2D image. That image is resized to `image_size x image_size`
  and replicated across 3 channels so it can feed a pretrained RGB CNN backbone.
  Rationale: averaging requires no arbitrary choice of "the important" channels
  (which would itself be an undocumented bias) and keeps the pretrained
  first-conv-layer weights meaningful (no reshaping the backbone's input).
  Trade-off: this discards inter-channel spatial/topographic information (e.g.
  frontal alpha asymmetry, a known depression biomarker) — a real limitation,
  documented here and in README.md rather than silently assumed away.

Note on SPWVD: the obvious library choice, `tftb` (pip name `tftb`, the Python
port of the MATLAB TFTB toolbox used e.g. by Khare, Bajaj & Acharya, "PDCNNet,"
IEEE Sensors J., 2021, for this exact SPWVD-CNN recipe), is NOT installable
alongside this project's deep-learning stack: its only release on PyPI (0.2.0)
pins numpy==1.26.4 exactly, and attempting to build that numpy version here
fails outright ("NumPy requires GCC >= 8.4" -- verified by actually trying the
install, not assumed). Downgrading the whole project's numpy to satisfy one
dependency would break torch/mne/scikit-image, which all expect numpy>=2.
Rather than doing that, SPWVD is implemented directly here following the
standard discrete formula (Boashash; the same Eq. (1) used in the PDCNNet
reference above), evaluated only at the requested time/frequency bins for
speed. As of this version, SPWVD (not CWT) is the pipeline's default
transform, config: `tfd.transform: "spwvd"`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from scipy.signal.windows import kaiser
from skimage.transform import resize


def _frequency_axis(cfg: SimpleNamespace) -> np.ndarray:
    return np.geomspace(cfg.tfd.freq_min_hz, cfg.tfd.freq_max_hz, cfg.tfd.n_freq_bins)


def cwt_scalogram(signal_uv: np.ndarray, fs: float, freqs: np.ndarray, wavelet: str) -> np.ndarray:
    """Magnitude scalogram, shape (n_freqs, n_times)."""
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * fs / freqs
    coeffs, _ = pywt.cwt(signal_uv, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coeffs)


def spwvd_scalogram(
    signal_uv: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    time_bins: int,
    lag_window_len: int = 63,
    time_window_len: int = 31,
    kaiser_beta: float = 5.0,
) -> np.ndarray:
    """Smoothed Pseudo Wigner-Ville Distribution, shape (n_freqs, time_bins).

    Window lengths follow the PDCNNet reference (Khare, Bajaj & Acharya, 2021):
    63 taps for the frequency-domain (lag) smoothing window and 31 for the
    time-domain smoothing window, both of the form 2^m - 1 for fast FFT-free
    evaluation. That paper also uses a Kaiser window for both g(.) and h(.) in
    Eq. (1) (rather than this project's earlier Hamming window), which is
    matched here; `kaiser_beta` is not given a specific value in the reference,
    so beta=5.0 is used as a documented, reasonable default (moderate
    main-lobe/sidelobe trade-off) rather than an unstated assumption.

    Evaluated only at `time_bins` positions (not every sample) and directly at
    the requested `freqs` (via a small DFT-style matmul instead of a full FFT),
    which keeps this fast enough to run per-epoch, per-channel.
    """
    n_samples = len(signal_uv)
    z = hilbert(signal_uv)

    half_lag = lag_window_len // 2
    half_time = time_window_len // 2
    pad = half_lag + half_time
    z_padded = np.pad(z, pad, mode="reflect")

    time_smooth = kaiser(time_window_len, kaiser_beta)
    time_smooth = time_smooth / time_smooth.sum()
    lag_smooth = kaiser(lag_window_len, kaiser_beta)

    tau = np.arange(-half_lag, half_lag + 1)
    phases = np.exp(-1j * 4 * np.pi * np.outer(freqs, tau) / fs)  # (n_freqs, lag_len)

    time_positions = np.linspace(0, n_samples - 1, time_bins).astype(int)
    tf_image = np.zeros((len(freqs), time_bins), dtype=np.float64)

    for ti, n in enumerate(time_positions):
        n_p = n + pad
        local_corr = np.zeros(len(tau), dtype=complex)
        for gi, m in enumerate(range(-half_time, half_time + 1)):
            local_corr += time_smooth[gi] * (
                z_padded[n_p + m + tau] * np.conj(z_padded[n_p + m - tau])
            )
        windowed = lag_smooth * local_corr
        tf_image[:, ti] = np.real(2 * (phases @ windowed))

    return np.abs(tf_image)


def _resize_normalize(image: np.ndarray, image_size: int) -> np.ndarray:
    image = resize(image, (image_size, image_size), anti_aliasing=True)
    image = np.log1p(image)
    lo, hi = image.min(), image.max()
    image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    return image.astype(np.float32)


def epoch_to_tfd_image(
    epoch_uv: np.ndarray, fs: float, cfg: SimpleNamespace, channel_indices: list[int] | None = None
) -> np.ndarray:
    """epoch_uv: (n_channels, n_times) -> single-channel image (image_size, image_size), float32 in [0, 1].

    `channel_indices`: which rows of `epoch_uv` to use, e.g. [0] for a single
    channel's own image (no averaging -- the mean of one channel is itself) or
    e.g. [0, 1, 2, 3] to average a chosen subset. `None` (default) uses every
    channel in `epoch_uv`, matching the original channel-averaged behaviour.
    Used by the channel/condition/model comparison study (see
    scripts/run_experiment_grid.py) to generate one image variant per
    candidate channel plus one "combined" (averaged) variant, from the same
    preprocessed epoch, without re-running preprocessing per variant.
    """
    if channel_indices is not None:
        epoch_uv = epoch_uv[list(channel_indices)]
    freqs = _frequency_axis(cfg)
    n_channels = epoch_uv.shape[0]

    if cfg.tfd.transform == "cwt":
        per_channel = np.empty((n_channels, cfg.tfd.n_freq_bins, epoch_uv.shape[1]), dtype=np.float64)
        for c in range(n_channels):
            per_channel[c] = cwt_scalogram(epoch_uv[c], fs, freqs, cfg.tfd.cwt_wavelet)
    elif cfg.tfd.transform == "spwvd":
        per_channel = np.empty((n_channels, cfg.tfd.n_freq_bins, cfg.tfd.spwvd_time_bins), dtype=np.float64)
        for c in range(n_channels):
            per_channel[c] = spwvd_scalogram(epoch_uv[c], fs, freqs, time_bins=cfg.tfd.spwvd_time_bins)
    else:
        raise ValueError(f"Unknown tfd.transform: {cfg.tfd.transform!r} (expected 'cwt' or 'spwvd')")

    averaged = per_channel.mean(axis=0)  # documented multi-channel combination strategy
    return _resize_normalize(averaged, cfg.tfd.image_size)


def image_to_rgb_tensor(image: np.ndarray) -> np.ndarray:
    """(H, W) float32 in [0, 1] -> (3, H, W) float32, replicated across channels."""
    return np.repeat(image[np.newaxis, :, :], 3, axis=0)


def build_tfd_dataset(
    cfg: SimpleNamespace,
    epoch_metadata: pd.DataFrame,
    force: bool = False,
    channel_indices: list[int] | None = None,
    variant_tag: str = "",
) -> pd.DataFrame:
    """Compute (or load cached) TFD images for every epoch listed in `epoch_metadata`.

    Caches one .npy per epoch (single-channel image, RGB replication happens at
    load time in the Dataset to avoid tripling disk usage) and returns metadata
    with a `tfd_path` column added.

    `channel_indices` / `variant_tag`: used by the channel comparison study to
    produce several image variants (e.g. one per candidate channel, plus a
    "combined" average) from the same cached epochs. `variant_tag` namespaces
    the cache path so variants never collide; leaving both at their defaults
    reproduces the original single-variant (all channels averaged) cache paths
    exactly, so existing runs/results are unaffected.
    """
    cache_subdir = getattr(cfg.data, "cache_subdir", "") or ""
    base_dir = Path(cfg.data.processed_dir) / cache_subdir if cache_subdir else Path(cfg.data.processed_dir)
    suffix = f"_{variant_tag}" if variant_tag else ""
    out_dir = base_dir / "tfd_images" / f"{cfg.tfd.transform}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_metadata_path = base_dir / f"tfd_metadata_{cfg.tfd.transform}{suffix}.csv"

    if cache_metadata_path.exists() and not force:
        return pd.read_csv(cache_metadata_path)

    fs = cfg.eeg.sampling_rate_hz
    tfd_paths = []
    loaded_subjects: dict[str, np.ndarray] = {}

    for _, row in epoch_metadata.iterrows():
        subject_id = row["subject_id"]
        if subject_id not in loaded_subjects:
            loaded_subjects[subject_id] = np.load(row["epoch_path"])
        epoch_uv = loaded_subjects[subject_id][int(row["epoch_index"])]

        out_path = out_dir / f"{subject_id}_{int(row['epoch_index'])}.npy"
        if force or not out_path.exists():
            image = epoch_to_tfd_image(epoch_uv, fs, cfg, channel_indices=channel_indices)
            np.save(out_path, image)
        tfd_paths.append(str(out_path))

    result = epoch_metadata.copy()
    result["tfd_path"] = tfd_paths
    result.to_csv(cache_metadata_path, index=False)
    return result
