"""Multi-format biosignal file parser.

Supports CSV, EDF, and MAT formats. Returns unified signal data structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def parse_file(file_path: str, signal_type: Optional[str] = None) -> dict:
    """Parse a biosignal file and return unified data structure.

    Returns:
        dict with keys:
            - data: np.ndarray of shape (n_channels, n_samples)
            - channels: list of channel names
            - sampling_rate: float
            - duration_sec: float
            - signal_type: str (ecg/eeg/emg or auto-detected)
            - format: str (csv/edf/mat)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return _parse_csv(path, signal_type)
    elif ext in (".edf", ".bdf"):
        return _parse_edf(path, signal_type)
    elif ext == ".mat":
        return _parse_mat(path, signal_type)
    elif ext == ".txt":
        return _parse_csv(path, signal_type)  # treat .txt as CSV
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _parse_csv(path: Path, signal_type: Optional[str]) -> dict:
    """Parse CSV/TXT biosignal files.

    Expected formats:
    - Single column: single-channel signal, one sample per row
    - Multiple columns: multi-channel, each column is a channel
    - Optional header row with channel names
    - Optional 'time' or 'timestamp' column (will be excluded from signals)
    """
    df = pd.read_csv(path)

    # Drop time columns if present
    time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp", "t", "sample", "index")]
    if time_cols:
        df = df.drop(columns=time_cols)

    # Convert to numeric, drop non-numeric columns
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if df.empty:
        raise ValueError("No numeric data found in CSV file")

    data = df.values.T  # (n_channels, n_samples)
    channels = list(df.columns) if not all(isinstance(c, int) for c in df.columns) else [f"Ch{i+1}" for i in range(data.shape[0])]

    # Guess sampling rate from data length and signal type
    sampling_rate = _guess_sampling_rate(data.shape[1], signal_type)

    return {
        "data": data.astype(np.float64),
        "channels": channels,
        "sampling_rate": sampling_rate,
        "duration_sec": data.shape[1] / sampling_rate,
        "signal_type": signal_type or _guess_signal_type(channels, sampling_rate),
        "format": "csv",
    }


def _parse_edf(path: Path, signal_type: Optional[str]) -> dict:
    """Parse EDF/BDF files using pyedflib."""
    import pyedflib

    f = pyedflib.EdfReader(str(path))
    try:
        n_channels = f.signals_in_file
        channels = f.getSignalLabels()
        sampling_rates = [f.getSampleFrequency(i) for i in range(n_channels)]

        # Use the most common sampling rate
        sr = max(set(sampling_rates), key=sampling_rates.count)

        data = []
        selected_channels = []
        for i in range(n_channels):
            if sampling_rates[i] == sr:
                data.append(f.readSignal(i))
                selected_channels.append(channels[i])

        data = np.array(data)  # (n_channels, n_samples)
    finally:
        f.close()

    return {
        "data": data.astype(np.float64),
        "channels": selected_channels,
        "sampling_rate": sr,
        "duration_sec": data.shape[1] / sr,
        "signal_type": signal_type or _guess_signal_type(selected_channels, sr),
        "format": "edf",
    }


def _parse_mat(path: Path, signal_type: Optional[str]) -> dict:
    """Parse MATLAB .mat files using scipy."""
    from scipy.io import loadmat

    mat = loadmat(str(path))

    # Find the main data array (largest numeric array, skip metadata keys)
    data_key = None
    max_size = 0
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray) and val.dtype.kind in ("f", "i", "u"):
            if val.size > max_size:
                max_size = val.size
                data_key = key

    if data_key is None:
        raise ValueError("No numeric data array found in MAT file")

    data = mat[data_key].astype(np.float64)

    # Ensure shape is (n_channels, n_samples) — assume more samples than channels
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.shape[0] > data.shape[1]:
        data = data.T

    channels = [f"Ch{i+1}" for i in range(data.shape[0])]
    sampling_rate = _guess_sampling_rate(data.shape[1], signal_type)

    return {
        "data": data,
        "channels": channels,
        "sampling_rate": sampling_rate,
        "duration_sec": data.shape[1] / sampling_rate,
        "signal_type": signal_type or _guess_signal_type(channels, sampling_rate),
        "format": "mat",
    }


def _guess_sampling_rate(n_samples: int, signal_type: Optional[str]) -> float:
    """Guess sampling rate based on signal type defaults."""
    defaults = {"ecg": 360.0, "eeg": 256.0, "emg": 1000.0}
    if signal_type and signal_type in defaults:
        return defaults[signal_type]
    return 256.0  # Default fallback


def _guess_signal_type(channels: list[str], sampling_rate: float) -> str:
    """Guess signal type from channel names and sampling rate."""
    ch_lower = [c.lower() for c in channels]
    ch_str = " ".join(ch_lower)

    if any(k in ch_str for k in ("ecg", "lead", "mlii", "v1", "v2", "v3", "v4", "v5", "v6", "avr", "avl", "avf")):
        return "ecg"
    if any(k in ch_str for k in ("eeg", "fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4", "o1", "o2", "fz", "cz", "pz")):
        return "eeg"
    if any(k in ch_str for k in ("emg", "muscle", "flexor", "extensor")):
        return "emg"

    # Guess from sampling rate
    if sampling_rate >= 500:
        return "emg"
    if sampling_rate <= 128:
        return "eeg"
    return "ecg"
