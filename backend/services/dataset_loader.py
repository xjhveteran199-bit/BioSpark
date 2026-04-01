"""
Dataset loader for labeled biosignal training data.

Supported formats:
  - CSV with a 'label' (or 'class'/'target'/'y') column
  - ZIP archive with folder-per-class structure: class_name/sample.csv
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Columns that are treated as time/index rather than signal data
_TIME_COL_NAMES = {"time", "t", "timestamp", "seconds", "sec", "ms", "index"}
# Columns that are treated as labels
_LABEL_COL_NAMES = {"label", "class", "target", "y"}


def _signal_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Return column names that are signal data (not time, not excluded)."""
    return [
        c for c in df.columns
        if c not in exclude and c.strip().lower() not in _TIME_COL_NAMES
    ]


def _parse_labeled_csv(df: pd.DataFrame) -> dict:
    """
    Parse a DataFrame where each row is a timestep annotated with a label.
    The label column must be named 'label', 'class', 'target', or 'y'.
    """
    # Find label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.strip().lower() in _LABEL_COL_NAMES:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            "No label column found. Expected a column named: "
            "'label', 'class', 'target', or 'y'."
        )

    sig_cols = _signal_cols(df, exclude=[label_col])
    if not sig_cols:
        raise ValueError(
            "No signal columns found after excluding the label and time columns."
        )

    labels = df[label_col].astype(str).tolist()
    class_names = sorted(set(labels))
    class_counts = {cls: labels.count(cls) for cls in class_names}

    signal_values = df[sig_cols].values
    preview_vals = signal_values[:500, 0].tolist() if len(signal_values) > 0 else []

    return {
        "format": "csv_labeled",
        "label_column": label_col,
        "signal_columns": sig_cols,
        "class_names": class_names,
        "class_counts": class_counts,
        "total_samples": len(df),
        "signal_length": len(df),
        "n_channels": len(sig_cols),
        "preview": {
            "values": preview_vals,
            "channel_name": sig_cols[0],
        },
    }


def _parse_zip_dataset(zip_bytes: bytes) -> dict:
    """
    Parse a ZIP archive with folder-per-class structure.
    Expected layout: <class_name>/<any_depth>/sample.csv
    The immediate parent folder of each CSV is used as the class name.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # Map class_name → list of zip entry paths
        class_files: dict[str, list[str]] = {}
        for name in names:
            parts = Path(name).parts
            # Skip macOS metadata, hidden entries, and non-CSV files
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith(".csv"):
                continue
            if len(parts) < 2:
                continue  # CSV at root — no class folder
            class_name = parts[-2]  # immediate parent folder = class label
            class_files.setdefault(class_name, []).append(name)

        if not class_files:
            raise ValueError(
                "No class folders with CSV files found in the ZIP archive. "
                "Expected structure: class_name/sample.csv"
            )

        class_names = sorted(class_files.keys())
        class_counts = {cls: len(files) for cls, files in class_files.items()}
        total_samples = sum(class_counts.values())

        # Sample the first file to get signal shape
        first_file = class_files[class_names[0]][0]
        with zf.open(first_file) as f:
            sample_df = pd.read_csv(f)

        sig_cols = _signal_cols(sample_df, exclude=[])
        signal_length = len(sample_df)
        n_channels = len(sig_cols)

        preview_vals = (
            sample_df[sig_cols[0]].values[:500].tolist() if sig_cols else []
        )

        return {
            "format": "zip_folder",
            "signal_columns": sig_cols,
            "class_names": class_names,
            "class_counts": class_counts,
            "total_samples": total_samples,
            "signal_length": signal_length,
            "n_channels": n_channels,
            "preview": {
                "values": preview_vals,
                "channel_name": sig_cols[0] if sig_cols else "ch0",
            },
        }


def load_labeled_dataset(filename: str, file_bytes: bytes) -> dict:
    """
    Parse a labeled biosignal dataset from raw file bytes.

    Parameters
    ----------
    filename : str
        Original filename (used to detect format via extension).
    file_bytes : bytes
        Raw file content.

    Returns
    -------
    dict with keys:
        format, class_names, class_counts, total_samples,
        signal_length, n_channels, signal_columns, preview
    """
    ext = Path(filename).suffix.lower()

    if ext == ".zip":
        return _parse_zip_dataset(file_bytes)

    if ext in (".csv", ".txt"):
        text = file_bytes.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        return _parse_labeled_csv(df)

    raise ValueError(
        f"Unsupported file format: '{ext}'. "
        "Upload a CSV file (with a 'label' column) or a ZIP archive "
        "(with folder-per-class structure)."
    )
