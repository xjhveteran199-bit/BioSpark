"""
Training engine for user-uploaded labeled datasets.

Reuses the proven ECGArrhythmiaCNN architecture (3-block 1D-CNN) and
auto-adjusts the output layer to match the number of classes in the
user's dataset.  Training runs in a background thread and pushes
per-epoch metrics via async callbacks so a WebSocket handler can
forward them to the browser in real-time.
"""

import asyncio
import io
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.services.dataset_loader import _TIME_COL_NAMES, _LABEL_COL_NAMES


# ---------------------------------------------------------------------------
# Generic 1-D CNN (same topology as ECGArrhythmiaCNN)
# ---------------------------------------------------------------------------

class Signal1DCNN(nn.Module):
    """Lightweight 1D-CNN that works for any signal length / class count.

    3 conv blocks → global-avg-pool → FC head.
    Also exposes an ``extract_features`` method for t-SNE (Phase 3).
    """

    def __init__(self, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 128, 1)
        x = x.squeeze(-1)          # (B, 128)
        return self.classifier(x)  # (B, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-d feature vector (penultimate layer) for t-SNE."""
        x = self.features(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Training job state
# ---------------------------------------------------------------------------

class TrainingJob:
    """Holds all state for a single training run."""

    def __init__(self, job_id: str, config: dict, class_names: list[str]):
        self.job_id = job_id
        self.status = "pending"  # pending | training | completed | failed
        self.config = config
        self.class_names = class_names
        self.history: list[dict] = []         # per-epoch metrics
        self.best_val_acc: float = 0.0
        self.error: Optional[str] = None
        self.model: Optional[Signal1DCNN] = None
        self.val_X: Optional[np.ndarray] = None
        self.val_y: Optional[np.ndarray] = None
        self.n_channels: int = 1              # resolved channel count

        # Async callbacks — called from the training thread with an
        # ``asyncio.run_coroutine_threadsafe`` call.
        self._callbacks: list[Callable] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_callback(self, cb: Callable, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._callbacks.append(cb)

    def unregister_callback(self, cb: Callable):
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    def _emit(self, payload: dict):
        """Send a payload to all registered WebSocket callbacks."""
        for cb in list(self._callbacks):
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(cb(payload), self._loop)


# ---------------------------------------------------------------------------
# Dataset → tensor conversion
# ---------------------------------------------------------------------------

def _resolve_n_channels(config: dict, summary: dict) -> int:
    """Determine the effective number of input channels.

    Priority: explicit config > auto-detect from column prefixes > 1.
    """
    user_ch = config.get("n_channels", 0)
    if user_ch > 0:
        return user_ch
    if summary.get("channel_detected"):
        return summary["n_channels"]
    return 1


def _reshape_for_channels(X: np.ndarray, n_channels: int) -> np.ndarray:
    """Reshape flat (N, total) array to (N, n_channels, samples_per_channel)."""
    N, total = X.shape
    if n_channels <= 1:
        return X.reshape(N, 1, total)
    if total % n_channels != 0:
        raise ValueError(
            f"Cannot split {total} signal columns into {n_channels} channels evenly. "
            f"{total} is not divisible by {n_channels}."
        )
    return X.reshape(N, n_channels, total // n_channels)


def _dataset_to_tensors(file_bytes: bytes, filename: str, summary: dict):
    """Convert raw file bytes into (X, y, class_names) numpy arrays.

    For CSV:  each row = 1 sample; signal columns = feature vector.
    For ZIP:  each file = 1 sample; first signal column = time series.

    When channel-prefixed columns are detected, columns are reordered so
    that all samples of ch1 come first, then ch2, etc.
    """
    fmt = summary["format"]
    class_names = summary["class_names"]
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    if fmt == "csv_labeled":
        text = file_bytes.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))

        label_col = summary.get("label_column", "label")

        # Reorder columns by channel if prefix detected
        channel_map = summary.get("channel_map", {})
        if channel_map:
            sig_cols = []
            for prefix in sorted(channel_map.keys()):
                sig_cols.extend(channel_map[prefix])
        else:
            sig_cols = summary["signal_columns"]

        X = df[sig_cols].values.astype(np.float32)
        y = df[label_col].astype(str).map(label_to_idx).values.astype(np.int64)
        return X, y, class_names

    elif fmt == "zip_folder":
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = zf.namelist()
            samples, labels = [], []

            for name in names:
                parts = Path(name).parts
                if any(p.startswith("__") or p.startswith(".") for p in parts):
                    continue
                if not name.lower().endswith(".csv") or len(parts) < 2:
                    continue

                class_name = parts[-2]
                if class_name not in label_to_idx:
                    continue

                with zf.open(name) as f:
                    sdf = pd.read_csv(f)

                time_cols = [c for c in sdf.columns if c.strip().lower() in _TIME_COL_NAMES]
                sig_cols = [c for c in sdf.columns if c not in time_cols]
                if not sig_cols:
                    continue

                # Use first signal column as the 1-D time series
                sig = sdf[sig_cols[0]].values.astype(np.float32)
                samples.append(sig)
                labels.append(label_to_idx[class_name])

            # Pad / truncate to uniform length
            lengths = [len(s) for s in samples]
            target_len = int(np.median(lengths))

            X_list = []
            for s in samples:
                if len(s) >= target_len:
                    X_list.append(s[:target_len])
                else:
                    X_list.append(np.pad(s, (0, target_len - len(s))))
            X = np.stack(X_list)
            y = np.array(labels, dtype=np.int64)
            return X, y, class_names

    raise ValueError(f"Unsupported dataset format: {fmt}")


# ---------------------------------------------------------------------------
# Training loop (runs in a worker thread)
# ---------------------------------------------------------------------------

def _run_training(job: TrainingJob, X: np.ndarray, y: np.ndarray):
    """Blocking training loop — call from a thread, not from async."""
    try:
        job.status = "training"

        cfg = job.config
        epochs = cfg.get("epochs", 30)
        lr = cfg.get("learning_rate", 1e-3)
        batch_size = cfg.get("batch_size", 64)
        val_split = cfg.get("val_split", 0.2)
        n_channels = job.n_channels

        n_samples = X.shape[0]
        n_classes = len(job.class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize per-sample (on flat array before reshape)
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std

        # Train / val split (stratified-ish)
        perm = np.random.permutation(n_samples)
        split = max(1, int(n_samples * (1 - val_split)))
        train_idx, val_idx = perm[:split], perm[split:]

        # Reshape to (N, C, L) — handles both single and multi-channel
        X_3d = _reshape_for_channels(X, n_channels)
        signal_len = X_3d.shape[2]

        X_train = torch.FloatTensor(X_3d[train_idx])
        y_train = torch.LongTensor(y[train_idx])
        X_val = torch.FloatTensor(X_3d[val_idx])
        y_val = torch.LongTensor(y[val_idx])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size * 2,
        )

        # Store val data (flat) for Phase 3 (confusion matrix / t-SNE)
        job.val_X = X[val_idx]
        job.val_y = y[val_idx]

        # Model
        model = Signal1DCNN(n_classes=n_classes, in_channels=n_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6,
        )

        best_val_acc = 0.0
        best_state = None

        job._emit({
            "type": "start",
            "total_epochs": epochs,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "signal_length": signal_len,
            "n_channels": n_channels,
            "n_classes": n_classes,
            "device": str(device),
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ---- train ----
            model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(yb)
                train_correct += (out.argmax(1) == yb).sum().item()
                train_total += len(yb)

            train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # ---- validate ----
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss_sum += loss.item() * len(yb)
                    val_correct += (out.argmax(1) == yb).sum().item()
                    val_total += len(yb)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            metrics = {
                "type": "epoch",
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_acc": round(val_acc, 5),
                "lr": current_lr,
                "elapsed_sec": round(elapsed, 2),
            }
            job.history.append(metrics)
            job._emit(metrics)

        # ---- done ----
        job.best_val_acc = best_val_acc
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        job.model = model.cpu()
        job.status = "completed"
        job._emit({
            "type": "complete",
            "best_val_acc": round(best_val_acc, 5),
            "total_epochs": epochs,
        })

    except Exception as exc:
        job.status = "failed"
        job.error = traceback.format_exc()
        job._emit({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Public manager (singleton)
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages all training jobs (in-memory)."""

    def __init__(self):
        self.jobs: dict[str, TrainingJob] = {}

    def start(
        self,
        job_id: str,
        file_bytes: bytes,
        filename: str,
        summary: dict,
        config: dict,
    ) -> TrainingJob:
        """Parse the dataset, create a job, and launch training in a thread."""
        X, y, class_names = _dataset_to_tensors(file_bytes, filename, summary)

        # Resolve effective n_channels and validate
        n_channels = _resolve_n_channels(config, summary)
        total_cols = X.shape[1]
        if n_channels > 1 and total_cols % n_channels != 0:
            raise ValueError(
                f"Cannot split {total_cols} signal columns into {n_channels} channels evenly. "
                f"{total_cols} is not divisible by {n_channels}."
            )

        job = TrainingJob(job_id=job_id, config=config, class_names=class_names)
        job.n_channels = n_channels
        self.jobs[job_id] = job

        import threading
        t = threading.Thread(
            target=_run_training, args=(job, X, y), daemon=True
        )
        t.start()
        return job

    def get(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)


training_manager = TrainingManager()


# ---------------------------------------------------------------------------
# Phase 3 — Post-training analysis helpers
# ---------------------------------------------------------------------------

def compute_confusion_matrix(job: TrainingJob) -> dict:
    """Run inference on val set and return confusion matrix + per-class metrics."""
    if job.model is None or job.val_X is None:
        raise ValueError("Model or validation data not available.")

    model = job.model
    model.eval()

    X_val = torch.FloatTensor(_reshape_for_channels(job.val_X, job.n_channels))
    with torch.no_grad():
        preds = model(X_val).argmax(dim=1).numpy()
    y_true = job.val_y

    n_classes = len(job.class_names)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, preds):
        matrix[int(t)][int(p)] += 1

    # Per-class precision / recall / f1
    per_class = []
    for i in range(n_classes):
        tp = int(matrix[i][i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({
            "class": job.class_names[i],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(matrix[i, :].sum()),
        })

    return {
        "matrix": matrix.tolist(),
        "class_names": job.class_names,
        "per_class": per_class,
        "accuracy": round(float(np.trace(matrix)) / max(float(matrix.sum()), 1), 4),
    }


def compute_tsne(job: TrainingJob, perplexity: float = 30.0) -> dict:
    """Extract 128-d features from the penultimate layer, reduce to 2-D via t-SNE."""
    if job.model is None or job.val_X is None:
        raise ValueError("Model or validation data not available.")

    model = job.model
    model.eval()

    X_val = torch.FloatTensor(_reshape_for_channels(job.val_X, job.n_channels))
    with torch.no_grad():
        feats = model.extract_features(X_val).numpy()  # (N, 128)

    from sklearn.manifold import TSNE

    # Clamp perplexity to valid range
    n_samples = feats.shape[0]
    perplexity = min(perplexity, max(2.0, (n_samples - 1) / 3.0))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=800)
    coords = tsne.fit_transform(feats)  # (N, 2)

    return {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "labels": [job.class_names[int(i)] for i in job.val_y],
        "label_indices": job.val_y.tolist(),
        "class_names": job.class_names,
    }
