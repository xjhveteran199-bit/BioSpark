"""Train a 1D-CNN for EEG Sleep Staging on Sleep-EDF dataset.

Downloads data from PhysioNet via MNE, trains a lightweight CNN for 5-class
sleep staging (Wake / N1 / N2 / N3 / REM).

Usage:
    python training/train_eeg_sleep.py

Output:
    backend/models/eeg_sleep_staging.pt  (PyTorch model)

Dataset:
    Sleep-EDF Expanded (PhysioNet)
    - 30-second epochs @ 100 Hz = 3000 samples per epoch
    - Channel: Fpz-Cz (single EEG channel)
    - 5 AASM sleep stages
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_OUTPUT = PROJECT_ROOT / "backend" / "models" / "eeg_sleep_staging.pt"
DATA_DIR = PROJECT_ROOT / "training" / "data" / "sleep_edf"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# AASM sleep stage classes
CLASS_NAMES = ["Wake (W)", "N1", "N2", "N3", "REM"]
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # N3 = old stage 3 + stage 4
    "Sleep stage R": 4,
}

EPOCH_DURATION = 30   # seconds
TARGET_SR = 100        # Hz (resample to this)
SEG_LEN = EPOCH_DURATION * TARGET_SR  # 3000 samples per epoch
N_CLASSES = 5

# Use first 20 subjects (40 nights) from Sleep-EDF Expanded
N_SUBJECTS = 20


class EEGSleepCNN(nn.Module):
    """1D-CNN for 5-class EEG sleep staging.

    Deeper architecture than ECG model due to longer input (3000 samples).
    4 conv blocks + global avg pool + FC head.
    ~150K parameters.
    """
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: capture low-frequency patterns
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),   # 3000 → 750

            # Block 2
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),   # 750 → 187

            # Block 3
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),   # 187 → 46

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 3000)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract penultimate layer features for t-SNE."""
        x = self.features(x)
        x = x.squeeze(-1)
        # Run through classifier up to last dropout before final linear
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear(256→128)
        x = self.classifier[2](x)  # ReLU
        return x  # 128-dim features


def download_sleep_edf():
    """Download Sleep-EDF data from PhysioNet using MNE."""
    print("=" * 60)
    print("Downloading Sleep-EDF Expanded dataset via MNE...")
    print("=" * 60)

    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        print("ERROR: MNE not installed. Run: pip install mne")
        print("  Then retry: python training/train_eeg_sleep.py")
        sys.exit(1)

    # Check if we already have cached numpy data
    cache_file = DATA_DIR / "sleep_edf_cache.npz"
    if cache_file.exists():
        print("  Found cached data, loading...")
        data = np.load(str(cache_file), allow_pickle=True)
        return data["segments"], data["labels"]

    from mne.datasets import sleep_physionet

    all_segments = []
    all_labels = []

    # Download and process each subject
    subjects = list(range(N_SUBJECTS))

    for subj_idx, subject in enumerate(subjects):
        print(f"\n  Processing subject {subject} ({subj_idx + 1}/{len(subjects)})...")

        try:
            # Download PSG and hypnogram files
            # Each subject has 2 nights (recording=0 and recording=1)
            for night in [0, 1]:
                try:
                    raw_files = sleep_physionet.age.fetch_data(
                        subjects=[subject], recording=[night],
                        path=str(DATA_DIR / "mne_data"),
                        on_missing="warn",
                    )
                except Exception as e:
                    print(f"    Night {night} not available: {e}")
                    continue

                if not raw_files:
                    continue

                psg_file, hyp_file = raw_files[0]

                # Load raw EEG
                raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)

                # Get annotations (sleep stages)
                annot = mne.read_annotations(hyp_file)
                raw.set_annotations(annot, verbose=False)

                # Create events from annotations
                event_id = {k: v for k, v in STAGE_MAP.items()
                            if k in [a["description"] for a in annot]}

                if not event_id:
                    print(f"    No valid sleep annotations found, skipping")
                    continue

                events, _ = mne.events_from_annotations(
                    raw, event_id=event_id, chunk_duration=EPOCH_DURATION,
                    verbose=False
                )

                # Pick EEG channel (Fpz-Cz preferred)
                eeg_channels = [ch for ch in raw.ch_names
                                if any(name in ch.lower() for name in
                                       ["fpz", "eeg fpz", "eeg"])]
                if eeg_channels:
                    raw.pick(eeg_channels[0])
                else:
                    # Fallback: pick first channel
                    raw.pick([raw.ch_names[0]])

                # Resample to target rate
                if raw.info["sfreq"] != TARGET_SR:
                    raw.resample(TARGET_SR, verbose=False)

                # Extract 30-second epochs
                data = raw.get_data()[0]  # (n_samples,)

                for event in events:
                    sample_idx = event[0]
                    label = event[2]

                    if label not in range(N_CLASSES):
                        continue

                    start = sample_idx
                    end = start + SEG_LEN

                    if start < 0 or end > len(data):
                        continue

                    segment = data[start:end].astype(np.float32)

                    # Normalize per-epoch
                    std = np.std(segment)
                    if std > 1e-10:
                        segment = (segment - np.mean(segment)) / std
                    else:
                        continue  # Skip flat segments

                    all_segments.append(segment)
                    all_labels.append(label)

        except Exception as e:
            print(f"    Error processing subject {subject}: {e}")
            continue

    if len(all_segments) == 0:
        print("\nERROR: No segments extracted. Check MNE installation and internet.")
        sys.exit(1)

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Cache for future runs
    np.savez_compressed(str(cache_file), segments=segments, labels=labels)
    print(f"\n  Cached {len(labels)} epochs to {cache_file}")

    return segments, labels


def balance_classes(segments, labels, max_per_class=8000):
    """Balance dataset by undersampling majority + oversampling minority."""
    balanced_segs = []
    balanced_labels = []

    for cls in range(N_CLASSES):
        cls_mask = labels == cls
        cls_segs = segments[cls_mask]
        n_cls = len(cls_segs)

        if n_cls == 0:
            print(f"  WARNING: No samples for class {CLASS_NAMES[cls]}")
            continue

        if n_cls > max_per_class:
            # Undersample
            indices = np.random.choice(n_cls, max_per_class, replace=False)
            cls_segs = cls_segs[indices]
        elif n_cls < max_per_class:
            # Oversample with noise augmentation
            target = min(max_per_class, n_cls * 3)
            indices = np.random.choice(n_cls, target, replace=True)
            cls_segs = cls_segs[indices]
            # Add small Gaussian noise for augmentation
            noise = np.random.normal(0, 0.05, cls_segs.shape).astype(np.float32)
            cls_segs = cls_segs + noise

        balanced_segs.append(cls_segs)
        balanced_labels.extend([cls] * len(cls_segs))

    segments = np.concatenate(balanced_segs)
    labels = np.array(balanced_labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(labels))
    return segments[perm], labels[perm]


def train():
    """Full EEG sleep staging training pipeline."""
    print("=" * 60)
    print("  BioSpark — EEG Sleep Staging Model Training")
    print("=" * 60)
    print(f"  Model:    EEGSleepCNN (4 conv blocks)")
    print(f"  Dataset:  Sleep-EDF Expanded (PhysioNet)")
    print(f"  Classes:  {N_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"  Input:    {SEG_LEN} samples (30s @ {TARGET_SR}Hz)")
    print(f"  Output:   {MODEL_OUTPUT}")
    print()

    # Step 1: Download and load data
    segments, labels = download_sleep_edf()

    if len(segments) == 0:
        print("ERROR: No segments extracted.")
        return

    # Print raw class distribution
    print(f"\n  Raw dataset: {len(labels)} epochs")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        pct = 100 * count / len(labels) if len(labels) > 0 else 0
        print(f"    {name}: {count} ({pct:.1f}%)")

    # Step 2: Balance classes
    segments, labels = balance_classes(segments, labels)
    print(f"\n  After balancing: {len(labels)} epochs")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"    {name}: {count}")

    # Step 3: Train/test split (80/20)
    n = len(labels)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train = torch.FloatTensor(segments[train_idx]).unsqueeze(1)  # (N, 1, 3000)
    y_train = torch.LongTensor(labels[train_idx])
    X_test = torch.FloatTensor(segments[test_idx]).unsqueeze(1)
    y_test = torch.LongTensor(labels[test_idx])

    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=128
    )

    # Step 4: Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training on: {device}")

    model = EEGSleepCNN(n_classes=N_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Class weights
    class_counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    # Step 5: Training loop
    best_acc = 0
    epochs = 50
    patience_counter = 0
    early_stop_patience = 15

    print(f"\n  Starting training ({epochs} epochs max, early stop patience={early_stop_patience})...")
    print("-" * 85)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_acc = correct / total
        avg_loss = train_loss / total

        # --- Evaluate ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                test_correct += (out.argmax(1) == y_batch).sum().item()
                test_total += len(y_batch)

        test_acc = test_correct / test_total
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            marker = " *BEST*"

            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "input_length": SEG_LEN,
                "n_classes": N_CLASSES,
                "test_accuracy": test_acc,
                "architecture": "EEGSleepCNN",
                "sampling_rate": TARGET_SR,
                "epoch_duration_sec": EPOCH_DURATION,
            }, str(MODEL_OUTPUT))
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"LR: {lr:.6f}{marker}"
        )

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
            break

    print("-" * 85)
    print(f"\n  Training complete!")
    print(f"  Best test accuracy: {best_acc:.4f}")
    print(f"  Model saved to: {MODEL_OUTPUT}")

    # Per-class accuracy on test set
    print(f"\n  Per-class test accuracy:")
    model.load_state_dict(torch.load(str(MODEL_OUTPUT), weights_only=False)["model_state_dict"])
    model.eval()
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            preds = out.argmax(1)
            for i in range(len(y_batch)):
                label = y_batch[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"    {name}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"    {name}: N/A (no test samples)")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    train()
