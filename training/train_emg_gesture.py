"""Train a 1D-CNN for EMG Gesture Recognition on NinaPro DB5 dataset.

Uses REAL NinaPro DB5 data from:
    training/EMG testing data/s{1..10}/S{N}_E{1,2,3}_A1.mat

Trains a multi-channel CNN on all 3 exercises:
    E1: 12 basic finger movements
    E2: 17 hand/wrist gestures
    E3: 23 grasping/functional movements
    Total: 52 gestures + Rest = 53 classes

Usage:
    python training/train_emg_gesture.py

Output:
    backend/models/emg_gesture_cnn.pt  (PyTorch model)

References:
    - Pizzolato et al. (2017) "Comparison of six electromyography acquisition setups"
    - https://ninapro.hevs.ch/
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional
from scipy.io import loadmat

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_OUTPUT = PROJECT_ROOT / "backend" / "models" / "emg_gesture_cnn.pt"
DATA_DIR = PROJECT_ROOT / "training" / "EMG testing data"
CACHE_DIR = PROJECT_ROOT / "training" / "data" / "ninapro"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# NinaPro DB5 — Full gesture set across all 3 exercises
# ------------------------------------------------------------------
# Exercise 1: 12 basic finger movements (labels 1-12)
E1_GESTURES = [
    "Index flexion",
    "Index extension",
    "Middle flexion",
    "Middle extension",
    "Ring flexion",
    "Ring extension",
    "Little finger flexion",
    "Little finger extension",
    "Thumb adduction",
    "Thumb abduction",
    "Thumb flexion",
    "Thumb extension",
]

# Exercise 2: 17 hand/wrist movements (labels 1-17)
E2_GESTURES = [
    "Thumb up",
    "Index+Middle extension",
    "Ring+Little flexion",
    "Thumb opposing little finger",
    "Abduction of all fingers",
    "Fingers flexed into fist",
    "Pointing index",
    "Adduction of extended fingers",
    "Wrist supination (mid finger)",
    "Wrist pronation (mid finger)",
    "Wrist supination (little finger)",
    "Wrist pronation (little finger)",
    "Wrist flexion",
    "Wrist extension",
    "Wrist radial deviation",
    "Wrist ulnar deviation",
    "Wrist extension with closed hand",
]

# Exercise 3: 23 grasping/functional movements (labels 1-23)
E3_GESTURES = [
    "Large diameter grasp",
    "Small diameter grasp",
    "Fixed hook grasp",
    "Index finger extension grasp",
    "Medium wrap",
    "Ring grasp",
    "Prismatic four finger grasp",
    "Stick grasp",
    "Writing tripod grasp",
    "Power sphere grasp",
    "Three finger sphere grasp",
    "Precision sphere grasp",
    "Tripod grasp",
    "Prismatic pinch grasp",
    "Tip pinch grasp",
    "Quadpod grasp",
    "Lateral grasp",
    "Parallel extension grasp",
    "Extension type grasp",
    "Power disk grasp",
    "Open a bottle with a tripod grasp",
    "Turn a screw",
    "Cut something",
]

# Build unified label map: 0=Rest, 1-12=E1, 13-29=E2, 30-52=E3
GESTURE_NAMES = ["Rest"] + E1_GESTURES + E2_GESTURES + E3_GESTURES
N_CLASSES = len(GESTURE_NAMES)   # 53
N_CHANNELS = 16                  # sEMG channels in DB5
SAMPLING_RATE = 200              # Hz
WINDOW_MS = 400                  # milliseconds per segment
SEG_LEN = int(SAMPLING_RATE * WINDOW_MS / 1000)  # 80 samples
OVERLAP = 0.5                    # 50% window overlap
N_SUBJECTS = 10

# Label offset per exercise (to create unified label space)
EXERCISE_LABEL_OFFSET = {
    1: 0,    # E1 labels 1-12 → unified 1-12
    2: 12,   # E2 labels 1-17 → unified 13-29
    3: 29,   # E3 labels 1-23 → unified 30-52
}


class EMGGestureCNN(nn.Module):
    """Multi-channel 1D-CNN for EMG gesture recognition.

    Takes multi-channel sEMG input and classifies into gesture classes.
    4 conv blocks with increasing depth + global avg pool + FC head.
    Designed for short windows (80 samples) with 16 channels.
    """
    def __init__(self, n_classes=53, in_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.features = nn.Sequential(
            # Block 1: capture per-channel activation patterns
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),     # 80 -> 40

            # Block 2: cross-channel feature fusion
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),     # 40 -> 20

            # Block 3: higher-level temporal patterns
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),     # 20 -> 10

            # Block 4: abstract gesture features
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, in_channels, seg_len)  e.g. (batch, 16, 80)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 256)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract penultimate-layer features (128-d) for t-SNE."""
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear(256->128)
        x = self.classifier[2](x)  # ReLU
        return x  # 128-dim features


# ------------------------------------------------------------------
# Data loading — NinaPro DB5 (real data)
# ------------------------------------------------------------------

def load_ninapro_db5() -> Tuple[np.ndarray, np.ndarray]:
    """Load all NinaPro DB5 data from training/EMG testing data/.

    Loads E1, E2, E3 for all 10 subjects, segments into 400ms windows,
    and maps labels to a unified 53-class label space.

    Returns:
        segments: (N, 16, 80) float32 array
        labels:   (N,) int64 array with values in [0, 52]
    """
    # Check cache first
    cache_file = CACHE_DIR / "ninapro_db5_full_cache.npz"
    if cache_file.exists():
        print("  Found cached data, loading...")
        data = np.load(str(cache_file))
        return data["segments"], data["labels"]

    all_segments = []
    all_labels = []
    step = int(SEG_LEN * (1 - OVERLAP))  # 40 samples step

    for subj in range(1, N_SUBJECTS + 1):
        subj_dir = DATA_DIR / f"s{subj}"
        if not subj_dir.exists():
            print(f"  WARNING: Subject {subj} directory not found, skipping")
            continue

        subj_segs = 0

        for exercise in [1, 2, 3]:
            mat_file = subj_dir / f"S{subj}_E{exercise}_A1.mat"
            if not mat_file.exists():
                print(f"    WARNING: {mat_file.name} not found, skipping")
                continue

            data = loadmat(str(mat_file))
            emg = data["emg"].astype(np.float32)              # (n_samples, 16)
            labels = data["restimulus"].flatten().astype(int)   # (n_samples,)

            label_offset = EXERCISE_LABEL_OFFSET[exercise]

            # Segment into 400ms windows with 50% overlap
            for start in range(0, len(emg) - SEG_LEN, step):
                end = start + SEG_LEN
                window_labels = labels[start:end]

                # Only use windows with a single consistent label
                unique_labels = np.unique(window_labels)
                if len(unique_labels) != 1:
                    continue

                orig_label = unique_labels[0]

                # Map to unified label space
                if orig_label == 0:
                    unified_label = 0  # Rest is always 0
                else:
                    unified_label = orig_label + label_offset

                if unified_label < 0 or unified_label >= N_CLASSES:
                    continue

                segment = emg[start:end].T  # (16, 80) — channels first

                # Per-channel z-score normalization
                for ch in range(segment.shape[0]):
                    std = np.std(segment[ch])
                    if std > 1e-10:
                        segment[ch] = (segment[ch] - np.mean(segment[ch])) / std
                    else:
                        segment[ch] = 0.0  # flat channel

                all_segments.append(segment)
                all_labels.append(unified_label)
                subj_segs += 1

        print(f"  Subject {subj:2d}: {subj_segs:6d} segments extracted")

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Cache for fast re-runs
    print(f"\n  Caching {len(labels)} segments to {cache_file}")
    np.savez_compressed(str(cache_file), segments=segments, labels=labels)

    return segments, labels


def balance_classes(segments, labels, max_per_class=3000, min_per_class=500):
    """Balance dataset: undersample majority, oversample minority with noise augmentation."""
    balanced_segs = []
    balanced_labels = []

    for cls in range(N_CLASSES):
        cls_mask = labels == cls
        cls_segs = segments[cls_mask]
        n_cls = len(cls_segs)

        if n_cls == 0:
            print(f"    WARNING: No samples for class {cls} ({GESTURE_NAMES[cls]})")
            continue

        if n_cls > max_per_class:
            # Undersample
            indices = np.random.choice(n_cls, max_per_class, replace=False)
            cls_segs = cls_segs[indices]
        elif n_cls < min_per_class:
            # Oversample with noise augmentation
            target = min(min_per_class, n_cls * 5)
            indices = np.random.choice(n_cls, target, replace=True)
            cls_segs = cls_segs[indices]
            # Add small noise for augmentation (preserve signal characteristics)
            noise = np.random.normal(0, 0.03, cls_segs.shape).astype(np.float32)
            cls_segs = cls_segs + noise

        balanced_segs.append(cls_segs)
        balanced_labels.extend([cls] * len(cls_segs))

    segments = np.concatenate(balanced_segs)
    labels = np.array(balanced_labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(labels))
    return segments[perm], labels[perm]


def train():
    """Full EMG gesture recognition training pipeline using NinaPro DB5."""
    print("=" * 70)
    print("  BioSpark  --  EMG Gesture Recognition Model Training")
    print("  Dataset:  NinaPro DB5 (REAL data, 10 subjects)")
    print("=" * 70)
    print(f"  Model:      EMGGestureCNN (4 conv blocks, multi-channel)")
    print(f"  Exercises:  E1 (12 finger) + E2 (17 hand/wrist) + E3 (23 grasp)")
    print(f"  Classes:    {N_CLASSES} (52 gestures + Rest)")
    print(f"  Input:      ({N_CHANNELS}, {SEG_LEN}) = 16ch x {WINDOW_MS}ms @ {SAMPLING_RATE}Hz")
    print(f"  Output:     {MODEL_OUTPUT}")
    print(f"  Data dir:   {DATA_DIR}")
    print()

    # Step 1: Load data
    segments, labels = load_ninapro_db5()

    if len(segments) == 0:
        print("ERROR: No data loaded. Check that .mat files exist in:")
        print(f"  {DATA_DIR}/s{{1..10}}/S{{N}}_E{{1,2,3}}_A1.mat")
        return

    # Print class distribution
    print(f"\n  Raw dataset: {len(labels)} segments")
    print(f"  Class distribution:")
    for i, name in enumerate(GESTURE_NAMES):
        count = np.sum(labels == i)
        if count > 0:
            pct = 100 * count / len(labels)
            print(f"    [{i:2d}] {name}: {count} ({pct:.1f}%)")

    # Step 2: Balance classes
    print(f"\n  Balancing classes...")
    segments, labels = balance_classes(segments, labels)
    n_actual_classes = len(np.unique(labels))
    print(f"  After balancing: {len(labels)} segments, {n_actual_classes} active classes")

    # Step 3: Train/test split (80/20)
    n = len(labels)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train = torch.FloatTensor(segments[train_idx])   # (N, 16, 80)
    y_train = torch.LongTensor(labels[train_idx])
    X_test = torch.FloatTensor(segments[test_idx])
    y_test = torch.LongTensor(labels[test_idx])

    print(f"\n  Train: {len(y_train)}, Test: {len(y_test)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=512
    )

    # Step 4: Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training on: {device}")

    model = EMGGestureCNN(n_classes=N_CLASSES, in_channels=N_CHANNELS).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = class_weights / (class_weights.sum() + 1e-10) * n_actual_classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )

    # Step 5: Training loop
    best_acc = 0
    epochs = 80
    patience_counter = 0
    early_stop_patience = 15

    print(f"\n  Starting training ({epochs} epochs max, early stop patience={early_stop_patience})...")
    print("-" * 90)

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

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": GESTURE_NAMES,
                "input_length": SEG_LEN,
                "n_classes": N_CLASSES,
                "in_channels": N_CHANNELS,
                "test_accuracy": test_acc,
                "architecture": "EMGGestureCNN",
                "sampling_rate": SAMPLING_RATE,
                "window_ms": WINDOW_MS,
                "exercises": ["E1_finger", "E2_hand_wrist", "E3_grasp"],
                "dataset": "NinaPro DB5",
                "n_subjects": N_SUBJECTS,
            }, str(MODEL_OUTPUT))
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.4f} | "
            f"Test: {test_acc:.4f} | "
            f"LR: {lr:.6f}{marker}"
        )

        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs)")
            break

    print("-" * 90)
    print(f"\n  Training complete!")
    print(f"  Best test accuracy: {best_acc:.4f}")
    print(f"  Model saved to: {MODEL_OUTPUT}")

    # ---- Per-class accuracy report ----
    print(f"\n  Loading best model for evaluation...")
    checkpoint = torch.load(str(MODEL_OUTPUT), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    # Print all classes grouped by exercise
    print(f"\n  === Per-class Test Accuracy ===")

    def print_exercise(name, start_idx, end_idx):
        print(f"\n  --- {name} ---")
        ex_correct = 0
        ex_total = 0
        for i in range(start_idx, end_idx):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"    [{i:2d}] {GESTURE_NAMES[i]:<42s} {acc:.4f} ({class_correct[i]}/{class_total[i]})")
                ex_correct += class_correct[i]
                ex_total += class_total[i]
        if ex_total > 0:
            print(f"    {'':42s} Avg: {ex_correct/ex_total:.4f}")

    # Rest
    if class_total[0] > 0:
        print(f"\n    [ 0] {'Rest':<42s} {class_correct[0]/class_total[0]:.4f} ({class_correct[0]}/{class_total[0]})")

    print_exercise("Exercise 1: Basic Finger Movements", 1, 13)
    print_exercise("Exercise 2: Hand/Wrist Movements", 13, 30)
    print_exercise("Exercise 3: Grasping Movements", 30, 53)

    overall = sum(class_correct) / max(sum(class_total), 1)
    print(f"\n  === Overall Test Accuracy: {overall:.4f} ===")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    train()
