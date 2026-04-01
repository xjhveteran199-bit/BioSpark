"""Train a 1D-CNN for ECG arrhythmia detection on MIT-BIH dataset.

Downloads data from PhysioNet, trains a lightweight CNN, saves model weights.

Usage:
    python training/train_ecg_arrhythmia.py

Output:
    backend/models/ecg_arrhythmia_cnn.pt  (PyTorch model)
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

MODEL_OUTPUT = PROJECT_ROOT / "backend" / "models" / "ecg_arrhythmia_cnn.pt"
DATA_DIR = PROJECT_ROOT / "training" / "data" / "mitbih"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# MIT-BIH record numbers
MIT_BIH_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

# AAMI classes mapping (beat annotation → 5-class)
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,        # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                   # Supraventricular
    'V': 2, 'E': 2,                                     # Ventricular
    'F': 3,                                              # Fusion
    '/': 4, 'f': 4, 'Q': 4,                             # Unknown/Paced
}
CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

SEG_LEN = 187  # Samples per heartbeat segment
HALF_SEG = SEG_LEN // 2


class ECGArrhythmiaCNN(nn.Module):
    """Lightweight 1D-CNN for 5-class ECG arrhythmia detection.

    Architecture: 3 conv blocks + global avg pool + FC
    ~85K parameters
    """
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
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
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, seg_len)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 128)
        x = self.classifier(x)
        return x


def download_mitbih():
    """Download MIT-BIH records from PhysioNet."""
    import wfdb

    print("Downloading MIT-BIH Arrhythmia Database...")
    downloaded = 0
    for rec_num in MIT_BIH_RECORDS:
        rec_name = str(rec_num)
        rec_path = DATA_DIR / rec_name
        if (DATA_DIR / f"{rec_name}.dat").exists():
            continue
        try:
            wfdb.dl_database('mitdb', str(DATA_DIR), records=[rec_name])
            downloaded += 1
        except Exception as e:
            print(f"  Warning: could not download record {rec_name}: {e}")

    if downloaded > 0:
        print(f"  Downloaded {downloaded} new records")
    else:
        print("  All records already cached")


def load_and_segment():
    """Load MIT-BIH records and extract heartbeat segments."""
    import wfdb

    all_segments = []
    all_labels = []

    print("Loading and segmenting records...")
    for rec_num in MIT_BIH_RECORDS:
        rec_name = str(rec_num)
        rec_path = str(DATA_DIR / rec_name)

        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            print(f"  Skipping {rec_name}: {e}")
            continue

        # Use first channel (usually MLII)
        signal = record.p_signal[:, 0]

        # Normalize per-record
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        # Extract segments around each beat annotation
        for idx, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
            if symbol not in AAMI_MAP:
                continue

            label = AAMI_MAP[symbol]
            start = sample - HALF_SEG
            end = start + SEG_LEN

            if start < 0 or end > len(signal):
                continue

            segment = signal[start:end]
            all_segments.append(segment)
            all_labels.append(label)

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Print class distribution
    print(f"  Total segments: {len(labels)}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"    {name}: {count} ({100*count/len(labels):.1f}%)")

    return segments, labels


def balance_classes(segments, labels, max_per_class=5000):
    """Balance dataset by undersampling majority classes."""
    balanced_segs = []
    balanced_labels = []

    for cls in range(5):
        cls_mask = labels == cls
        cls_segs = segments[cls_mask]
        if len(cls_segs) > max_per_class:
            indices = np.random.choice(len(cls_segs), max_per_class, replace=False)
            cls_segs = cls_segs[indices]
        elif len(cls_segs) > 0 and len(cls_segs) < max_per_class:
            # Oversample minority classes (with noise augmentation)
            n_needed = min(max_per_class, len(cls_segs) * 3)
            indices = np.random.choice(len(cls_segs), n_needed, replace=True)
            cls_segs = cls_segs[indices]
            # Add small noise for augmentation
            cls_segs = cls_segs + np.random.normal(0, 0.02, cls_segs.shape).astype(np.float32)

        balanced_segs.append(cls_segs)
        balanced_labels.extend([cls] * len(cls_segs))

    segments = np.concatenate(balanced_segs)
    labels = np.array(balanced_labels, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(labels))
    return segments[perm], labels[perm]


def train():
    """Full training pipeline."""
    # Step 1: Download data
    download_mitbih()

    # Step 2: Load and segment
    segments, labels = load_and_segment()

    if len(segments) == 0:
        print("ERROR: No segments extracted. Check data download.")
        return

    # Step 3: Balance classes
    segments, labels = balance_classes(segments, labels)
    print(f"\nAfter balancing: {len(labels)} segments")

    # Step 4: Train/test split (80/20)
    n = len(labels)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train = torch.FloatTensor(segments[train_idx]).unsqueeze(1)  # (N, 1, 187)
    y_train = torch.LongTensor(labels[train_idx])
    X_test = torch.FloatTensor(segments[test_idx]).unsqueeze(1)
    y_test = torch.LongTensor(labels[test_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    # Step 5: Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")

    model = ECGArrhythmiaCNN(n_classes=5).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=5).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 5
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0
    epochs = 30

    for epoch in range(epochs):
        # Train
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
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_acc = correct / total

        # Evaluate
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
        avg_loss = train_loss / total
        scheduler.step(avg_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {lr:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'input_length': SEG_LEN,
                'n_classes': 5,
                'test_accuracy': test_acc,
                'architecture': 'ECGArrhythmiaCNN',
            }, str(MODEL_OUTPUT))

    print(f"\nTraining complete! Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    train()
