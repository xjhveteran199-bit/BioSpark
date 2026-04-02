"""Export ECGArrhythmiaCNN to ONNX format for browser inference.

Run: python training/export_onnx.py

Output: backend/models/ecg_arrhythmia_cnn.onnx
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


class ECGArrhythmiaCNN(torch.nn.Module):
    """Lightweight 1D-CNN for 5-class ECG arrhythmia detection.

    Architecture: 3 conv blocks + global avg pool + FC
    ~85K parameters
    """
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv1d(1, 32, kernel_size=7, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # Block 2
            torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            # Block 3
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, seg_len)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 128)
        x = self.classifier(x)
        return x


def export_onnx():
    MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "ecg_arrhythmia_cnn.pt"
    OUTPUT_PATH = PROJECT_ROOT / "backend" / "models" / "ecg_arrhythmia_cnn.onnx"
    FRONTEND_MODEL_PATH = PROJECT_ROOT / "frontend" / "models" / "ecg_arrhythmia_cnn.onnx"

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("  Run 'python training/train_ecg_arrhythmia.py' first to train the model.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)

    # Create model and load weights
    model = ECGArrhythmiaCNN(n_classes=5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Architecture: {checkpoint.get('architecture', 'Unknown')}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 187)

    # Export to ONNX
    print(f"\nExporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print("  ONNX export successful!")

    # Copy to frontend for Vercel deployment
    import shutil
    FRONTEND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(OUTPUT_PATH), str(FRONTEND_MODEL_PATH))
    print(f"  Copied to frontend: {FRONTEND_MODEL_PATH}")

    # Verify the exported model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid!")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(OUTPUT_PATH))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Test with random data
        test_input = np.random.randn(1, 1, 187).astype(np.float32)
        output = session.run([output_name], {input_name: test_input})[0]
        print(f"  ONNX Runtime inference test passed! Output shape: {output.shape}")
        print(f"  Output: {output}")
    except ImportError:
        print("  (ONNX Runtime not installed, skipping runtime test)")
        print("  Install with: pip install onnxruntime")


if __name__ == "__main__":
    export_onnx()
