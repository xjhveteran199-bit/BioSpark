import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = Path(__file__).resolve().parent / "models"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

UPLOAD_DIR.mkdir(exist_ok=True)

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Upload limits
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = {".csv", ".edf", ".mat", ".txt"}

# Signal types
SIGNAL_TYPES = ["ecg", "eeg", "emg"]

# Preprocessing defaults
PREPROCESS_CONFIG = {
    "ecg": {
        "sampling_rate": 360,  # MIT-BIH default
        "lowcut": 0.5,
        "highcut": 40.0,
    },
    "eeg": {
        "sampling_rate": 256,
        "lowcut": 0.5,
        "highcut": 45.0,
    },
    "emg": {
        "sampling_rate": 1000,
        "lowcut": 20.0,
        "highcut": 450.0,
    },
}

# Model registry
MODEL_REGISTRY = {
    "ecg_arrhythmia": {
        "file": "ecg_arrhythmia_cnn.onnx",
        "signal_type": "ecg",
        "description": "ECG Arrhythmia Detection (5-class)",
        "classes": ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown (Q)"],
        "input_length": 187,
        "sampling_rate": 360,
    },
    "eeg_sleep": {
        "file": "eeg_sleep_staging.onnx",
        "signal_type": "eeg",
        "description": "EEG Sleep Staging (5-class)",
        "classes": ["Wake (W)", "N1", "N2", "N3", "REM"],
        "input_length": 3000,
        "sampling_rate": 100,
    },
    "emg_gesture": {
        "file": "emg_gesture_cnn.onnx",
        "signal_type": "emg",
        "description": "EMG Gesture Recognition",
        "classes": [f"Gesture {i}" for i in range(1, 53)],
        "input_length": 400,
        "sampling_rate": 1000,
    },
}
