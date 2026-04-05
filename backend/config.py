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
        "file": "eeg_sleep_staging.pt",
        "signal_type": "eeg",
        "description": "EEG Sleep Staging (5-class AASM)",
        "classes": ["Wake (W)", "N1", "N2", "N3", "REM"],
        "input_length": 3000,
        "sampling_rate": 100,
    },
    "emg_gesture": {
        "file": "emg_gesture_cnn.pt",
        "signal_type": "emg",
        "description": "EMG Gesture Recognition (53-class, NinaPro DB5)",
        "classes": [
            # 0: Rest
            "Rest",
            # E1: 1-12 Basic finger movements
            "Index flexion", "Index extension",
            "Middle flexion", "Middle extension",
            "Ring flexion", "Ring extension",
            "Little finger flexion", "Little finger extension",
            "Thumb adduction", "Thumb abduction",
            "Thumb flexion", "Thumb extension",
            # E2: 13-29 Hand/wrist movements
            "Thumb up", "Index+Middle extension", "Ring+Little flexion",
            "Thumb opposing little finger", "Abduction of all fingers",
            "Fingers flexed into fist", "Pointing index",
            "Adduction of extended fingers", "Wrist supination (mid finger)",
            "Wrist pronation (mid finger)", "Wrist supination (little finger)",
            "Wrist pronation (little finger)", "Wrist flexion",
            "Wrist extension", "Wrist radial deviation",
            "Wrist ulnar deviation", "Wrist extension with closed hand",
            # E3: 30-52 Grasping/functional movements
            "Large diameter grasp", "Small diameter grasp",
            "Fixed hook grasp", "Index finger extension grasp",
            "Medium wrap", "Ring grasp",
            "Prismatic four finger grasp", "Stick grasp",
            "Writing tripod grasp", "Power sphere grasp",
            "Three finger sphere grasp", "Precision sphere grasp",
            "Tripod grasp", "Prismatic pinch grasp",
            "Tip pinch grasp", "Quadpod grasp",
            "Lateral grasp", "Parallel extension grasp",
            "Extension type grasp", "Power disk grasp",
            "Open bottle with tripod", "Turn a screw",
            "Cut something",
        ],
        "input_length": 80,
        "in_channels": 16,
        "sampling_rate": 200,
    },
}
