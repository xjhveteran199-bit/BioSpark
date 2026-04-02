"""List available models endpoint."""

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

def GET(req):
    """GET /api/models - List available models."""
    import json
    return {
        "statusCode": 200,
        "body": json.dumps({
            "models": [
                {
                    "id": model_id,
                    "signal_type": info["signal_type"],
                    "description": info["description"],
                    "classes": info["classes"],
                    "input_length": info["input_length"],
                }
                for model_id, info in MODEL_REGISTRY.items()
            ]
        }),
        "headers": {"Content-Type": "application/json"}
    }
