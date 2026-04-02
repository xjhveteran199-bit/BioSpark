"""
BioSpark API - Vercel Serverless Functions
Minimal ASGI app approach
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "platform": "vercel"}


@app.get("/api/models")
async def list_models():
    return {
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
    }


@app.post("/api/upload")
async def upload(request: Request):
    """Combined upload + analyze endpoint."""
    try:
        form = await request.form()
        file = form.get("file")
        model_id = form.get("model_id", "ecg_arrhythmia")
        
        if not file:
            return Response(content=json.dumps({"error": "No file provided"}), status_code=400, media_type="application/json")
        
        content = await file.read()
        filename = file.filename
        
        # Parse file
        ext = filename.split('.')[-1].lower() if filename else "csv"
        
        if ext in ("csv", "txt"):
            text = content.decode("utf-8", errors="ignore")
            parsed = parse_csv(text)
        elif ext == "mat":
            parsed = parse_mat(content)
        else:
            return Response(content=json.dumps({"error": f"Unsupported format: {ext}"}), status_code=400, media_type="application/json")
        
        result = run_analysis(parsed, model_id)
        result["filename"] = filename
        
        return result
    
    except Exception as e:
        import traceback
        return Response(
            content=json.dumps({"error": str(e), "trace": traceback.format_exc()}),
            status_code=500,
            media_type="application/json"
        )


# --- Parsing Functions ---

def parse_csv(content: str):
    import numpy as np
    lines = content.strip().split('\n')
    if not lines:
        raise ValueError("Empty file")
    
    header = lines[0].lower()
    has_time = 'time' in header or 'sample' in header
    data_lines = lines[1:]
    
    if has_time:
        times = []
        values = []
        for line in data_lines:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    t = float(parts[0].strip())
                    v = float(parts[1].strip())
                    times.append(t)
                    values.append(v)
                except ValueError:
                    continue
        if not values:
            raise ValueError("No valid numeric data found")
        signal = np.array(values, dtype=np.float32)
        if len(times) > 1:
            dt = (times[-1] - times[0]) / (len(times) - 1)
            sampling_rate = 1.0 / dt if dt > 0 else 360.0
        else:
            sampling_rate = 360.0
    else:
        values = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            for p in line.split(','):
                try:
                    values.append(float(p.strip()))
                except ValueError:
                    continue
        if not values:
            raise ValueError("No valid numeric data found")
        signal = np.array(values, dtype=np.float32)
        sampling_rate = 360.0
    
    signal_type = "ecg"
    if 100 <= sampling_rate <= 256:
        signal_type = "eeg"
    elif sampling_rate > 500:
        signal_type = "emg"
    
    return {
        "data": signal.reshape(1, -1),
        "signal_type": signal_type,
        "channels": ["signal"],
        "sampling_rate": float(sampling_rate),
        "format": "csv",
    }


def parse_mat(content: bytes):
    import numpy as np
    from scipy.io import loadmat
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    try:
        mat = loadmat(temp_path)
        largest_key = None
        largest_size = 0
        for key in mat.keys():
            if key.startswith('__'):
                continue
            val = mat[key]
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                size = val.size
                if size > largest_size:
                    largest_size = size
                    largest_key = key
        
        if largest_key is None:
            raise ValueError("No numeric array found in .mat file")
        
        data = mat[largest_key]
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        channels = [f"ch{i}" for i in range(data.shape[0])]
        
        return {
            "data": data.astype(np.float32),
            "signal_type": "ecg",
            "channels": channels,
            "sampling_rate": 360.0,
            "format": "mat",
        }
    finally:
        os.unlink(temp_path)


def run_analysis(parsed, model_id):
    import numpy as np
    
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    
    model_info = MODEL_REGISTRY[model_id]
    
    # Preprocess
    preprocessed = preprocess_signal(
        parsed["data"],
        parsed["signal_type"],
        parsed["sampling_rate"],
        model_info["sampling_rate"]
    )
    
    segments = preprocessed["segments"]
    if not segments:
        raise ValueError("No segments extracted")
    
    # Demo mode
    predictions = demo_predict(segments, model_info["classes"])
    
    class_counts = {}
    class_confidences = {}
    for pred in predictions:
        cls = pred["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if cls not in class_confidences:
            class_confidences[cls] = []
        class_confidences[cls].append(pred["confidence"])
    
    avg_confidences = {cls: float(np.mean(confs)) for cls, confs in class_confidences.items()}
    dominant_class = max(class_counts, key=class_counts.get) if class_counts else model_info["classes"][0]
    
    n_samples = parsed["data"].shape[1]
    duration_sec = n_samples / parsed["sampling_rate"]
    
    max_preview = 5000
    if n_samples > max_preview:
        step = n_samples // max_preview
        preview_data = parsed["data"][:, ::step].tolist()
    else:
        preview_data = parsed["data"].tolist()
    
    return {
        "file_id": "vercel-stateless",
        "filename": parsed.get("filename", "unknown"),
        "signal_type": parsed["signal_type"],
        "format": parsed["format"],
        "channels": parsed["channels"],
        "n_channels": len(parsed["channels"]),
        "n_samples": n_samples,
        "sampling_rate": parsed["sampling_rate"],
        "duration_sec": round(duration_sec, 2),
        "preview_data": preview_data,
        "channel": parsed["channels"][0],
        "predictions": predictions,
        "summary": {
            "total_segments": len(predictions),
            "dominant_class": dominant_class,
            "class_distribution": class_counts,
            "average_confidences": avg_confidences,
        },
        "model_info": {
            "id": model_id,
            "description": model_info["description"],
            "classes": model_info["classes"],
        },
        "preprocessing": preprocessed["info"],
        "inference_backend": "demo",
        "demo_mode": True,
        "demo_note": "Running in DEMO mode — ONNX model not available on Vercel serverless",
    }


def preprocess_signal(data, signal_type, sampling_rate, target_sr):
    from scipy.signal import butter, filtfilt, resample
    import numpy as np
    
    signal = data[0]
    
    if signal_type == "ecg":
        filtered = _bandpass_filter(signal, 0.5, 40.0, sampling_rate)
        effective_sr = sampling_rate
        if target_sr and abs(sampling_rate - target_sr) > 0.1:
            n_target = int(len(filtered) * target_sr / sampling_rate)
            filtered = resample(filtered, n_target)
            effective_sr = target_sr
        normalized = _zscore_normalize(filtered)
        segments = _segment_signal(normalized, 187, overlap=0.5)
        info = {"preprocessing": "bandpass(0.5-40Hz) → normalize → segmentation", "segment_length": 187, "effective_sr": effective_sr}
    elif signal_type == "eeg":
        filtered = _bandpass_filter(signal, 0.5, 45.0, sampling_rate)
        effective_sr = target_sr or 100.0
        normalized = _zscore_normalize(filtered)
        epoch_samples = int(30 * effective_sr)
        segments = _segment_signal(normalized, epoch_samples)
        info = {"preprocessing": "bandpass(0.5-45Hz) → normalize → 30s epochs", "segment_length": epoch_samples, "effective_sr": effective_sr}
    elif signal_type == "emg":
        filtered = _bandpass_filter(signal, 20.0, 450.0, sampling_rate)
        rectified = np.abs(filtered)
        effective_sr = target_sr or 1000.0
        normalized = _zscore_normalize(rectified)
        seg_len = int(0.4 * effective_sr)
        segments = _segment_signal(normalized, seg_len, overlap=0.5)
        info = {"preprocessing": "bandpass → rectify → normalize → 400ms segments", "segment_length": seg_len, "effective_sr": effective_sr}
    else:
        segments = []
        info = {}
    
    return {"segments": segments, "info": info}


def _bandpass_filter(signal, lowcut, highcut, fs, order=4):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _zscore_normalize(signal):
    import numpy as np
    std = np.std(signal)
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def _segment_signal(signal, segment_length, overlap=0.0):
    import numpy as np
    step = max(1, int(segment_length * (1 - overlap)))
    segments = []
    for start in range(0, len(signal) - segment_length + 1, step):
        seg = signal[start:start + segment_length]
        segments.append(seg)
    if not segments and len(signal) > 0:
        padded = np.zeros(segment_length)
        padded[:len(signal)] = signal
        segments.append(padded)
    return segments


def demo_predict(segments, classes):
    import numpy as np
    predictions = []
    for seg in segments:
        mean = np.mean(seg)
        std = np.std(seg)
        rms = np.sqrt(np.mean(seg ** 2))
        
        seed = int(abs(mean * 1000 + std * 100 + rms * 10)) % 10000
        rng = np.random.RandomState(seed)
        probs = rng.dirichlet(np.ones(len(classes)) * 0.5)
        probs[0] = max(probs[0], 0.5)
        probs = probs / probs.sum()
        
        pred_idx = int(np.argmax(probs))
        predictions.append({
            "class": classes[pred_idx],
            "class_idx": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        })
    return predictions
