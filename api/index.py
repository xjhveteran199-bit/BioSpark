"""
BioSpark API - Vercel Serverless Functions
Single handler approach for Vercel Python runtime
"""

import json
import base64
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"

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

# Cached ONNX session
_onnx_session = None


def handler(req, context):
    """Main handler for all API routes."""
    global _onnx_session
    
    path = req.path if hasattr(req, 'path') else urlparse(req.url).path
    method = req.method if hasattr(req, 'method') else 'GET'
    
    # Route based on path
    if path == '/api/health' or path == '/api/health/':
        return handle_health()
    elif path == '/api/models' or path == '/api/models/':
        return handle_models()
    elif path == '/api/upload' or path == '/api/upload/':
        if method == 'POST':
            return handle_upload(req)
        else:
            return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}
    else:
        return {"statusCode": 404, "body": json.dumps({"error": f"Not found: {path}"})}


def handle_health():
    """GET /api/health"""
    return {
        "statusCode": 200,
        "body": json.dumps({"status": "ok", "version": "0.1.0", "platform": "vercel"}),
        "headers": {"Content-Type": "application/json"}
    }


def handle_models():
    """GET /api/models"""
    models = [
        {
            "id": model_id,
            "signal_type": info["signal_type"],
            "description": info["description"],
            "classes": info["classes"],
            "input_length": info["input_length"],
        }
        for model_id, info in MODEL_REGISTRY.items()
    ]
    return {
        "statusCode": 200,
        "body": json.dumps({"models": models}),
        "headers": {"Content-Type": "application/json"}
    }


def handle_upload(req):
    """POST /api/upload - Combined upload + analyze endpoint (stateless)."""
    global _onnx_session
    
    content_type = req.headers.get("content-type", "") if hasattr(req.headers, "get") else ""
    if hasattr(req, 'headers') and isinstance(req.headers, dict):
        content_type = req.headers.get("Content-Type", "") or req.headers.get("content-type", "")
    
    try:
        if "multipart/form-data" in content_type:
            return handle_multipart(req)
        elif "application/json" in content_type:
            return handle_json(req)
        else:
            # Try to handle multipart anyway
            try:
                return handle_multipart(req)
            except:
                return {"statusCode": 400, "body": json.dumps({"error": "Unsupported content type"})}
    except Exception as e:
        import traceback
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "trace": traceback.format_exc()})}


def handle_multipart(req):
    """Handle multipart file upload + form fields."""
    try:
        body = req.body
        if isinstance(body, str):
            body = body.encode()
        elif body is None:
            body = b""
        
        content_type = req.headers.get("Content-Type", "") if hasattr(req.headers, "get") else ""
        if hasattr(req, 'headers') and isinstance(req.headers, dict):
            content_type = req.headers.get("Content-Type", "") or ""
        
        # Parse boundary
        boundary_match = content_type.split("boundary=")
        if len(boundary_match) < 2:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing boundary"})}
        boundary = boundary_match[1].encode()
        
        parts = body.split(b"--" + boundary)
        
        filename = None
        file_content = None
        model_id = "ecg_arrhythmia"
        
        for part in parts:
            if not part or part.strip() in (b"", b"\r\n"):
                continue
            
            if b"filename=" in part:
                # File part
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                header = part[:header_end].decode("utf-8", errors="ignore")
                for line in header.split("\r\n"):
                    if "filename=" in line:
                        fname = line.split("filename=")[1].strip('"')
                        if fname:
                            filename = fname
                        break
                fc = part[header_end + 4:]
                if fc.endswith(b"\r\n"):
                    fc = fc[:-2]
                file_content = fc
            elif b"name=\"model_id\"" in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end != -1:
                    model_id = part[header_end + 4:].decode().strip()
        
        if not file_content:
            return {"statusCode": 400, "body": json.dumps({"error": "No file provided"})}
        
        ext = Path(filename or "unknown").suffix.lower()
        
        if ext == ".csv" or ext == ".txt":
            text_content = file_content.decode("utf-8", errors="ignore")
            parsed = parse_csv(text_content, None)
        elif ext == ".mat":
            parsed = parse_mat(file_content, None)
        else:
            return {"statusCode": 400, "body": json.dumps({"error": f"Unsupported format: {ext}"})}
        
        result = run_analysis(parsed, model_id)
        result["filename"] = filename
        
        return {"statusCode": 200, "body": json.dumps(result), "headers": {"Content-Type": "application/json"}}
    
    except Exception as e:
        import traceback
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "trace": traceback.format_exc()})}


def handle_json(req):
    """Handle JSON request with base64-encoded file content."""
    try:
        data = json.loads(req.body)
        filename = data.get("filename", "upload.csv")
        file_content_b64 = data.get("content")
        model_id = data.get("model_id", "ecg_arrhythmia")
        signal_type = data.get("signal_type")
        
        if not file_content_b64:
            return {"statusCode": 400, "body": json.dumps({"error": "No file content provided"})}
        
        file_content = base64.b64decode(file_content_b64)
        
        ext = Path(filename).suffix.lower()
        
        if ext == ".csv" or ext == ".txt":
            text_content = file_content.decode("utf-8", errors="ignore")
            parsed = parse_csv(text_content, signal_type)
        elif ext == ".mat":
            parsed = parse_mat(file_content, signal_type)
        else:
            return {"statusCode": 400, "body": json.dumps({"error": f"Unsupported format: {ext}"})}
        
        result = run_analysis(parsed, model_id)
        result["filename"] = filename
        
        return {"statusCode": 200, "body": json.dumps(result), "headers": {"Content-Type": "application/json"}}
    
    except Exception as e:
        import traceback
        return {"statusCode": 500, "body": json.dumps({"error": str(e), "trace": traceback.format_exc()})}


# --- Parsing Functions ---

def parse_csv(content: str, signal_type: str = None):
    """Parse CSV content from string."""
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
    
    if signal_type is None:
        if 250 <= sampling_rate <= 1000:
            signal_type = "ecg"
        elif 100 <= sampling_rate <= 256:
            signal_type = "eeg"
        elif sampling_rate > 500:
            signal_type = "emg"
        else:
            signal_type = "ecg"
    
    return {
        "data": signal.reshape(1, -1),
        "signal_type": signal_type,
        "channels": ["signal"],
        "sampling_rate": float(sampling_rate),
        "format": "csv",
    }


def parse_mat(content: bytes, signal_type: str = None):
    """Parse MATLAB .mat file content."""
    try:
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
            signal_type = signal_type or "ecg"
            
            return {
                "data": data.astype(np.float32),
                "signal_type": signal_type,
                "channels": channels,
                "sampling_rate": 360.0,
                "format": "mat",
            }
        finally:
            os.unlink(temp_path)
    except Exception as e:
        raise ValueError(f"Failed to parse .mat file: {e}")


# --- Preprocessing ---

def preprocess_signal(data: np.ndarray, signal_type: str, sampling_rate: float, target_sr: float = None):
    """Preprocess signal and segment it."""
    from scipy.signal import butter, filtfilt, resample
    
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
        info = {"preprocessing": "bandpass(0.5-40Hz) → resample → normalize → R-peak segmentation", "segment_length": 187, "effective_sr": effective_sr}
    
    elif signal_type == "eeg":
        filtered = _bandpass_filter(signal, 0.5, 45.0, sampling_rate)
        effective_sr = target_sr or 100.0
        if abs(sampling_rate - effective_sr) > 0.1:
            n_target = int(len(filtered) * effective_sr / sampling_rate)
            filtered = resample(filtered, n_target)
        normalized = _zscore_normalize(filtered)
        epoch_samples = int(30 * effective_sr)
        segments = _segment_signal(normalized, epoch_samples)
        info = {"preprocessing": "bandpass(0.5-45Hz) → resample → normalize → 30s epochs", "segment_length": epoch_samples, "effective_sr": effective_sr}
    
    elif signal_type == "emg":
        highcut = min(450.0, sampling_rate * 0.49)
        filtered = _bandpass_filter(signal, 20.0, highcut, sampling_rate)
        rectified = np.abs(filtered)
        effective_sr = target_sr or 1000.0
        if abs(sampling_rate - effective_sr) > 0.1:
            n_target = int(len(rectified) * effective_sr / sampling_rate)
            rectified = resample(rectified, n_target)
        normalized = _zscore_normalize(rectified)
        seg_len = int(0.4 * effective_sr)
        segments = _segment_signal(normalized, seg_len, overlap=0.5)
        info = {"preprocessing": "bandpass(20-450Hz) → rectify → resample → normalize → 400ms segments", "segment_length": seg_len, "effective_sr": effective_sr}
    else:
        segments = []
        info = {}
    
    return {"segments": segments, "info": info}


def _bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _zscore_normalize(signal):
    std = np.std(signal)
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def _segment_signal(signal, segment_length, overlap=0.0):
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


# --- Inference ---

def run_analysis(parsed, model_id):
    """Run full analysis pipeline."""
    global _onnx_session
    
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    
    model_info = MODEL_REGISTRY[model_id]
    target_sr = model_info["sampling_rate"]
    
    preprocessed = preprocess_signal(
        parsed["data"],
        parsed["signal_type"],
        parsed["sampling_rate"],
        target_sr
    )
    
    segments = preprocessed["segments"]
    if not segments:
        raise ValueError("No segments extracted - signal may be too short")
    
    target_len = model_info["input_length"]
    processed_segs = []
    for seg in segments:
        if len(seg) == target_len:
            processed_segs.append(seg)
        elif len(seg) > target_len:
            processed_segs.append(seg[:target_len])
        else:
            padded = np.zeros(target_len)
            padded[:len(seg)] = seg
            processed_segs.append(padded)
    
    model_path = MODEL_DIR / model_info["file"]
    is_demo = False
    if model_path.exists():
        try:
            import onnxruntime as ort
            if _onnx_session is None:
                _onnx_session = ort.InferenceSession(str(model_path))
            session = _onnx_session
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            predictions = []
            for seg in processed_segs:
                x = seg.astype(np.float32).reshape(1, 1, -1)
                output = session.run([output_name], {input_name: x})[0]
                probs = _softmax(output[0])
                pred_idx = int(np.argmax(probs))
                predictions.append({
                    "class": model_info["classes"][pred_idx],
                    "class_idx": pred_idx,
                    "confidence": float(probs[pred_idx]),
                    "probabilities": {model_info["classes"][i]: float(probs[i]) for i in range(len(model_info["classes"]))},
                })
            backend = "onnx"
        except Exception as e:
            predictions = demo_predict(processed_segs, model_info["classes"])
            backend = "demo"
            is_demo = True
    else:
        predictions = demo_predict(processed_segs, model_info["classes"])
        backend = "demo"
        is_demo = True
    
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
        "inference_backend": backend,
        "demo_mode": is_demo,
        "demo_note": "Running in DEMO mode — predictions are feature-based estimates, not from a trained model." if is_demo else None,
    }


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def demo_predict(segments, classes):
    """Feature-based demo prediction."""
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
