"""Model inference engine — supports PyTorch (.pt) and ONNX (.onnx) models.

Loads pre-trained models and runs inference on preprocessed biosignal segments.
Falls back to a demo predictor if model files are not available.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import MODEL_DIR, MODEL_REGISTRY


class Predictor:
    """Manages model loading and inference (PyTorch + ONNX + Demo)."""

    def __init__(self):
        self._models: dict = {}

    def _load_model(self, model_id: str):
        """Load a model into memory. Tries .pt (PyTorch) first, then .onnx."""
        if model_id in self._models:
            return self._models[model_id]

        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        registry = MODEL_REGISTRY[model_id]
        base_name = Path(registry["file"]).stem

        # Try PyTorch first
        pt_path = MODEL_DIR / f"{base_name}.pt"
        if pt_path.exists():
            model_obj = self._load_pytorch(pt_path, model_id)
            if model_obj is not None:
                self._models[model_id] = ("pytorch", model_obj)
                return self._models[model_id]

        # Try ONNX
        onnx_path = MODEL_DIR / registry["file"]
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                self._models[model_id] = ("onnx", session)
                return self._models[model_id]
            except ImportError:
                pass

        # No model found
        return None

    def _load_pytorch(self, path: Path, model_id: str):
        """Load a PyTorch model from .pt checkpoint."""
        try:
            import torch
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

            arch = checkpoint.get("architecture", "")
            n_classes = checkpoint.get("n_classes", 5)

            if arch == "ECGArrhythmiaCNN":
                from training.train_ecg_arrhythmia import ECGArrhythmiaCNN
                model = ECGArrhythmiaCNN(n_classes=n_classes)
            elif arch == "EEGSleepCNN":
                from training.train_eeg_sleep import EEGSleepCNN
                model = EEGSleepCNN(n_classes=n_classes)
            elif arch == "EMGGestureCNN":
                from training.train_emg_gesture import EMGGestureCNN
                model = EMGGestureCNN(n_classes=n_classes)
            else:
                print(f"  Unknown architecture: {arch}, skipping PyTorch load")
                return None

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            accuracy = checkpoint.get("test_accuracy", "N/A")
            print(f"  Loaded PyTorch model: {arch} (test acc: {accuracy})")
            return model

        except Exception as e:
            print(f"  Failed to load PyTorch model {path}: {e}")
            return None

    def predict(self, model_id: str, segments: list[np.ndarray]) -> dict:
        """Run inference on preprocessed segments."""
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = MODEL_REGISTRY[model_id]
        loaded = self._load_model(model_id)

        if loaded is not None:
            backend_type, model_obj = loaded
            if backend_type == "pytorch":
                return self._predict_pytorch(model_obj, model_id, segments)
            else:
                return self._predict_onnx(model_obj, model_id, segments)
        else:
            return self._predict_demo(model_id, segments)

    def _predict_pytorch(self, model, model_id: str, segments: list[np.ndarray]) -> dict:
        """Run PyTorch inference."""
        import torch

        model_info = MODEL_REGISTRY[model_id]
        classes = model_info["classes"]

        predictions = []
        with torch.no_grad():
            for seg in segments:
                x = torch.FloatTensor(seg).unsqueeze(0).unsqueeze(0)  # (1, 1, seg_len)
                output = model(x)
                probs = torch.softmax(output, dim=1).numpy()[0]

                pred_idx = int(np.argmax(probs))
                predictions.append({
                    "class": classes[pred_idx],
                    "class_idx": pred_idx,
                    "confidence": float(probs[pred_idx]),
                    "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
                })

        result = _build_result(predictions, model_info)
        result["inference_backend"] = "pytorch"
        return result

    def _predict_onnx(self, session, model_id: str, segments: list[np.ndarray]) -> dict:
        """Run ONNX inference."""
        model_info = MODEL_REGISTRY[model_id]
        classes = model_info["classes"]
        input_name = session.get_inputs()[0].name

        predictions = []
        for seg in segments:
            x = seg.astype(np.float32)
            input_shape = session.get_inputs()[0].shape
            if len(input_shape) == 3:
                x = x.reshape(1, 1, -1)
            else:
                x = x.reshape(1, -1)

            output = session.run(None, {input_name: x})[0]
            probs = _softmax(output[0])

            pred_idx = int(np.argmax(probs))
            predictions.append({
                "class": classes[pred_idx],
                "class_idx": pred_idx,
                "confidence": float(probs[pred_idx]),
                "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            })

        result = _build_result(predictions, model_info)
        result["inference_backend"] = "onnx"
        return result

    def _predict_demo(self, model_id: str, segments: list[np.ndarray]) -> dict:
        """Demo predictor when no model files are available."""
        model_info = MODEL_REGISTRY[model_id]
        classes = model_info["classes"]
        n_classes = len(classes)

        predictions = []
        for seg in segments:
            features = _extract_simple_features(seg)
            probs = _feature_to_probs(features, n_classes)

            pred_idx = int(np.argmax(probs))
            predictions.append({
                "class": classes[pred_idx],
                "class_idx": pred_idx,
                "confidence": float(probs[pred_idx]),
                "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            })

        result = _build_result(predictions, model_info)
        result["demo_mode"] = True
        result["demo_note"] = (
            "Running in DEMO mode — predictions are feature-based estimates, not from a trained model. "
            "Run 'python training/train_ecg_arrhythmia.py' to train a real model."
        )
        return result


def _extract_simple_features(seg: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(seg)),
        "std": float(np.std(seg)),
        "max": float(np.max(seg)),
        "min": float(np.min(seg)),
        "rms": float(np.sqrt(np.mean(seg ** 2))),
        "zero_crossings": int(np.sum(np.diff(np.sign(seg)) != 0)),
        "energy": float(np.sum(seg ** 2)),
    }


def _feature_to_probs(features: dict, n_classes: int) -> np.ndarray:
    seed_val = int(abs(features["mean"] * 1000 + features["std"] * 100 + features["rms"] * 10)) % 10000
    rng = np.random.RandomState(seed_val)
    alpha = np.ones(n_classes) * 0.3
    alpha[0] = 2.0
    probs = rng.dirichlet(alpha)
    max_idx = np.argmax(probs)
    probs[max_idx] = max(probs[max_idx], 0.6)
    probs = probs / probs.sum()
    return probs


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _build_result(predictions: list[dict], model_info: dict) -> dict:
    classes = model_info["classes"]

    class_counts = {}
    for pred in predictions:
        cls = pred["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    class_confidences = {}
    for pred in predictions:
        cls = pred["class"]
        if cls not in class_confidences:
            class_confidences[cls] = []
        class_confidences[cls].append(pred["confidence"])
    avg_confidences = {cls: float(np.mean(confs)) for cls, confs in class_confidences.items()}

    dominant_class = max(class_counts, key=class_counts.get) if class_counts else classes[0]

    return {
        "predictions": predictions,
        "summary": {
            "total_segments": len(predictions),
            "dominant_class": dominant_class,
            "class_distribution": class_counts,
            "average_confidences": avg_confidences,
        },
        "model_info": {
            "id": [k for k, v in MODEL_REGISTRY.items() if v == model_info][0],
            "description": model_info["description"],
            "classes": classes,
        },
    }


# Singleton predictor instance
predictor = Predictor()
