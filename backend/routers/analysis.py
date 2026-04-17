"""Analysis/prediction endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.config import MODEL_REGISTRY
from backend.routers.upload import get_parsed_data
from backend.services.preprocess import preprocess
from backend.services.predictor import predictor

router = APIRouter()


def _prepare_segments(file_id: str, model_id: str, channel: int) -> tuple:
    """Shared preprocessing: parse → preprocess → pad/truncate segments.

    Returns (segments, parsed, preprocessed).
    """
    import numpy as np

    parsed = get_parsed_data(file_id)

    if model_id not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise HTTPException(400, f"Unknown model: {model_id}. Available: {available}")

    model_info = MODEL_REGISTRY[model_id]

    if parsed["signal_type"] != model_info["signal_type"]:
        raise HTTPException(
            400,
            f"Signal type mismatch: file is '{parsed['signal_type']}' but model expects '{model_info['signal_type']}'."
        )

    if channel >= len(parsed["channels"]):
        raise HTTPException(400, f"Channel {channel} out of range. File has {len(parsed['channels'])} channels.")

    try:
        preprocessed = preprocess(
            data=parsed["data"],
            signal_type=parsed["signal_type"],
            sampling_rate=parsed["sampling_rate"],
            target_sr=model_info["sampling_rate"],
            channel_idx=channel,
            model_id=model_id,
        )
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {str(e)}")

    if not preprocessed["segments"]:
        raise HTTPException(400, "No segments extracted after preprocessing. Signal may be too short.")

    target_len = model_info["input_length"]
    is_multichannel = preprocessed["info"].get("multichannel", False)
    segments = []

    for seg in preprocessed["segments"]:
        if is_multichannel:
            if seg.shape[-1] == target_len:
                segments.append(seg)
            elif seg.shape[-1] > target_len:
                segments.append(seg[:, :target_len])
            else:
                padded = np.zeros((seg.shape[0], target_len), dtype=np.float32)
                padded[:, :seg.shape[-1]] = seg
                segments.append(padded)
        else:
            if len(seg) == target_len:
                segments.append(seg)
            elif len(seg) > target_len:
                segments.append(seg[:target_len])
            else:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[:len(seg)] = seg
                segments.append(padded)

    return segments, parsed, preprocessed


@router.post("/analyze/{file_id}")
def analyze_signal(
    file_id: str,
    model_id: str = Query(..., description="Model to use for prediction"),
    channel: int = Query(0, description="Channel index to analyze"),
):
    """Run analysis on an uploaded file using a pre-trained model.

    1. Retrieves parsed signal data
    2. Preprocesses the signal
    3. Runs model inference
    4. Returns predictions with confidence scores
    """
    segments, parsed, preprocessed = _prepare_segments(file_id, model_id, channel)

    # Run prediction
    try:
        result = predictor.predict(model_id, segments)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    result["file_id"] = file_id
    result["channel"] = parsed["channels"][channel]
    result["preprocessing"] = preprocessed["info"]

    return result


@router.post("/gradcam/{file_id}")
def gradcam_signal(
    file_id: str,
    model_id: str = Query(..., description="Model to use"),
    channel: int = Query(0, description="Channel index to analyze"),
    target_class: Optional[int] = Query(None, description="Target class index (None = predicted)"),
    max_segments: int = Query(20, description="Max segments to compute Grad-CAM for"),
):
    """Compute Grad-CAM attention heatmaps for an uploaded signal.

    Requires a PyTorch model — ONNX-only models fall back to a
    gradient-free approximation.  Returns per-segment heatmaps aligned
    with the input signal, showing which regions the CNN focuses on.
    """
    segments, parsed, preprocessed = _prepare_segments(file_id, model_id, channel)

    model_info = MODEL_REGISTRY[model_id]
    in_channels = model_info.get("in_channels", 1)

    # Load PyTorch model
    loaded = predictor._load_model(model_id)
    if loaded is None or loaded[0] != "pytorch":
        raise HTTPException(
            400,
            "Grad-CAM requires a PyTorch model (.pt). "
            "This model is ONNX-only or not available. "
            "Train a PyTorch model first."
        )

    _, pt_model = loaded

    try:
        from backend.services.gradcam import compute_gradcam_for_segments
        results = compute_gradcam_for_segments(
            model=pt_model,
            segments=segments,
            in_channels=in_channels,
            target_class=target_class,
            max_segments=max_segments,
        )
    except Exception as e:
        raise HTTPException(500, f"Grad-CAM computation failed: {str(e)}")

    return {
        "file_id": file_id,
        "model_id": model_id,
        "channel": parsed["channels"][channel],
        "classes": model_info["classes"],
        "total_segments": len(segments),
        "computed_segments": len(results),
        "gradcam": results,
    }
