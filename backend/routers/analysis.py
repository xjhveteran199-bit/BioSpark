"""Analysis/prediction endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.config import MODEL_REGISTRY
from backend.routers.upload import get_parsed_data
from backend.services.preprocess import preprocess
from backend.services.predictor import predictor

router = APIRouter()


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
    # Get parsed data
    parsed = get_parsed_data(file_id)

    # Validate model
    if model_id not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise HTTPException(400, f"Unknown model: {model_id}. Available: {available}")

    model_info = MODEL_REGISTRY[model_id]

    # Validate signal type compatibility
    if parsed["signal_type"] != model_info["signal_type"]:
        raise HTTPException(
            400,
            f"Signal type mismatch: file is '{parsed['signal_type']}' but model expects '{model_info['signal_type']}'. "
            f"You can override signal type during upload with ?signal_type={model_info['signal_type']}"
        )

    # Validate channel
    if channel >= len(parsed["channels"]):
        raise HTTPException(400, f"Channel {channel} out of range. File has {len(parsed['channels'])} channels.")

    # Preprocess
    try:
        preprocessed = preprocess(
            data=parsed["data"],
            signal_type=parsed["signal_type"],
            sampling_rate=parsed["sampling_rate"],
            target_sr=model_info["sampling_rate"],
            channel_idx=channel,
        )
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {str(e)}")

    if not preprocessed["segments"]:
        raise HTTPException(400, "No segments extracted after preprocessing. Signal may be too short.")

    # Ensure segments match model input length
    target_len = model_info["input_length"]
    segments = []
    for seg in preprocessed["segments"]:
        if len(seg) == target_len:
            segments.append(seg)
        elif len(seg) > target_len:
            segments.append(seg[:target_len])
        else:
            # Pad short segments
            import numpy as np
            padded = np.zeros(target_len)
            padded[:len(seg)] = seg
            segments.append(padded)

    # Run prediction
    try:
        result = predictor.predict(model_id, segments)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    result["file_id"] = file_id
    result["channel"] = parsed["channels"][channel]
    result["preprocessing"] = preprocessed["info"]

    return result
