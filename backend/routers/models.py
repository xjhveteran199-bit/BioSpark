"""Model info endpoints."""

from fastapi import APIRouter
from backend.config import MODEL_REGISTRY

router = APIRouter()


@router.get("/models")
def list_models():
    """List all available pre-trained models."""
    result = []
    for model_id, info in MODEL_REGISTRY.items():
        result.append({
            "id": model_id,
            "signal_type": info["signal_type"],
            "description": info["description"],
            "classes": info["classes"],
            "input_length": info["input_length"],
        })
    return {"models": result}


@router.get("/models/{model_id}")
def get_model_info(model_id: str):
    """Get details about a specific model."""
    if model_id not in MODEL_REGISTRY:
        return {"error": f"Model '{model_id}' not found"}
    info = MODEL_REGISTRY[model_id]
    return {
        "id": model_id,
        "signal_type": info["signal_type"],
        "description": info["description"],
        "classes": info["classes"],
        "input_length": info["input_length"],
        "sampling_rate": info["sampling_rate"],
    }
