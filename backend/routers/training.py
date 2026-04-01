"""
Training router — Phase 1: Labeled data upload and dataset parsing.
"""

import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.dataset_loader import load_labeled_dataset

router = APIRouter()

# In-memory dataset store keyed by dataset_id
_dataset_cache: dict[str, dict] = {}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@router.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload a labeled biosignal dataset for model training.

    Accepted formats:
    - **CSV** — must contain a `label` (or `class`/`target`/`y`) column
    - **ZIP** — folder-per-class structure: `class_name/sample.csv`

    Returns a dataset summary including class names, sample counts per class,
    signal shape, and a short preview for visualization.
    """
    filename = file.filename or "dataset"
    ext = Path_ext(filename)

    if ext not in ("csv", "txt", "zip"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Upload a .csv or .zip file.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 200 MB.",
        )

    try:
        summary = load_labeled_dataset(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse dataset: {exc}"
        )

    dataset_id = str(uuid.uuid4())[:8]
    _dataset_cache[dataset_id] = {
        "filename": filename,
        "summary": summary,
        # Store bytes so Phase 2 (trainer) can access the raw data
        "file_bytes": file_bytes,
    }

    return {"dataset_id": dataset_id, "filename": filename, **summary}


@router.get("/train/dataset/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Return the summary for a previously uploaded dataset."""
    entry = _dataset_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        "dataset_id": dataset_id,
        "filename": entry["filename"],
        **entry["summary"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def Path_ext(filename: str) -> str:
    """Return lowercase extension without the dot."""
    from pathlib import Path
    return Path(filename).suffix.lstrip(".").lower()
