"""
Training router — Phase 1 & 2.

Phase 1: Labeled data upload and dataset parsing.
Phase 2: Training start + real-time WebSocket metrics streaming.
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.services.dataset_loader import load_labeled_dataset
from backend.services.trainer import training_manager, compute_confusion_matrix, compute_tsne

router = APIRouter()

# In-memory dataset store keyed by dataset_id
_dataset_cache: dict[str, dict] = {}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


# ---------------------------------------------------------------------------
# Phase 1 — Upload
# ---------------------------------------------------------------------------

@router.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload a labeled biosignal dataset for model training.

    Accepted formats:
    - **CSV** — must contain a `label` (or `class`/`target`/`y`) column
    - **ZIP** — folder-per-class structure: `class_name/sample.csv`
    """
    filename = file.filename or "dataset"
    ext = _path_ext(filename)

    if ext not in ("csv", "txt", "zip"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Upload a .csv or .zip file.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum 200 MB.")

    try:
        summary = load_labeled_dataset(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse dataset: {exc}")

    dataset_id = str(uuid.uuid4())[:8]
    _dataset_cache[dataset_id] = {
        "filename": filename,
        "summary": summary,
        "file_bytes": file_bytes,
    }

    return {"dataset_id": dataset_id, "filename": filename, **summary}


@router.get("/train/dataset/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Return the summary for a previously uploaded dataset."""
    entry = _dataset_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"dataset_id": dataset_id, "filename": entry["filename"], **entry["summary"]}


# ---------------------------------------------------------------------------
# Phase 2 — Training
# ---------------------------------------------------------------------------

class TrainStartRequest(BaseModel):
    dataset_id: str
    epochs: int = Field(default=30, ge=1, le=200)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    batch_size: int = Field(default=64, ge=4, le=512)
    val_split: float = Field(default=0.2, gt=0.0, lt=1.0)


@router.post("/train/start")
async def start_training(req: TrainStartRequest):
    """
    Start a training job on a previously uploaded dataset.

    Returns a ``job_id`` that can be used with the WebSocket endpoint
    ``/api/train/ws/{job_id}`` to receive live epoch metrics.
    """
    entry = _dataset_cache.get(req.dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    config = {
        "epochs": req.epochs,
        "learning_rate": req.learning_rate,
        "batch_size": req.batch_size,
        "val_split": req.val_split,
    }

    job_id = str(uuid.uuid4())[:8]

    try:
        training_manager.start(
            job_id=job_id,
            file_bytes=entry["file_bytes"],
            filename=entry["filename"],
            summary=entry["summary"],
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {exc}")

    return {"job_id": job_id, "status": "started", "config": config}


@router.get("/train/{job_id}/status")
async def get_training_status(job_id: str):
    """Poll-based fallback — returns current job status + full history."""
    job = training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return {
        "job_id": job_id,
        "status": job.status,
        "best_val_acc": job.best_val_acc,
        "history": job.history,
        "error": job.error,
        "class_names": job.class_names,
        "config": job.config,
    }


@router.websocket("/train/ws/{job_id}")
async def training_websocket(ws: WebSocket, job_id: str):
    """
    Stream live training metrics over a WebSocket connection.

    Messages are JSON objects with a ``type`` field:
    - ``start``    — training configuration summary
    - ``epoch``    — per-epoch metrics (train_loss, val_loss, train_acc, val_acc)
    - ``complete`` — final summary (best_val_acc, total_epochs)
    - ``error``    — error message

    The server also sends any metrics that were recorded *before* the
    WebSocket connected (so late-joiners don't miss early epochs).
    """
    await ws.accept()

    job = training_manager.get(job_id)
    if job is None:
        await ws.send_json({"type": "error", "message": "Job not found."})
        await ws.close()
        return

    loop = asyncio.get_event_loop()

    # Replay any already-recorded history (handles late-connecting clients)
    for past_metric in job.history:
        await ws.send_json(past_metric)

    # If job already finished, send status and close
    if job.status == "completed":
        await ws.send_json({
            "type": "complete",
            "best_val_acc": round(job.best_val_acc, 5),
            "total_epochs": len(job.history),
        })
        await ws.close()
        return
    if job.status == "failed":
        await ws.send_json({"type": "error", "message": job.error or "Training failed"})
        await ws.close()
        return

    # Live streaming via callback
    queue: asyncio.Queue = asyncio.Queue()

    async def _on_metric(payload: dict):
        await queue.put(payload)

    job.register_callback(_on_metric, loop)

    try:
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
            if payload.get("type") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job.unregister_callback(_on_metric)


# ---------------------------------------------------------------------------
# Phase 3 — Post-training visualizations
# ---------------------------------------------------------------------------

@router.get("/train/{job_id}/confusion_matrix")
async def get_confusion_matrix(job_id: str):
    """
    Compute and return the confusion matrix for the validation set.

    Returns the matrix as a 2-D array, class names, per-class precision /
    recall / F1, and overall accuracy.
    """
    job = training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return compute_confusion_matrix(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/train/{job_id}/tsne")
async def get_tsne(job_id: str, perplexity: float = 30.0):
    """
    Compute t-SNE on penultimate-layer features (128-d → 2-D) for the
    validation set.

    Returns x/y coordinates + class labels for a Plotly scatter plot.
    """
    job = training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return compute_tsne(job, perplexity=perplexity)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()
