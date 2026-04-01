"""File upload endpoints."""

import uuid
import json
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional

from backend.config import UPLOAD_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from backend.services.format_parser import parse_file

router = APIRouter()

# In-memory store for parsed file data (session-based)
_parsed_cache: dict[str, dict] = {}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    signal_type: Optional[str] = Query(None, description="Signal type: ecg, eeg, or emg"),
):
    """Upload a biosignal file and get parsed signal data."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    # Save to disk
    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    save_path.write_bytes(content)

    # Parse
    try:
        parsed = parse_file(str(save_path), signal_type)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    # Store parsed data in cache (convert numpy to list for JSON)
    _parsed_cache[file_id] = parsed

    # Downsample for preview (max 5000 points per channel)
    n_samples = parsed["data"].shape[1]
    max_preview = 5000
    if n_samples > max_preview:
        step = n_samples // max_preview
        preview_data = parsed["data"][:, ::step].tolist()
    else:
        preview_data = parsed["data"].tolist()

    return {
        "file_id": file_id,
        "filename": file.filename,
        "signal_type": parsed["signal_type"],
        "format": parsed["format"],
        "channels": parsed["channels"],
        "n_channels": len(parsed["channels"]),
        "n_samples": n_samples,
        "sampling_rate": parsed["sampling_rate"],
        "duration_sec": round(parsed["duration_sec"], 2),
        "preview_data": preview_data,
    }


@router.get("/files/{file_id}")
def get_file_info(file_id: str):
    """Get parsed file info by ID."""
    if file_id not in _parsed_cache:
        raise HTTPException(404, "File not found. Please re-upload.")

    parsed = _parsed_cache[file_id]
    return {
        "file_id": file_id,
        "signal_type": parsed["signal_type"],
        "channels": parsed["channels"],
        "n_channels": len(parsed["channels"]),
        "n_samples": parsed["data"].shape[1],
        "sampling_rate": parsed["sampling_rate"],
        "duration_sec": round(parsed["duration_sec"], 2),
    }


def get_parsed_data(file_id: str) -> dict:
    """Internal: get parsed data for a file_id (used by analysis router)."""
    if file_id not in _parsed_cache:
        raise HTTPException(404, "File not found. Please re-upload.")
    return _parsed_cache[file_id]
