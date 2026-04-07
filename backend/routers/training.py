"""
Training router — Phases 1–4.

Phase 1: Labeled data upload and dataset parsing.
Phase 2: Training start + real-time WebSocket metrics streaming.
Phase 3: Post-training visualizations (confusion matrix, t-SNE).
Phase 4: Export endpoints (model .pt, history JSON, CSV exports, HTML report).
"""

import asyncio
import csv
import io
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.services.dataset_loader import load_labeled_dataset

# Lazy-import trainer (depends on torch, which may not be available on serverless)
_trainer_module = None

def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        from backend.services import trainer as _mod
        _trainer_module = _mod
    return _trainer_module

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
    n_channels: int = Field(default=0, ge=0, le=64, description="0=auto-detect, 1=single channel, >1=split evenly")
    # Auto-optimization fields
    auto_mode: bool = Field(default=False, description="Enable auto LR finder, architecture selection, and class weights")
    early_stopping_patience: int = Field(default=10, ge=3, le=50, description="Epochs without improvement before early stop")
    use_class_weights: bool = Field(default=True, description="Auto-compute class weights for imbalanced data")


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
        "n_channels": req.n_channels,
        "auto_mode": req.auto_mode,
        "early_stopping_patience": req.early_stopping_patience,
        "use_class_weights": req.use_class_weights,
    }

    job_id = str(uuid.uuid4())[:8]

    try:
        _get_trainer().training_manager.start(
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
    job = _get_trainer().training_manager.get(job_id)
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

    job = _get_trainer().training_manager.get(job_id)
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
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return _get_trainer().compute_confusion_matrix(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/train/{job_id}/tsne")
async def get_tsne(job_id: str, perplexity: float = 30.0):
    """
    Compute t-SNE on penultimate-layer features (128-d → 2-D) for the
    validation set.

    Returns x/y coordinates + class labels for a Plotly scatter plot.
    """
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return _get_trainer().compute_tsne(job, perplexity=perplexity)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Phase 4 — Export endpoints
# ---------------------------------------------------------------------------

def _require_completed_job(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    return job


@router.get("/train/{job_id}/export/model")
async def export_model(job_id: str):
    """Download the trained PyTorch model as a .pt file."""
    job = _require_completed_job(job_id)
    buf = io.BytesIO()
    import torch
    torch.save(job.model.state_dict(), buf)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="biospark_model_{job_id}.pt"'},
    )


@router.get("/train/{job_id}/export/history")
async def export_history(job_id: str):
    """Download training history as JSON (epoch-by-epoch metrics + config)."""
    job = _require_completed_job(job_id)
    payload = {
        "job_id": job_id,
        "config": job.config,
        "class_names": job.class_names,
        "best_val_acc": job.best_val_acc,
        "history": job.history,
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_history_{job_id}.json"'},
    )


@router.get("/train/{job_id}/export/confusion_matrix_csv")
async def export_confusion_matrix_csv(job_id: str):
    """Download confusion matrix + per-class metrics as CSV."""
    job = _require_completed_job(job_id)
    data = _get_trainer().compute_confusion_matrix(job)

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Confusion matrix section
    writer.writerow(["# Confusion Matrix (rows=true, cols=predicted)"])
    writer.writerow([""] + data["class_names"])
    for i, row in enumerate(data["matrix"]):
        writer.writerow([data["class_names"][i]] + row)
    writer.writerow([])

    # Per-class metrics
    writer.writerow(["# Per-Class Metrics"])
    writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
    for c in data["per_class"]:
        writer.writerow([c["class"], c["precision"], c["recall"], c["f1"], c["support"]])
    writer.writerow(["Overall Accuracy", data["accuracy"], "", "", ""])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="confusion_matrix_{job_id}.csv"'},
    )


@router.get("/train/{job_id}/export/tsne_csv")
async def export_tsne_csv(job_id: str):
    """Download t-SNE coordinates as CSV (x, y, label)."""
    job = _require_completed_job(job_id)
    data = _get_trainer().compute_tsne(job)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x", "y", "label"])
    for x, y, label in zip(data["x"], data["y"], data["labels"]):
        writer.writerow([round(x, 6), round(y, 6), label])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="tsne_{job_id}.csv"'},
    )


@router.get("/train/{job_id}/export/report")
async def export_report(job_id: str):
    """Generate a self-contained HTML report with all training results."""
    job = _require_completed_job(job_id)
    cm_data = _get_trainer().compute_confusion_matrix(job)
    tsne_data = _get_trainer().compute_tsne(job)

    history_json = json.dumps(job.history, ensure_ascii=False)
    cm_json = json.dumps(cm_data, ensure_ascii=False)
    tsne_json = json.dumps(tsne_data, ensure_ascii=False)
    config_json = json.dumps(job.config, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BioSpark Training Report — {job_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #2563eb; }} h2 {{ color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  .chart {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
  .config-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
  .config-item {{ background: white; border-radius: 8px; padding: 0.75rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
  .config-item .label {{ font-size: 0.8rem; color: #64748b; }} .config-item .value {{ font-size: 1.2rem; font-weight: 600; color: #2563eb; }}
  .footer {{ text-align: center; margin-top: 3rem; color: #94a3b8; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>BioSpark Training Report</h1>
<p>Job ID: <code>{job_id}</code> &middot; Classes: {len(job.class_names)} ({', '.join(job.class_names)}) &middot; Best Val Accuracy: <strong>{job.best_val_acc*100:.2f}%</strong></p>

<h2>Training Configuration</h2>
<div class="config-grid">
  <div class="config-item"><div class="label">Epochs</div><div class="value">{job.config.get('epochs','?')}</div></div>
  <div class="config-item"><div class="label">Learning Rate</div><div class="value">{job.config.get('learning_rate','?')}</div></div>
  <div class="config-item"><div class="label">Batch Size</div><div class="value">{job.config.get('batch_size','?')}</div></div>
  <div class="config-item"><div class="label">Val Split</div><div class="value">{job.config.get('val_split','?')}</div></div>
</div>

<h2>Training Curves</h2>
<div class="grid">
  <div class="chart" id="loss-chart"></div>
  <div class="chart" id="acc-chart"></div>
</div>

<h2>Confusion Matrix</h2>
<div class="grid">
  <div class="chart" id="cm-chart"></div>
  <div>
    <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>
    {''.join(f'<tr><td>{c["class"]}</td><td>{c["precision"]*100:.1f}%</td><td>{c["recall"]*100:.1f}%</td><td>{c["f1"]*100:.1f}%</td><td>{c["support"]}</td></tr>' for c in cm_data["per_class"])}
    <tr style="font-weight:600;border-top:2px solid #cbd5e1"><td>Overall</td><td colspan="4">{cm_data["accuracy"]*100:.2f}%</td></tr>
    </tbody></table>
  </div>
</div>

<h2>t-SNE Feature Visualization</h2>
<div class="chart" id="tsne-chart" style="height:500px;"></div>

<div class="footer">Generated by BioSpark &middot; AI-Powered Biosignal Training Platform</div>

<script>
const history = {history_json};
const cm = {cm_json};
const tsne = {tsne_json};
const COLORS = ['#2563eb','#7c3aed','#059669','#d97706','#dc2626','#0891b2','#65a30d','#c026d3','#ea580c','#0f766e'];

// Loss chart
Plotly.newPlot('loss-chart', [
  {{ x: history.map(h=>h.epoch), y: history.map(h=>h.train_loss), name:'Train Loss', mode:'lines+markers', line:{{color:'#2563eb'}} }},
  {{ x: history.map(h=>h.epoch), y: history.map(h=>h.val_loss), name:'Val Loss', mode:'lines+markers', line:{{color:'#dc2626',dash:'dash'}} }},
], {{ margin:{{t:30,r:10,b:40,l:50}}, xaxis:{{title:'Epoch'}}, yaxis:{{title:'Loss'}}, legend:{{orientation:'h',y:1.12}} }}, {{responsive:true, displayModeBar:false}});

// Accuracy chart
Plotly.newPlot('acc-chart', [
  {{ x: history.map(h=>h.epoch), y: history.map(h=>h.train_acc), name:'Train Acc', mode:'lines+markers', line:{{color:'#2563eb'}} }},
  {{ x: history.map(h=>h.epoch), y: history.map(h=>h.val_acc), name:'Val Acc', mode:'lines+markers', line:{{color:'#dc2626',dash:'dash'}} }},
], {{ margin:{{t:30,r:10,b:40,l:50}}, xaxis:{{title:'Epoch'}}, yaxis:{{title:'Accuracy'}}, legend:{{orientation:'h',y:1.12}} }}, {{responsive:true, displayModeBar:false}});

// Confusion matrix
const n = cm.class_names.length;
const annot = [];
for(let i=0;i<n;i++) for(let j=0;j<n;j++) annot.push({{x:cm.class_names[j],y:cm.class_names[i],text:String(cm.matrix[i][j]),showarrow:false,font:{{color:cm.matrix[i][j]>0?'white':'#999',size:13}}}});
Plotly.newPlot('cm-chart', [{{z:cm.matrix,x:cm.class_names,y:cm.class_names,type:'heatmap',colorscale:[[0,'#f0f4ff'],[0.5,'#6366f1'],[1,'#1e1b4b']],showscale:false}}],
  {{margin:{{t:20,r:10,b:60,l:80}},xaxis:{{title:'Predicted'}},yaxis:{{title:'True',autorange:'reversed'}},annotations:annot}}, {{responsive:true,displayModeBar:false}});

// t-SNE
const traces = tsne.class_names.map((cls,i) => {{
  const idx = tsne.labels.map((l,j)=>l===cls?j:-1).filter(j=>j>=0);
  return {{x:idx.map(j=>tsne.x[j]),y:idx.map(j=>tsne.y[j]),mode:'markers',type:'scatter',name:cls,marker:{{size:7,color:COLORS[i%COLORS.length],opacity:0.8}}}};
}});
Plotly.newPlot('tsne-chart', traces, {{margin:{{t:10,r:10,b:40,l:40}},xaxis:{{title:'t-SNE 1',zeroline:false}},yaxis:{{title:'t-SNE 2',zeroline:false}},legend:{{orientation:'h',y:1.08}}}}, {{responsive:true,displayModeBar:false}});
</script>
</body></html>"""

    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="biospark_report_{job_id}.html"'},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()
