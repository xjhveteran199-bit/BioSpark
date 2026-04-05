# BioSpark — Claude Session Handoff
> Last updated: 2026-04-01
> Read this file at the START of every new Claude session for this project.

---

## Project Location
`D:/Claude code/BioSignal-Platform/`

## Run Dev Server
```bash
python -m uvicorn backend.main:app --reload --reload-dir backend --reload-dir frontend --host 127.0.0.1 --port 8000
```
Open: http://localhost:8000

## GitHub
https://github.com/xjhveteran199-bit/BioSpark

---

## What BioSpark Currently Does (v0.1 — COMPLETE)

### Backend (FastAPI)
- `POST /api/upload` — accepts CSV/EDF/MAT/TXT, parses signal, returns metadata + downsampled preview
- `POST /api/analyze/{file_id}` — runs ECG arrhythmia inference, returns per-segment predictions
- `GET /api/models` — lists available models
- `GET /api/health` — health check

### ML Model (COMPLETE)
- **ECG Arrhythmia CNN** (`backend/models/ecg_arrhythmia_cnn.pt`)
- Architecture: 3× Conv1d blocks + GlobalAvgPool + FC head, **44,293 params**
- Trained on MIT-BIH (PhysioNet), 5-class AAMI: N / S / V / F / Q
- **Test accuracy: 94.13%**
- Fallback: demo mode (feature-based pseudo-predictions) when no `.pt` file found

### Frontend (Vanilla JS + Plotly.js)
- 4-step SPA: Upload → Signal Preview → Model Select → Results
- Drag-and-drop upload, signal type + sampling rate selection
- Interactive Plotly chart: time domain + client-side FFT spectrum
- Bilingual UI: EN / 中文 toggle
- Results: dominant class, confidence bars, Plotly pie chart, per-segment table

### Current Export (results.js)
- **Export JSON** (`biosignal_results.json`): full result object — segments, probabilities, summary, model info
- **Export CSV** (`biosignal_results.csv`): columns `segment, class, confidence`

---

## NEXT DEVELOPMENT PHASE — Training Platform Pivot

### User's Core Requirement
> "用户导入数据 → CNN-1D开始训练 → 直接得到主要结果：Accuracy/Loss曲线、Confusion Matrix、t-SNE"

This is a **major architecture pivot**: from inference-only → user-driven training platform.

---

## 4-Phase Development Plan

### Phase 1: Labeled Data Upload + Parsing
**Goal**: User uploads their own labeled dataset → system parses it

- Supported input formats:
  - CSV with a `label` column (e.g., `time, signal, label`)
  - ZIP folder: `class_A/001.csv`, `class_B/001.csv`, ...
- New endpoint: `POST /api/train/upload` — accepts labeled data, validates, returns dataset summary
- New service: `backend/services/dataset_loader.py`
  - Auto-detect label column or folder structure
  - Return: class names, sample counts per class, signal length stats
- Frontend: new "Training" tab/section with dataset preview (class distribution bar chart)

### Phase 2: Training Engine + Real-time Metrics via WebSocket
**Goal**: CNN-1D trains on uploaded data, user watches live Loss/Acc curves

- New endpoint: `WebSocket /api/train/ws/{job_id}` — streams training metrics
- New service: `backend/services/trainer.py`
  - Reuses `ECGArrhythmiaCNN` architecture (auto-adjusts output layer to N classes)
  - Trains in background thread (asyncio + ThreadPoolExecutor)
  - Emits JSON every epoch: `{"epoch": 5, "train_loss": 0.32, "val_loss": 0.41, "train_acc": 0.88, "val_acc": 0.84}`
- New endpoint: `POST /api/train/start` — kicks off training job, returns `job_id`
- Frontend: Live Plotly line charts for Loss and Accuracy (dual y-axis or 2 subplots)
- Hyperparameter controls: epochs, learning rate, batch size, train/val split

### Phase 3: Post-Training Visualizations
**Goal**: After training completes, show Confusion Matrix + t-SNE

- **Confusion Matrix**:
  - Backend computes with sklearn on validation set
  - Returns as 2D array → Plotly heatmap with class labels
  - Endpoint: `GET /api/train/{job_id}/confusion_matrix`

- **t-SNE**:
  - Backend extracts penultimate-layer features (128-dim) for all val samples
  - sklearn `TSNE(n_components=2)` → 2D coordinates + class labels
  - Returns as `{"x": [...], "y": [...], "labels": [...], "colors": [...]}`
  - Frontend: Plotly scatter with color-coded classes
  - Endpoint: `GET /api/train/{job_id}/tsne`

### Phase 4: Export + Report
**Goal**: User can download everything

- Download trained model: `GET /api/train/{job_id}/model` → `.pt` file
- Export training history: JSON with full epoch-by-epoch metrics
- Export confusion matrix: CSV
- Export t-SNE: CSV with `x, y, label` columns
- (Stretch) HTML report: single-file self-contained report with all charts embedded

---

## Key Architectural Decisions Already Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Model format | PyTorch `.pt` (not ONNX) | Python 3.14 incompatible with onnx package |
| Static serving | Sub-path mount `/css`, `/js` | Catch-all `mount("/")` shadows API routes |
| FFT | Client-side JS (Cooley-Tukey) | No server round-trip needed |
| Cache busting | `?v=2` on script tags | Prevents stale JS in browser |
| Training viz | WebSocket streaming | Real-time feedback without polling |

---

## File Structure (Current)
```
BioSignal-Platform/
├── backend/
│   ├── main.py                    # FastAPI app, route registration order matters
│   ├── config.py                  # MODEL_REGISTRY, PREPROCESS_CONFIG
│   ├── routers/
│   │   ├── upload.py
│   │   ├── analysis.py
│   │   └── models.py
│   ├── services/
│   │   ├── format_parser.py       # CSV/EDF/MAT parsing
│   │   ├── preprocess.py          # bandpass filter, R-peak, segmentation
│   │   └── predictor.py           # PyTorch inference + demo fallback
│   ├── models/
│   │   └── ecg_arrhythmia_cnn.pt  # Trained model (94.1% acc)
│   └── tests/
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js                 # Lang toggle, section flow
│       ├── uploader.js            # Drag-drop, upload API call
│       ├── visualizer.js          # Plotly time/FFT charts
│       └── results.js             # Results render + JSON/CSV export
├── training/
│   └── train_ecg_arrhythmia.py   # MIT-BIH training script
├── sample_data/                   # Demo CSV/EDF files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── CLAUDE_SESSION.md              # ← THIS FILE
```

---

## Where to Start Next Session

**Start Phase 1**: Build the labeled data upload endpoint and dataset parser.

1. Create `backend/services/dataset_loader.py`
2. Create `backend/routers/training.py` with `POST /api/train/upload`
3. Add new "Train" section to `frontend/index.html`
4. Register router in `backend/main.py`

Then continue Phase 2 → 3 → 4 in order.

---

## Known Issues / Watch Out For
- **Route order**: Always register API routers BEFORE `app.mount()` static files in `main.py`
- **Python 3.14**: Do NOT install `onnx` or `onnxruntime` — incompatible
- **Browser cache**: Increment `?v=N` on script tags after JS changes
- **Training in background**: Use `asyncio.get_event_loop().run_in_executor()` with ThreadPoolExecutor, NOT `asyncio.run()` inside async context
- **WebSocket**: FastAPI supports WebSockets natively via `from fastapi import WebSocket`
