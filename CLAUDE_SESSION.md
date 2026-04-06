# BioSpark — Claude Session Handoff
> Last updated: 2026-04-06 (v0.3)
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

## Live Deployments
- Railway (full): https://efficient-integrity-production-736e.up.railway.app
- Vercel (frontend): https://infallible-blackwell.vercel.app

---

## What BioSpark Currently Does (v0.3)

### 3 Pre-trained Models

| Model | Signal | Classes | Accuracy | File |
|-------|--------|---------|----------|------|
| ECG Arrhythmia CNN | ECG | 5 (N/S/V/F/Q) | 94.1% | `ecg_arrhythmia_cnn.pt` |
| EEG Sleep Staging CNN | EEG | 5 (W/N1/N2/N3/REM) | Training... | `eeg_sleep_staging.pt` |
| EMG Gesture CNN | EMG | 53 (52 gestures + Rest) | 42.7% | `emg_gesture_cnn.pt` |

### Backend (FastAPI)
- `POST /api/upload` — accepts CSV/EDF/MAT/TXT, parses signal, returns metadata + preview
- `POST /api/analyze/{file_id}` — runs inference (supports multi-channel EMG)
- `GET /api/models` — lists available models (ECG/EEG/EMG)
- `GET /api/health` — health check
- `POST /api/train/upload` — upload labeled training dataset
- `POST /api/train/start` — start custom CNN training job
- `WebSocket /api/train/ws/{job_id}` — real-time training metrics
- `GET /api/train/{job_id}/confusion_matrix` — post-training analysis
- `GET /api/train/{job_id}/tsne` — feature visualization
- `GET /api/train/{job_id}/export/*` — model/report/data export

### Frontend (Vanilla JS + Plotly.js)
- 4-step inference SPA: Upload → Signal Preview → Model Select → Results
- 5-phase training console: Upload → Configure → Train (live) → Analyze → Export
- Dark sci-fi theme, bilingual (EN/CN), responsive layout

### Model Architectures

**ECGArrhythmiaCNN** — 3 conv blocks, 44K params, input (1, 187)
**EEGSleepCNN** — 4 conv blocks, ~150K params, input (1, 3000)
**EMGGestureCNN** — 4 conv blocks, 388K params, input (16, 80) multi-channel

### Key Architecture Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Model format | PyTorch `.pt` | Primary format for all models |
| EMG input | Multi-channel (16ch) | NinaPro DB5 uses 16 sEMG channels |
| EEG epochs | 30s @ 100Hz = 3000 samples | AASM standard for sleep staging |
| EMG windows | 400ms @ 200Hz = 80 samples | Standard for gesture recognition |
| Training viz | WebSocket streaming | Real-time feedback |

---

## File Structure
```
BioSignal-Platform/
├── backend/
│   ├── main.py                    # FastAPI app
│   ├── config.py                  # MODEL_REGISTRY (3 models), PREPROCESS_CONFIG
│   ├── routers/
│   │   ├── upload.py              # File upload & parsing
│   │   ├── analysis.py            # Model inference (multi-channel support)
│   │   ├── models.py              # Model listing
│   │   └── training.py            # Training pipeline API + WebSocket
│   ├── services/
│   │   ├── format_parser.py       # CSV/EDF/MAT parsing
│   │   ├── preprocess.py          # ECG/EEG/EMG preprocessing + multi-ch EMG
│   │   ├── predictor.py           # PyTorch/ONNX/Demo inference
│   │   ├── trainer.py             # CNN training engine (Signal1DCNN)
│   │   └── dataset_loader.py      # Labeled data parsing (CSV/ZIP)
│   └── models/
│       ├── ecg_arrhythmia_cnn.pt  # Trained ECG model (94.1%)
│       ├── emg_gesture_cnn.pt     # Trained EMG model (42.7%, NinaPro DB5)
│       └── README.md
├── frontend/
│   ├── index.html
│   ├── css/style.css              # Dark sci-fi theme
│   └── js/
│       ├── app.js                 # Lang toggle, section flow
│       ├── uploader.js            # Drag-drop upload
│       ├── visualizer.js          # Plotly time/FFT charts
│       ├── results.js             # Results + export
│       ├── trainer.js             # Training console UI
│       └── inference.js           # Browser-side inference
├── training/
│   ├── train_ecg_arrhythmia.py   # MIT-BIH training (94.1% acc)
│   ├── train_eeg_sleep.py        # Sleep-EDF training (in progress)
│   ├── train_emg_gesture.py      # NinaPro DB5 training (53-class, 42.7%)
│   └── export_onnx.py            # PyTorch → ONNX converter
├── sample_data/
├── requirements.txt
├── Dockerfile
├── PLAN.md                        # Architecture & roadmap
└── CLAUDE_SESSION.md              # ← THIS FILE
```

---

## Training Scripts

### Train EMG Gesture Model (NinaPro DB5)
```bash
# Requires: training/EMG testing Data/s{1..10}/*.mat (extract from zips)
python training/train_emg_gesture.py
# Output: backend/models/emg_gesture_cnn.pt
# 53 classes, 16ch input, ~42.7% accuracy on real data
```

### Train EEG Sleep Model (Sleep-EDF)
```bash
# Auto-downloads from PhysioNet via MNE (first run takes ~30min)
python training/train_eeg_sleep.py
# Output: backend/models/eeg_sleep_staging.pt
# 5 classes (W/N1/N2/N3/REM), 30s epochs @ 100Hz
```

### Train ECG Model (MIT-BIH)
```bash
# Auto-downloads from PhysioNet via wfdb
python training/train_ecg_arrhythmia.py
# Output: backend/models/ecg_arrhythmia_cnn.pt
# 5 classes (AAMI), 94.1% accuracy
```

---

## Where to Start Next Session

**Priority 1:** Improve EMG model accuracy (currently 42.7%)
- Try deeper architecture or attention mechanisms
- Per-subject normalization
- Data augmentation (time warp, noise, channel dropout)

**Priority 2:** Complete EEG sleep staging model training

**Priority 3:** User authentication + database integration

---

## Known Issues / Watch Out For
- **Route order**: Always register API routers BEFORE `app.mount()` static files in `main.py`
- **Model files are committed**: `.pt` files (~2.6MB total) are in the repo for Railway deployment
- **EMG multi-channel**: predictor.py handles both single-channel and 16-channel EMG input
- **NinaPro data**: `.mat` files in `training/EMG testing Data/` (not committed, ~400MB)
- **Sleep-EDF data**: Auto-downloaded by MNE to `training/data/sleep_edf/` (~2GB)
- **Training cache**: `.npz` caches in `training/data/ninapro/` — delete to re-process raw data
