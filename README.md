<p align="center">
  <img src="https://img.shields.io/badge/BioSpark-v0.1-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHJ4PSI4IiBmaWxsPSIjMjU2M2ViIi8+PHBhdGggZD0iTTQgMTYgTDggMTYgTDEwIDEwIEwxMyAyMiBMMTYgNiBMMTkgMjIgTDIyIDEwIEwyNCAxNiBMMjggMTYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+" alt="BioSpark" />
</p>

<h1 align="center">BioSpark</h1>

<p align="center">
  <strong>Upload biosignal data. AI analyzes. Instant results.</strong><br>
  <sub>上传生物信号数据 · AI智能分析 · 即时结果</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ECG-心电图-red?style=flat-square" alt="ECG" />
  <img src="https://img.shields.io/badge/EEG-脑电图-blue?style=flat-square" alt="EEG" />
  <img src="https://img.shields.io/badge/EMG-肌电图-green?style=flat-square" alt="EMG" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Plotly.js-Interactive-3F4F75?style=flat-square&logo=plotly" alt="Plotly" />
</p>

---

## What is BioSpark?

BioSpark is a web platform that lets you **upload multi-channel wearable biosignal data** (ECG, EEG, EMG) and get **AI-powered analysis results instantly** — no model building required.

> **No code. No training. Just upload and analyze.**

### Key Features

- **Drag & drop upload** — CSV, EDF, MAT, TXT formats with auto-detection
- **Interactive signal visualization** — Plotly.js with zoom, pan, time domain + FFT spectrum
- **Pre-trained DL models** — Real CNN trained on MIT-BIH (94.1% accuracy), with demo mode fallback
- **Multi-signal support** — ECG arrhythmia detection, EEG sleep staging, EMG gesture recognition
- **Bilingual UI** — English / 中文 with one-click toggle
- **Export results** — JSON and CSV export with per-segment predictions
- **Docker ready** — One command to deploy anywhere

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/BioSpark.git
cd BioSpark
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Train a model (optional — demo mode works without this)

```bash
python training/train_ecg_arrhythmia.py
```

This downloads MIT-BIH data from PhysioNet and trains a 1D-CNN (~2 min on CPU).

### 3. Run

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** and start analyzing!

### Docker

```bash
docker compose up --build
```

---

## How It Works

```
Upload File → Auto-detect Format & Signal Type → Interactive Preview
    → Select AI Model → Preprocessing (bandpass filter, normalize, segment)
    → Model Inference (PyTorch CNN) → Results + Confidence Scores
    → Export JSON/CSV
```

### Architecture

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI + Python 3.10+ |
| **Preprocessing** | NeuroKit2 + SciPy (bandpass filter, R-peak detection, segmentation) |
| **Model Serving** | PyTorch (CPU) with ONNX Runtime fallback |
| **Frontend** | Vanilla HTML/CSS/JS + Plotly.js |
| **Deployment** | Docker + docker-compose |

### Pre-trained Models

| Task | Signal | Architecture | Accuracy | Dataset |
|------|--------|-------------|----------|---------|
| Arrhythmia Detection | ECG | 1D-CNN (44K params) | **94.1%** | MIT-BIH |
| Sleep Staging | EEG | CNN (coming soon) | — | Sleep-EDF |
| Gesture Recognition | EMG | CNN (coming soon) | — | NinaPro |

---

## Project Structure

```
BioSpark/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Model registry & settings
│   ├── routers/             # API endpoints (upload, analysis, models)
│   ├── services/            # Core logic (parser, preprocessor, predictor)
│   ├── models/              # Trained model weights (.pt)
│   └── tests/               # Unit tests
├── frontend/
│   ├── index.html           # SPA with bilingual UI
│   ├── css/style.css        # Responsive design
│   └── js/                  # Upload, visualization, results, app logic
├── training/
│   └── train_ecg_arrhythmia.py   # MIT-BIH training script
├── sample_data/             # Demo CSV files (ECG, EEG, EMG)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload biosignal file |
| `POST` | `/api/analyze/{file_id}` | Run model inference |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/files/{file_id}` | Get file metadata |
| `GET` | `/api/health` | Health check |

Full API docs at **http://localhost:8000/docs** (Swagger UI).

---

## Supported Formats

| Format | Extension | Library | Notes |
|--------|-----------|---------|-------|
| CSV | `.csv`, `.txt` | pandas | Auto-detect time/signal columns |
| EDF/BDF | `.edf`, `.bdf` | pyedflib | Standard EEG/PSG format |
| MATLAB | `.mat` | scipy.io | Auto-find largest numeric array |

---

## Roadmap

- [x] ECG arrhythmia detection (MIT-BIH, 94.1%)
- [x] Interactive signal visualization (time + FFT)
- [x] Bilingual UI (EN/中文)
- [x] Docker deployment
- [ ] EEG sleep staging model (Sleep-EDF)
- [ ] EMG gesture recognition model (NinaPro)
- [ ] Integrate HuggingFace models (ECGFounder, U-Sleep)
- [ ] Attention heatmap visualization
- [ ] Batch processing API
- [ ] Cloud deployment (Railway / Vercel)

---

## License

MIT

---

<p align="center">
  <sub>Built with FastAPI + PyTorch + Plotly.js</sub><br>
  <sub>从生物信号中点燃AI洞察 ⚡</sub>
</p>
