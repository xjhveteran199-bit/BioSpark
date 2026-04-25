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

## Railway 部署须知 / Railway Deployment Notes

> **必读**：默认 SQLite 数据库会被 Railway 容器重启擦除 → 注册的账号下次登录会显示「账号不存在」。生产部署**必须**挂 PostgreSQL 插件。

> **Important**: BioSpark uses SQLite by default, but Railway's container filesystem is **ephemeral** — every redeploy or auto-restart wipes the local DB file. User accounts created today will be gone tomorrow. **You must attach a PostgreSQL plugin** for production.

**5 步 / 5-step setup**:

1. Open your Railway project dashboard.
2. Click **`+ New`** → **`Database`** → **`Add PostgreSQL`**.
3. Wait ~30 seconds for the plugin to provision. Railway automatically injects `DATABASE_URL` into your web service.
4. Trigger a redeploy (push a commit, or **Deployments → Redeploy**).
5. Verify in Railway logs: you should see
   ```
   DB engine initialized: driver=postgresql+asyncpg ...
   DB self-check: users table count = N
   ```
   If you see `driver=sqlite+aiosqlite` instead, the env var is not set — recheck the plugin attachment.

The code (`backend/database.py`) auto-rewrites `postgres://` → `postgresql+asyncpg://`, so no code changes are needed.

### CPU 训练时长 / CPU Training Performance Notes

Railway hobby tier runs on a **shared vCPU** — there is no GPU. Realistic training times:

| Preset | Sample size | Expected time |
|--------|-------------|---------------|
| Quick Test | < 1k | 30s – 2 min |
| Smart Auto | 1k – 5k | 3 – 10 min |
| Publication Ready | 5k – 20k | 15 – 45 min |

For datasets > 5k samples, we recommend training **locally** (or on a beefier instance) and uploading the resulting checkpoint via the warm-start mechanism.

**Tuning knobs** (env vars):
- `BIOSPARK_DATALOADER_WORKERS` (default `0`, **forced 0 on Railway**) — set to `2` or `4` on a beefy local multi-core box for ~15-30% epoch speedup. Higher values are ignored when `RAILWAY_ENVIRONMENT` is set, because `fork()`ing workers in a 512 MB-1 GB container causes OOM kills.
- The "Search optimal learning rate" advanced option adds **2-3 minutes** of LR range testing — disabled by default.

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
