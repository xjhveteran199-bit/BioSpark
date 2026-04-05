# Plan: BioSignal Analysis Web Platform

## Context

Build a web platform where users can upload multi-channel wearable biosignal data (ECG, EMG, EEG), run pre-trained deep learning models, and get analysis results — without needing to build models themselves. Users can also train custom CNN models on their own labeled data.

---

## Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Backend | FastAPI (Python 3.10+) | Async, auto OpenAPI docs, native Python ML ecosystem |
| Deep Learning | PyTorch 2.x (CPU) | 1D-CNN training & inference |
| Preprocessing | NeuroKit2 + MNE-Python + SciPy | ECG/EMG via NeuroKit2, EEG via MNE |
| Model Serving | PyTorch (.pt) + ONNX Runtime | PyTorch primary, ONNX for browser |
| Frontend | Vanilla HTML/CSS/JS + Plotly.js | Simple, no build step needed |
| Signal Viz | Plotly.js | Interactive zoom/pan, multi-channel support |
| File Formats | CSV, EDF, MAT, TXT | Via pandas, pyedflib, scipy.io |
| Deployment | Docker / Railway / Vercel | Container + Serverless dual mode |

## Pre-trained Model Zoo

| Task | Signal | Model | Params | Dataset | Classes | Accuracy | Status |
|------|--------|-------|--------|---------|---------|----------|--------|
| Arrhythmia Detection | ECG | 1D-CNN (3 blocks) | 44K | MIT-BIH | 5: N/S/V/F/Q | 94.1% | **DEPLOYED** |
| Sleep Staging | EEG | 1D-CNN (4 blocks) | ~150K | Sleep-EDF | 5: W/N1/N2/N3/REM | Training | **IN PROGRESS** |
| Gesture Recognition | EMG | Multi-ch 1D-CNN (4 blocks) | 388K | NinaPro DB5 | 53: 52 gestures + Rest | 42.7% | **DEPLOYED** |

## User Flow

```
Upload file -> Format auto-detect -> Signal preview (Plotly)
-> Select analysis task -> Preprocessing (auto) -> Model inference
-> Results + confidence + annotated signal -> Download report

OR

Upload labeled data -> Configure CNN training -> Real-time monitoring (WebSocket)
-> Confusion matrix + t-SNE -> Export model/report
```

---

## Version History

| Version | Date | Content |
|---------|------|---------|
| v0.1 | 2026-04-01 | File upload, signal viz, ECG inference |
| v0.2 | 2026-04-02 | 5-phase training pipeline, WebSocket, export |
| v0.2.1 | 2026-04-05 | Railway/Vercel dual deployment |
| v0.2.2 | 2026-04-05 | Dark sci-fi UI theme, Plotly dark mode |
| **v0.3** | **2026-04-06** | **EEG sleep staging model + EMG gesture model (NinaPro DB5, 53-class)** |

---

## Next Steps (Priority Order)

### Phase 1: Model Optimization (Current)
- [ ] EEG Sleep Staging model — complete training on Sleep-EDF (in progress)
- [ ] EMG model accuracy improvement — deeper architecture, data augmentation, per-subject fine-tuning
- [ ] Attention heatmap (Grad-CAM) — visualize CNN focus regions

### Phase 2: User System + Database (1-2 months)
- [ ] JWT authentication (login/register)
- [ ] PostgreSQL integration — persist users, models, training history
- [ ] User-specific model storage

### Phase 3: Platform Ecosystem (3-6 months)
- [ ] Pre-trained model marketplace
- [ ] HuggingFace integration (ECGFounder, U-Sleep)
- [ ] Auto hyperparameter search (Grid/Random/Bayesian)
- [ ] Data augmentation (time warp, noise injection, sliding window)
- [ ] Team collaboration spaces

### Phase 4: Real-time + Commercialization (6-12 months)
- [ ] Real-time signal stream inference (WebSocket from wearable devices)
- [ ] Federated learning
- [ ] FDA/CE compliance
- [ ] Mobile SDK (iOS/Android)
- [ ] Edge inference (TensorRT/CoreML)
