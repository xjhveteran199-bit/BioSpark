# Plan: BioSignal Analysis Web Platform

## Context

Build a web platform where users can upload multi-channel wearable biosignal data (ECG, EMG, EEG), run pre-trained deep learning models, and get analysis results — without needing to build models themselves.

---

## Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Backend | FastAPI (Python 3.10+) | Async, auto OpenAPI docs, native Python ML ecosystem |
| Preprocessing | NeuroKit2 + MNE-Python | ECG/EMG via NeuroKit2, EEG via MNE |
| Model Serving | ONNX Runtime | Cross-framework, lightweight, fast inference |
| Frontend | Vanilla HTML/CSS/JS + Plotly.js | Simple, no build step needed |
| Signal Viz | Plotly.js | Interactive zoom/pan, multi-channel support |
| File Formats | CSV, EDF, MAT | Via pandas, pyedflib, scipy.io |

## Pre-trained Model Zoo (MVP)

| Task | Signal | Model | Params | Dataset | Output |
|------|--------|-------|--------|---------|--------|
| Arrhythmia Detection | ECG | 1D-CNN | ~100K | MIT-BIH | 5-class: N, S, V, F, Q |
| 12-lead Classification | ECG | ResNet-18 1D | ~5M | PTB-XL | Multi-label |
| Sleep Staging | EEG | U-Sleep / EEGNet | ~30K-1M | Sleep-EDF | 5-class: W, N1, N2, N3, REM |
| Emotion Recognition | EEG | CNN-LSTM | ~500K | DEAP | Valence/Arousal |
| Gesture Recognition | EMG | 1D-CNN | ~200K | NinaPro DB1 | 52 gesture classes |

## User Flow

```
Upload file → Format auto-detect → Signal preview (Plotly)
→ Select analysis task → Preprocessing (auto) → Model inference
→ Results + confidence + annotated signal → Download report
```
