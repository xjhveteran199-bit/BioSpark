# Deploy BioSpark to Vercel

## Overview

BioSpark can be deployed to Vercel for frontend-only hosting. All model inference runs in the browser using **ONNX Runtime Web** — no backend server required!

## Prerequisites

- Node.js 18+ 
- Python 3.10+ with PyTorch (for ONNX export)
- Vercel CLI (`npm i -g vercel`)

## Step 1: Export Model to ONNX

The pre-trained PyTorch model needs to be converted to ONNX format for browser inference:

```bash
cd BioSpark

# Install dependencies
pip install torch onnx onnxruntime

# Export model (creates both backend/models/.onnx and frontend/models/.onnx)
python training/export_onnx.py
```

This generates `frontend/models/ecg_arrhythmia_cnn.onnx` which will be deployed to Vercel.

## Step 2: Test Locally

Before deploying, test the inference page:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8080

# Open http://localhost:8080/inference_test.html
```

## Step 3: Deploy to Vercel

### Option A: Vercel CLI

```bash
# Login to Vercel
vercel login

# Deploy (from project root)
vercel

# For production deployment
vercel --prod
```

### Option B: GitHub Integration

1. Push your code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your repository
4. Configure:
   - Framework Preset: "Other"
   - Root Directory: `./`
   - Build Command: (leave empty)
   - Output Directory: `frontend`

## Project Structure for Vercel

```
BioSpark/
├── frontend/
│   ├── index.html          # Main app (requires backend)
│   ├── inference_test.html # Standalone inference (works on Vercel!)
│   ├── models/
│   │   └── ecg_arrhythmia_cnn.onnx  # ONNX model for browser inference
│   ├── js/
│   │   ├── inference.js    # ONNX Runtime Web + preprocessing
│   │   └── ...
│   └── css/
├── vercel.json             # Vercel configuration
└── ...
```

## How It Works

### Browser-Based Inference

1. User uploads CSV file in the browser
2. `inference.js` performs preprocessing:
   - Bandpass filter (Butterworth)
   - Z-score normalization
   - R-peak detection & segmentation
3. ONNX Runtime Web runs the CNN model entirely in the browser
4. Results displayed inline — **no backend API calls!**

### Limitations

- **Demo mode fallback**: If ONNX model file isn't available, a feature-based demo predictor is used
- **Single signal type**: Currently only ECG arrhythmia detection works in browser
- **No file format auto-detection**: EDF/MAT parsing not implemented in browser (CSV only)

### Full-Stack Alternative

For the complete app with all signal types (ECG, EEG, EMG) and backend processing:

```bash
# Run locally with backend
docker compose up

# Or deploy backend to Railway/Render
# See README.md for details
```

## Troubleshooting

### "ONNX model not available"
- Make sure `frontend/models/ecg_arrhythmia_cnn.onnx` exists
- Check browser console for loading errors
- The app will fall back to demo mode if model fails to load

### "CORS errors"
- Vercel serves all files from same origin — no CORS issues
- If testing locally with a backend, ensure backend allows cross-origin requests

### Model file too large
- ONNX model is ~350KB (compressed)
- Vercel free tier serves files up to 4.5MB per route
- Should work without issues

## File Sizes

| File | Size | Notes |
|------|------|-------|
| `ecg_arrhythmia_cnn.onnx` | ~350KB | Compressed ONNX model |
| `ort.min.js` | ~1.5MB | ONNX Runtime Web (loaded from CDN) |
| `inference.js` | ~14KB | Our preprocessing + inference code |

Total: ~2MB for browser inference (plus CDN cache benefits)
