from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.routers import upload, analysis, models as models_router, training

app = FastAPI(
    title="BioSpark",
    description="Upload biosignal data (ECG/EEG/EMG), run pre-trained DL models, get predictions.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API routes (must be registered BEFORE static file mount) ---

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(models_router.router, prefix="/api", tags=["Models"])
app.include_router(training.router, prefix="/api", tags=["Training"])

# --- Serve frontend static assets (css/js/assets) under /static ---
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/css", StaticFiles(directory=str(frontend_dir / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend_dir / "js")), name="js")
    if (frontend_dir / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

    # Serve index.html for the root path (SPA fallback)
    @app.get("/")
    async def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))
