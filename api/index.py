"""
Vercel serverless entry point for the BioSpark FastAPI app.

Vercel expects a WSGI/ASGI app exposed as `app` in api/index.py.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `backend.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.main import app  # noqa: F401 — Vercel picks up `app`
