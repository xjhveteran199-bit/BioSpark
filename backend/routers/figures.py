"""
Figures router — publication-quality figure generation and download.

All endpoints require a completed training job and return either
PNG (300 DPI) or SVG figures styled for Nature, IEEE, or Science journals.
"""

import asyncio
import io
import zipfile
from functools import partial

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter()

# Lazy imports to avoid loading torch/matplotlib at startup
_trainer_module = None
_figures_module = None


def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        from backend.services import trainer as _mod
        _trainer_module = _mod
    return _trainer_module


def _get_figures():
    global _figures_module
    if _figures_module is None:
        from backend.services import publication_figures as _mod
        _figures_module = _mod
    return _figures_module


def _require_completed_job(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    return job


def _media_type(fmt: str) -> str:
    return "image/svg+xml" if fmt == "svg" else "image/png"


async def _run_in_executor(fn, *args):
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# Figure endpoints
# ---------------------------------------------------------------------------

@router.get("/train/{job_id}/figures/training_curves")
async def get_training_curves_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality training curves (loss + accuracy)."""
    job = _require_completed_job(job_id)
    if not job.history:
        raise HTTPException(status_code=409, detail="No training history available.")

    figs = _get_figures()
    data = await _run_in_executor(figs.render_training_curves, job.history, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="training_curves_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/confusion_matrix")
async def get_confusion_matrix_figure(
    job_id: str,
    mode: str = Query("both", pattern="^(count|normalized|both)$"),
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality confusion matrix heatmap."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    cm_data = trainer.compute_confusion_matrix(job)
    data = await _run_in_executor(figs.render_confusion_matrix, cm_data, mode, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="confusion_matrix_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/tsne")
async def get_tsne_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality t-SNE scatter plot."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    tsne_data = trainer.compute_tsne(job)
    data = await _run_in_executor(figs.render_tsne, tsne_data, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="tsne_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/per_class_metrics")
async def get_per_class_metrics_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality per-class Precision/Recall/F1 bar chart."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    cm_data = trainer.compute_confusion_matrix(job)
    data = await _run_in_executor(figs.render_per_class_metrics, cm_data, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="per_class_metrics_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/architecture")
async def get_architecture_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return model architecture diagram."""
    job = _require_completed_job(job_id)
    figs = _get_figures()

    if job.model is None:
        raise HTTPException(status_code=409, detail="Model not available.")

    # Reconstruct input shape from job metadata
    input_shape = (job.n_channels, job.val_X.shape[1] // max(job.n_channels, 1))
    data = await _run_in_executor(
        figs.render_architecture_diagram, job.model, input_shape, style, fmt,
    )

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="architecture_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/all.zip")
async def download_all_figures(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
):
    """Download all 5 figures in both PNG and SVG formats as a ZIP archive."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    cm_data = trainer.compute_confusion_matrix(job)
    tsne_data = trainer.compute_tsne(job)
    input_shape = (job.n_channels, job.val_X.shape[1] // max(job.n_channels, 1))

    # Generate all figures
    figure_specs = [
        ("training_curves", figs.render_training_curves, [job.history, style]),
        ("confusion_matrix", figs.render_confusion_matrix, [cm_data, "both", style]),
        ("tsne", figs.render_tsne, [tsne_data, style]),
        ("per_class_metrics", figs.render_per_class_metrics, [cm_data, style]),
        ("architecture", figs.render_architecture_diagram, [job.model, input_shape, style]),
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, render_fn, args in figure_specs:
            for fmt in ("png", "svg"):
                img_bytes = await _run_in_executor(render_fn, *args, fmt)
                zf.writestr(f"{name}.{fmt}", img_bytes)

        # README
        zf.writestr("README.txt", (
            f"BioSpark Publication Figures\n"
            f"Job ID: {job_id}\n"
            f"Style: {style}\n"
            f"Best Val Accuracy: {job.best_val_acc:.4f}\n\n"
            f"Files:\n"
            f"  training_curves.png/svg  — Loss + Accuracy training curves\n"
            f"  confusion_matrix.png/svg — Confusion matrix heatmap (count + normalized)\n"
            f"  tsne.png/svg             — t-SNE feature space visualization\n"
            f"  per_class_metrics.png/svg— Per-class Precision / Recall / F1 bar chart\n"
            f"  architecture.png/svg     — Model architecture diagram\n\n"
            f"Generated by BioSpark — AI-Powered Biosignal Analysis Platform\n"
        ))

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="biospark_figures_{job_id}.zip"'},
    )
