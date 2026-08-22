"""
Main Starlette application entry point for SmartReconcile AI.
"""

import os
from pathlib import Path
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from app.core.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION, DATA_DIR
from app.api.routes import (
    upload_invoice_file,
    list_staged_drafts,
    get_staged_draft,
    review_staged_line,
    commit_staged_draft,
    list_master_pos,
    list_demo_scenarios,
    get_demo_scenario,
    run_reconciliation,
    submit_human_review,
    run_batch_simulation_benchmark
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "ui" / "static"
TEMPLATES_DIR = BASE_DIR / "ui" / "templates"
DEMO_PDF_DIR = DATA_DIR / "demo_invoices_pdf"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


async def index(request: Request):
    """
    Renders the main interactive AP reconciliation interface.
    """
    html_path = TEMPLATES_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


async def health_check(request: Request):
    return JSONResponse({
        "status": "healthy",
        "app": APP_TITLE,
        "version": APP_VERSION
    })


routes = [
    Route("/", endpoint=index, methods=["GET"]),
    Route("/health", endpoint=health_check, methods=["GET"]),
    
    # Ingestion, Staging, and Commit workflow
    Route("/api/upload", endpoint=upload_invoice_file, methods=["POST"]),
    Route("/api/staging", endpoint=list_staged_drafts, methods=["GET"]),
    Route("/api/staging/{staging_id}", endpoint=get_staged_draft, methods=["GET"]),
    Route("/api/staging/{staging_id}/review", endpoint=review_staged_line, methods=["POST"]),
    Route("/api/staging/{staging_id}/commit", endpoint=commit_staged_draft, methods=["POST"]),
    Route("/api/pos", endpoint=list_master_pos, methods=["GET"]),

    # Fallback & simulation demo endpoints
    Route("/api/scenarios", endpoint=list_demo_scenarios, methods=["GET"]),
    Route("/api/scenarios/{invoice_id}", endpoint=get_demo_scenario, methods=["GET"]),
    Route("/api/reconcile", endpoint=run_reconciliation, methods=["POST"]),
    Route("/api/review", endpoint=submit_human_review, methods=["POST"]),
    Route("/api/benchmark/run-all", endpoint=run_batch_simulation_benchmark, methods=["POST"]),
    
    # Static assets
    Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    # Read-only source PDFs for the Upload tab's "Quick Demo" cards — served so
    # the browser can fetch the real file and upload it through /api/upload,
    # instead of faking extraction via the /api/scenarios JSON shortcut.
    Mount("/demo-pdfs", app=StaticFiles(directory=str(DEMO_PDF_DIR)), name="demo-pdfs"),
]

DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"

app = Starlette(debug=DEBUG, routes=routes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
