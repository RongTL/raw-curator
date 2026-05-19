"""FastAPI app serving the review UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import cluster, decide, photo, queue, submit
from app.config import settings

app = FastAPI(title="raw-curator")

app.include_router(queue.router, prefix="/api/queue", tags=["queue"])
app.include_router(photo.router, prefix="/api/photo", tags=["photo"])
app.include_router(cluster.router, prefix="/api/cluster", tags=["cluster"])
app.include_router(decide.router, prefix="/api/decide", tags=["decide"])
app.include_router(submit.router, prefix="/api/submit", tags=["submit"])

# Raw preview/thumb files live in the bind-mounted cache dir.
app.mount("/cache", StaticFiles(directory=str(settings.cache)), name="cache")

# Static SPA assets bundled in the image.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR)), name="ui")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
