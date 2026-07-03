"""FastAPI app serving the control-center UI + review API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import get_monitor, get_runner
from app.api.routes import cluster, decide, photo, pipeline, queue, submit, system
from app.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Construct the singletons on the event loop before any threadpool request
    # can race the lru_cache factories, then start the 1 s stats sampler.
    get_runner()
    sampler = asyncio.create_task(get_monitor().run())
    yield
    sampler.cancel()


app = FastAPI(title="raw-curator", lifespan=lifespan)

app.include_router(queue.router, prefix="/api/queue", tags=["queue"])
app.include_router(photo.router, prefix="/api/photo", tags=["photo"])
app.include_router(cluster.router, prefix="/api/cluster", tags=["cluster"])
app.include_router(decide.router, prefix="/api/decide", tags=["decide"])
app.include_router(submit.router, prefix="/api/submit", tags=["submit"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Raw preview/thumb files live in the bind-mounted cache dir. check_dir=False:
# the dir exists in-container but not necessarily at import time elsewhere.
app.mount(
    "/cache", StaticFiles(directory=str(settings.cache), check_dir=False), name="cache"
)

# Static SPA assets bundled in the image.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR)), name="ui")


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
