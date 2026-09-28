"""FastAPI app serving the control-center UI + review API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from app.api.deps import get_monitor, get_runner
from app.api.routes import cluster, decide, photo, pipeline, queue, system
from app.config import settings


class _NoStoreStatic(StaticFiles):
    """Static files served with ``Cache-Control: no-store``.

    The UI is CDN-React with no build step, so its module filenames never change.
    Without this a browser caches an old ``review.js`` across image rebuilds and
    the running UI silently lags the code. The assets are tiny, so never caching
    them costs nothing and keeps a rebuilt UI in sync.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


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
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Raw preview/thumb files live in the bind-mounted cache dir. check_dir=False:
# the dir exists in-container but not necessarily at import time elsewhere.
app.mount("/cache", StaticFiles(directory=str(settings.cache), check_dir=False), name="cache")

# Share JPEGs from export, so the review UI can show the enhanced result.
# check_dir=False: the dir appears only after the first export run.
app.mount(
    "/jpeg",
    StaticFiles(directory=str(settings.photos / "jpeg"), check_dir=False),
    name="jpeg",
)

# Static SPA assets bundled in the image. Served no-store (see _NoStoreStatic) so a
# rebuilt UI isn't masked by a browser-cached module.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", _NoStoreStatic(directory=str(_STATIC_DIR)), name="ui")


@app.get("/api/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})
