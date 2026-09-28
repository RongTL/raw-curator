"""UI static assets are served no-store so a rebuilt UI isn't masked by a browser-cached module."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.main import _NoStoreStatic


def test_ui_assets_served_no_store(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "app.js").write_text("export const x = 1;\n")
    app = FastAPI()
    app.mount("/ui", _NoStoreStatic(directory=str(static)), name="ui")
    with TestClient(app) as client:
        resp = client.get("/ui/app.js")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-store"
