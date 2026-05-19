"""Centralized configuration. All knobs come from environment variables (.env)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cpu_workers() -> int:
    """Default to the host's logical core count so ingest/filter/export
    saturate all SMT threads. Falls back to 4 if the OS won't tell us."""
    return os.cpu_count() or 4


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAWCURATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    photos: Path = Field(default=Path("/data/photos"))
    cache: Path = Field(default=Path("/data/cache"))
    models: Path = Field(default=Path("/data/models"))
    xmp: Path = Field(default=Path("/data/xmp"))

    cpu_workers: int = Field(default_factory=_default_cpu_workers)

    preview_long_edge: int = 3000
    thumb_long_edge: int = 512
    jpeg_quality_preview: int = 92
    jpeg_quality_thumb: int = 88

    clip_batch: int = 8
    iqa_batch: int = 1

    burst_seconds: int = 2
    phash_hamming_threshold: int = 8
    clip_cosine_threshold: float = 0.92

    enhance_ai_scale: float = 0.7  # 24MP -> ~4.2kx2.8k for SCUNet, peaks ~5.5 GB on 6 GB cards
    enhance_target_res: str = "native"
    enhance_denoise: bool = True
    enhance_face_restore: bool = True
    enhance_codeformer_w: float = 0.7
    enhance_out_format: str = "tiff16"

    jpeg_quality: int = 92
    jpeg_long_edge: int = 0  # 0 = native resolution; e.g. 4000 to cap for sharing
    jpeg_progressive: bool = True
    jpeg_subdir: str = "jpeg"

    @property
    def db_path(self) -> Path:
        return self.cache / "session.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def previews_dir(self) -> Path:
        return self.cache / "previews"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache / "thumbs"


settings = Settings()
