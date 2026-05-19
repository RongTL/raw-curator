"""Centralized configuration. All knobs come from environment variables (.env)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    cpu_workers: int = 4

    preview_long_edge: int = 3000
    thumb_long_edge: int = 512
    jpeg_quality_preview: int = 92
    jpeg_quality_thumb: int = 88

    clip_batch: int = 8
    iqa_batch: int = 1

    burst_seconds: int = 2
    phash_hamming_threshold: int = 8
    clip_cosine_threshold: float = 0.92

    enhance_ai_scale: float = 0.4  # 24MP -> ~2.4kx1.6k for SCUNet, fits 6 GB
    enhance_target_res: str = "native"
    enhance_denoise: bool = True
    enhance_face_restore: bool = True
    enhance_codeformer_w: float = 0.7
    enhance_out_format: str = "tiff16"

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
