"""ORM schema for the ephemeral session DB."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SessionMeta(Base):
    __tablename__ = "session_meta"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp()
    )
    __table_args__ = (CheckConstraint("id = 1", name="session_meta_singleton"),)


class Photo(Base):
    __tablename__ = "photos"
    hash: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    thumb_path: Mapped[str | None] = mapped_column(Text)
    preview_path: Mapped[str | None] = mapped_column(Text)
    file_kind: Mapped[str | None] = mapped_column(String(8))

    camera_make: Mapped[str | None] = mapped_column(String(64))
    camera_body: Mapped[str | None] = mapped_column(String(128))
    lens: Mapped[str | None] = mapped_column(String(128))
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    iso: Mapped[int | None] = mapped_column(Integer)
    shutter: Mapped[float | None] = mapped_column(Float)
    aperture: Mapped[float | None] = mapped_column(Float)
    focal_length: Mapped[float | None] = mapped_column(Float)
    orientation: Mapped[int | None] = mapped_column(Integer)

    blur_var: Mapped[float | None] = mapped_column(Float)
    phash: Mapped[str | None] = mapped_column(String(32))
    dhash: Mapped[str | None] = mapped_column(String(32))
    exposure_flag: Mapped[str | None] = mapped_column(String(32))
    hist_mean: Mapped[float | None] = mapped_column(Float)

    aesthetic_score: Mapped[float | None] = mapped_column(Float)
    technical_score: Mapped[float | None] = mapped_column(Float)
    musiq_score: Mapped[float | None] = mapped_column(Float)
    maniqa_score: Mapped[float | None] = mapped_column(Float)

    cluster_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("clusters.id", ondelete="SET NULL")
    )
    is_recommended: Mapped[bool] = mapped_column(Integer, default=0)

    faces: Mapped[list[Face]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    cluster: Mapped[Cluster | None] = relationship(back_populates="photos")
    decision: Mapped[Decision | None] = relationship(
        back_populates="photo", uselist=False, cascade="all, delete-orphan"
    )
    quality_report: Mapped[PhotoQualityReport | None] = relationship(
        back_populates="photo", uselist=False, cascade="all, delete-orphan"
    )


Index("ix_photos_captured_at", Photo.captured_at)
Index("ix_photos_camera_captured", Photo.camera_body, Photo.captured_at)
Index("ix_photos_phash", Photo.phash)


class PhotoEmbedding(Base):
    __tablename__ = "photo_embeddings"
    photo_hash: Mapped[str] = mapped_column(
        String(32), ForeignKey("photos.hash", ondelete="CASCADE"), primary_key=True
    )
    dim: Mapped[int] = mapped_column(Integer, nullable=False, default=768)
    vec: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class Face(Base):
    __tablename__ = "faces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    photo_hash: Mapped[str] = mapped_column(
        String(32), ForeignKey("photos.hash", ondelete="CASCADE"), nullable=False
    )
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    det_score: Mapped[float] = mapped_column(Float, nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    group_id: Mapped[int | None] = mapped_column(Integer)

    photo: Mapped[Photo] = relationship(back_populates="faces")


Index("ix_faces_photo", Face.photo_hash)


class Cluster(Base):
    __tablename__ = "clusters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128))
    size: Mapped[int] = mapped_column(Integer, default=0)

    photos: Mapped[list[Photo]] = relationship(back_populates="cluster")


class ClusterMember(Base):
    __tablename__ = "cluster_members"
    cluster_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clusters.id", ondelete="CASCADE"), primary_key=True
    )
    photo_hash: Mapped[str] = mapped_column(
        String(32), ForeignKey("photos.hash", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("cluster_id", "photo_hash"),)


class Decision(Base):
    __tablename__ = "decisions"
    photo_hash: Mapped[str] = mapped_column(
        String(32), ForeignKey("photos.hash", ondelete="CASCADE"), primary_key=True
    )
    selected: Mapped[str] = mapped_column(String(16), nullable=False, default="undecided")
    score_tier: Mapped[str] = mapped_column(String(16), nullable=False, default="unset")
    stars: Mapped[int] = mapped_column(Integer, default=0)
    favorite: Mapped[bool] = mapped_column(Integer, default=0)
    enhance_requested: Mapped[bool] = mapped_column(Integer, default=0)
    action: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    applied: Mapped[bool] = mapped_column(Integer, default=0)
    note: Mapped[str | None] = mapped_column(Text)

    photo: Mapped[Photo] = relationship(back_populates="decision")


Index("ix_decisions_applied", Decision.applied)


class PhotoQualityReport(Base):
    """Per-photo Auto Enhancement Engine measurement + scores.

    Populated by `app/enhancement/enhance_job.py` after each measure→score
    pass. Wiped by `make reset` (which drops the DB file and re-runs
    `alembic upgrade head`).
    """

    __tablename__ = "quality_reports"
    photo_hash: Mapped[str] = mapped_column(
        String(32), ForeignKey("photos.hash", ondelete="CASCADE"), primary_key=True
    )

    # §1 Exposure
    mean_luma: Mapped[float] = mapped_column(Float, nullable=False)
    shadow_clip: Mapped[float] = mapped_column(Float, nullable=False)
    highlight_clip: Mapped[float] = mapped_column(Float, nullable=False)
    midtone_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    midtone_deviation: Mapped[float] = mapped_column(Float, nullable=False)

    # §2 Dynamic Range
    dr_p95_p5: Mapped[float] = mapped_column(Float, nullable=False)
    local_dr_mean: Mapped[float] = mapped_column(Float, nullable=False)

    # §3 Color
    rg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    bg_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    avg_saturation: Mapped[float] = mapped_column(Float, nullable=False)
    oversat_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    skin_hue_var: Mapped[float | None] = mapped_column(Float)

    # §4 Sharpness
    lap_var: Mapped[float] = mapped_column(Float, nullable=False)
    edge_density: Mapped[float] = mapped_column(Float, nullable=False)
    hf_energy: Mapped[float] = mapped_column(Float, nullable=False)

    # §5 Noise
    luma_noise: Mapped[float] = mapped_column(Float, nullable=False)
    chroma_noise: Mapped[float] = mapped_column(Float, nullable=False)

    # §1.4 / §6 sub-scores + Q (all 0..100)
    score_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    score_dynamic_range: Mapped[float] = mapped_column(Float, nullable=False)
    score_color: Mapped[float] = mapped_column(Float, nullable=False)
    score_sharpness: Mapped[float] = mapped_column(Float, nullable=False)
    score_noise: Mapped[float] = mapped_column(Float, nullable=False)
    score_q: Mapped[float] = mapped_column(Float, nullable=False)

    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.current_timestamp()
    )

    photo: Mapped[Photo] = relationship(back_populates="quality_report")


Index("ix_quality_reports_q", PhotoQualityReport.score_q)
