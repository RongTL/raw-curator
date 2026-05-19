"""ORM schema for the ephemeral session DB."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

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
    thumb_path: Mapped[Optional[str]] = mapped_column(Text)
    preview_path: Mapped[Optional[str]] = mapped_column(Text)
    file_kind: Mapped[Optional[str]] = mapped_column(String(8))

    camera_make: Mapped[Optional[str]] = mapped_column(String(64))
    camera_body: Mapped[Optional[str]] = mapped_column(String(128))
    lens: Mapped[Optional[str]] = mapped_column(String(128))
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    iso: Mapped[Optional[int]] = mapped_column(Integer)
    shutter: Mapped[Optional[float]] = mapped_column(Float)
    aperture: Mapped[Optional[float]] = mapped_column(Float)
    focal_length: Mapped[Optional[float]] = mapped_column(Float)
    orientation: Mapped[Optional[int]] = mapped_column(Integer)

    blur_var: Mapped[Optional[float]] = mapped_column(Float)
    phash: Mapped[Optional[str]] = mapped_column(String(32))
    dhash: Mapped[Optional[str]] = mapped_column(String(32))
    exposure_flag: Mapped[Optional[str]] = mapped_column(String(32))
    hist_mean: Mapped[Optional[float]] = mapped_column(Float)

    aesthetic_score: Mapped[Optional[float]] = mapped_column(Float)
    technical_score: Mapped[Optional[float]] = mapped_column(Float)
    musiq_score: Mapped[Optional[float]] = mapped_column(Float)
    maniqa_score: Mapped[Optional[float]] = mapped_column(Float)

    cluster_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("clusters.id", ondelete="SET NULL")
    )
    is_recommended: Mapped[bool] = mapped_column(Integer, default=0)

    faces: Mapped[list["Face"]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )
    cluster: Mapped[Optional["Cluster"]] = relationship(back_populates="photos")
    decision: Mapped[Optional["Decision"]] = relationship(
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
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    group_id: Mapped[Optional[int]] = mapped_column(Integer)

    photo: Mapped[Photo] = relationship(back_populates="faces")


Index("ix_faces_photo", Face.photo_hash)


class Cluster(Base):
    __tablename__ = "clusters"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(128))
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
    rank: Mapped[Optional[int]] = mapped_column(Integer)
    score: Mapped[Optional[float]] = mapped_column(Float)
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
    note: Mapped[Optional[str]] = mapped_column(Text)

    photo: Mapped[Photo] = relationship(back_populates="decision")


Index("ix_decisions_applied", Decision.applied)
