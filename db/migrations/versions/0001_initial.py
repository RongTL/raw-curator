"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "session_meta",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="session_meta_singleton"),
    )

    op.create_table(
        "clusters",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("label", sa.String(128)),
        sa.Column("size", sa.Integer(), server_default="0"),
    )

    op.create_table(
        "photos",
        sa.Column("hash", sa.String(32), primary_key=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("thumb_path", sa.Text()),
        sa.Column("preview_path", sa.Text()),
        sa.Column("camera_make", sa.String(64)),
        sa.Column("camera_body", sa.String(128)),
        sa.Column("lens", sa.String(128)),
        sa.Column("captured_at", sa.DateTime(timezone=True)),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("iso", sa.Integer()),
        sa.Column("shutter", sa.Float()),
        sa.Column("aperture", sa.Float()),
        sa.Column("focal_length", sa.Float()),
        sa.Column("orientation", sa.Integer()),
        sa.Column("blur_var", sa.Float()),
        sa.Column("phash", sa.String(32)),
        sa.Column("dhash", sa.String(32)),
        sa.Column("exposure_flag", sa.String(32)),
        sa.Column("hist_mean", sa.Float()),
        sa.Column("aesthetic_score", sa.Float()),
        sa.Column("technical_score", sa.Float()),
        sa.Column("musiq_score", sa.Float()),
        sa.Column("maniqa_score", sa.Float()),
        sa.Column(
            "cluster_id",
            sa.Integer(),
            sa.ForeignKey("clusters.id", ondelete="SET NULL"),
        ),
        sa.Column("is_recommended", sa.Integer(), server_default="0"),
    )
    op.create_index("ix_photos_captured_at", "photos", ["captured_at"])
    op.create_index("ix_photos_camera_captured", "photos", ["camera_body", "captured_at"])
    op.create_index("ix_photos_phash", "photos", ["phash"])

    op.create_table(
        "photo_embeddings",
        sa.Column(
            "photo_hash",
            sa.String(32),
            sa.ForeignKey("photos.hash", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("dim", sa.Integer(), nullable=False, server_default="768"),
        sa.Column("vec", sa.LargeBinary(), nullable=False),
    )

    op.create_table(
        "faces",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "photo_hash",
            sa.String(32),
            sa.ForeignKey("photos.hash", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("bbox_x", sa.Integer(), nullable=False),
        sa.Column("bbox_y", sa.Integer(), nullable=False),
        sa.Column("bbox_w", sa.Integer(), nullable=False),
        sa.Column("bbox_h", sa.Integer(), nullable=False),
        sa.Column("det_score", sa.Float(), nullable=False),
        sa.Column("embedding", sa.LargeBinary()),
        sa.Column("group_id", sa.Integer()),
    )
    op.create_index("ix_faces_photo", "faces", ["photo_hash"])

    op.create_table(
        "cluster_members",
        sa.Column(
            "cluster_id",
            sa.Integer(),
            sa.ForeignKey("clusters.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "photo_hash",
            sa.String(32),
            sa.ForeignKey("photos.hash", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("rank", sa.Integer()),
        sa.Column("score", sa.Float()),
        sa.UniqueConstraint("cluster_id", "photo_hash"),
    )

    op.create_table(
        "decisions",
        sa.Column(
            "photo_hash",
            sa.String(32),
            sa.ForeignKey("photos.hash", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("selected", sa.String(16), nullable=False, server_default="undecided"),
        sa.Column("score_tier", sa.String(16), nullable=False, server_default="unset"),
        sa.Column("stars", sa.Integer(), server_default="0"),
        sa.Column("favorite", sa.Integer(), server_default="0"),
        sa.Column("enhance_requested", sa.Integer(), server_default="0"),
        sa.Column("action", sa.String(32), nullable=False, server_default="none"),
        sa.Column("applied", sa.Integer(), server_default="0"),
        sa.Column("note", sa.Text()),
    )
    op.create_index("ix_decisions_applied", "decisions", ["applied"])

    op.execute("INSERT INTO session_meta (id) VALUES (1)")


def downgrade() -> None:
    op.drop_table("decisions")
    op.drop_table("cluster_members")
    op.drop_table("faces")
    op.drop_table("photo_embeddings")
    op.drop_index("ix_photos_phash", table_name="photos")
    op.drop_index("ix_photos_camera_captured", table_name="photos")
    op.drop_index("ix_photos_captured_at", table_name="photos")
    op.drop_table("photos")
    op.drop_table("clusters")
    op.drop_table("session_meta")
