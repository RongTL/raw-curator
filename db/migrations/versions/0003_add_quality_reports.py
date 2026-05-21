"""add quality_reports table for the Auto Enhancement Engine

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "quality_reports",
        sa.Column("photo_hash", sa.String(32), nullable=False),
        # §1 Exposure
        sa.Column("mean_luma", sa.Float, nullable=False),
        sa.Column("shadow_clip", sa.Float, nullable=False),
        sa.Column("highlight_clip", sa.Float, nullable=False),
        sa.Column("midtone_ratio", sa.Float, nullable=False),
        sa.Column("midtone_deviation", sa.Float, nullable=False),
        # §2 Dynamic Range
        sa.Column("dr_p95_p5", sa.Float, nullable=False),
        sa.Column("local_dr_mean", sa.Float, nullable=False),
        # §3 Color
        sa.Column("rg_ratio", sa.Float, nullable=False),
        sa.Column("bg_ratio", sa.Float, nullable=False),
        sa.Column("avg_saturation", sa.Float, nullable=False),
        sa.Column("oversat_ratio", sa.Float, nullable=False),
        sa.Column("skin_hue_var", sa.Float, nullable=True),
        # §4 Sharpness
        sa.Column("lap_var", sa.Float, nullable=False),
        sa.Column("edge_density", sa.Float, nullable=False),
        sa.Column("hf_energy", sa.Float, nullable=False),
        # §5 Noise
        sa.Column("luma_noise", sa.Float, nullable=False),
        sa.Column("chroma_noise", sa.Float, nullable=False),
        # §1.4 / §6 sub-scores
        sa.Column("score_exposure", sa.Float, nullable=False),
        sa.Column("score_dynamic_range", sa.Float, nullable=False),
        sa.Column("score_color", sa.Float, nullable=False),
        sa.Column("score_sharpness", sa.Float, nullable=False),
        sa.Column("score_noise", sa.Float, nullable=False),
        sa.Column("score_q", sa.Float, nullable=False),
        sa.Column(
            "measured_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.current_timestamp(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("photo_hash"),
        sa.ForeignKeyConstraint(
            ["photo_hash"], ["photos.hash"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_quality_reports_q", "quality_reports", ["score_q"]
    )


def downgrade() -> None:
    op.drop_index("ix_quality_reports_q", table_name="quality_reports")
    op.drop_table("quality_reports")
