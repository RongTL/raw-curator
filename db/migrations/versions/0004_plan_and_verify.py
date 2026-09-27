"""quality_reports: recipe, verification, and Step-3 metric columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLS = [
    sa.Column("plan_json", sa.Text(), nullable=True),
    sa.Column("verify_json", sa.Text(), nullable=True),
    sa.Column("score_q_after", sa.Float(), nullable=True),
    sa.Column("degraded", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("neutral_fraction", sa.Float(), nullable=True),
    sa.Column("rg_neutral", sa.Float(), nullable=True),
    sa.Column("bg_neutral", sa.Float(), nullable=True),
    sa.Column("mean_chroma", sa.Float(), nullable=True),
    sa.Column("lap_var_top", sa.Float(), nullable=True),
]


def upgrade() -> None:
    for col in _COLS:
        op.add_column("quality_reports", col)


def downgrade() -> None:
    with op.batch_alter_table("quality_reports") as batch:
        for col in reversed(_COLS):
            batch.drop_column(col.name)
