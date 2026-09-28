"""decisions: export_choice + keep_raw for the before/after export flow

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLS = [
    sa.Column("export_choice", sa.String(length=16), nullable=False, server_default="undecided"),
    sa.Column("keep_raw", sa.Integer(), nullable=False, server_default="1"),
]


def upgrade() -> None:
    for col in _COLS:
        op.add_column("decisions", col)


def downgrade() -> None:
    with op.batch_alter_table("decisions") as batch:
        for col in reversed(_COLS):
            batch.drop_column(col.name)
