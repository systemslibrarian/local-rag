"""files: add chunk_count column

Revision ID: a1b2c3d4e5f6
Revises: f3c1d9b2a4e7
Create Date: 2026-04-06 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f3c1d9b2a4e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("files", sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("files", "chunk_count")
