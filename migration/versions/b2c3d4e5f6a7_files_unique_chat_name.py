"""files: add unique constraint on (chat_id, name)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-06 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove duplicate (chat_id, name) rows, keeping the oldest file
    conn = op.get_bind()
    conn.execute(sa.text("""
        DELETE FROM files
        WHERE id NOT IN (
            SELECT DISTINCT ON (chat_id, name) id
            FROM files
            ORDER BY chat_id, name, created_at ASC
        )
    """))
    op.create_unique_constraint("uq_files_chat_id_name", "files", ["chat_id", "name"])


def downgrade() -> None:
    op.drop_constraint("uq_files_chat_id_name", "files", type_="unique")
