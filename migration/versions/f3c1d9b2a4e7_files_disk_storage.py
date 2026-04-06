"""files: replace content blob with storage_path

Revision ID: f3c1d9b2a4e7
Revises: c1b7c6f4d2a1
Create Date: 2026-04-06 00:00:00.000000

"""
import os
from pathlib import Path
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "f3c1d9b2a4e7"
down_revision: Union[str, None] = "c1b7c6f4d2a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add nullable column first
    op.add_column("files", sa.Column("storage_path", sa.String(length=512), nullable=True))

    # 2. Data migration: write existing PDF bytes to disk, record path
    storage_folder = os.environ.get("FILE_STORAGE_FOLDER", "./data/files")
    Path(storage_folder).mkdir(parents=True, exist_ok=True)

    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, content FROM files")).fetchall()
    for row in rows:
        file_path = Path(storage_folder) / f"{row.id}.pdf"
        with open(file_path, "wb") as fh:
            fh.write(row.content)
        conn.execute(
            sa.text("UPDATE files SET storage_path = :path WHERE id = :id"),
            {"path": str(file_path), "id": str(row.id)},
        )

    # 3. Make NOT NULL once populated
    op.alter_column("files", "storage_path", nullable=False)

    # 4. Drop old blob column
    op.drop_column("files", "content")


def downgrade() -> None:
    op.add_column("files", sa.Column("content", sa.LargeBinary(), nullable=True))

    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, storage_path FROM files")).fetchall()
    for row in rows:
        try:
            with open(row.storage_path, "rb") as fh:
                content = fh.read()
            conn.execute(
                sa.text("UPDATE files SET content = :content WHERE id = :id"),
                {"content": content, "id": str(row.id)},
            )
        except Exception:
            conn.execute(
                sa.text("UPDATE files SET content = :content WHERE id = :id"),
                {"content": b"", "id": str(row.id)},
            )

    op.alter_column("files", "content", nullable=False)
    op.drop_column("files", "storage_path")
