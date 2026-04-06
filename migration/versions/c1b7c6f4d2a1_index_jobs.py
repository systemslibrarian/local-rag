"""index jobs

Revision ID: c1b7c6f4d2a1
Revises: 4558aa425581
Create Date: 2026-04-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1b7c6f4d2a1'
down_revision: Union[str, None] = '4558aa425581'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'index_jobs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('chat_id', sa.UUID(), nullable=False),
        sa.Column('file_id', sa.UUID(), nullable=True),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('pages', sa.Integer(), nullable=False),
        sa.Column('chunks', sa.Integer(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_index_jobs_chat_id'), 'index_jobs', ['chat_id'], unique=False)
    op.create_index(op.f('ix_index_jobs_file_id'), 'index_jobs', ['file_id'], unique=False)
    op.create_index(op.f('ix_index_jobs_file_name'), 'index_jobs', ['file_name'], unique=False)
    op.create_index(op.f('ix_index_jobs_status'), 'index_jobs', ['status'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_index_jobs_status'), table_name='index_jobs')
    op.drop_index(op.f('ix_index_jobs_file_name'), table_name='index_jobs')
    op.drop_index(op.f('ix_index_jobs_file_id'), table_name='index_jobs')
    op.drop_index(op.f('ix_index_jobs_chat_id'), table_name='index_jobs')
    op.drop_table('index_jobs')