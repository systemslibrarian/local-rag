import uuid
from datetime import datetime

from sqlalchemy import UUID, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from internal.domain.entity import Entity


class IndexJob(Entity):
    __tablename__ = "index_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    file_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    pages: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    chunks: Mapped[int] = mapped_column(Integer(), nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<IndexJob(id={self.id}, chat_id='{self.chat_id}', file_name='{self.file_name}', "
            f"status='{self.status}', created_at='{self.created_at}')>"
        )