import uuid

from sqlalchemy import desc
from sqlalchemy.future import select

from app.file.model.index_job import IndexJob
from internal.config.db_config import DBConfig
from internal.domain.base_repository import BaseRepository


class IndexJobRepository(BaseRepository[IndexJob]):
    def __init__(self, db_config: DBConfig) -> None:
        super().__init__(IndexJob, db_config)

    async def all_for_chat(self, chat_id: uuid.UUID) -> list[IndexJob]:
        async with self.db_config.getSession() as session:
            stmt = select(self.model_class).filter_by(chat_id=chat_id).order_by(desc(self.model_class.created_at))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def find_active_by_chat_and_name(self, chat_id: uuid.UUID, file_name: str) -> IndexJob | None:
        async with self.db_config.getSession() as session:
            stmt = select(self.model_class).filter(
                self.model_class.chat_id == chat_id,
                self.model_class.file_name == file_name,
                self.model_class.status.in_(["queued", "running"]),
            )
            result = await session.execute(stmt)
            return result.scalars().first()