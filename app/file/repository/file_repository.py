import uuid

from sqlalchemy.future import select

from app.file.model.file import File
from internal.config.db_config import DBConfig
from internal.domain.base_repository import BaseRepository


class FileRepository(BaseRepository[File]):
    def __init__(self, db_config: DBConfig) -> None:
        super().__init__(File, db_config)

    async def find_by_chat_and_name(self, chat_id: uuid.UUID, name: str) -> File | None:
        async with self.db_config.getSession() as session:
            stmt = select(self.model_class).filter_by(chat_id=chat_id, name=name)
            result = await session.execute(stmt)
            return result.scalars().first()
