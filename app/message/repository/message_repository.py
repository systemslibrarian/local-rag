

from typing import Sequence

from sqlalchemy.future import select

from app.message.model.message import Message
from internal.config.db_config import DBConfig
from internal.domain.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    def __init__(self, db_config: DBConfig) -> None:
        super().__init__(Message, db_config)

    async def all(self, conditions: dict | None = None) -> Sequence[Message]:
        if conditions is None:
            conditions = {}
        async with self.db_config.getSession() as session:
            stmt = (
                select(self.model_class)
                .filter_by(**conditions)
                .order_by(self.model_class.created_at)
            )
            result = await session.execute(stmt)
            return result.scalars().all()
