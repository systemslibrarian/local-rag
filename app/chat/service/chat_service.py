import uuid
from dataclasses import dataclass
from typing import Sequence

from app.chat.dto.chat_schema import ChatCreate, ChatUpdate
from app.chat.model.chat import Chat
from app.chat.repository.chat_repository import ChatRepository
from app.file.repository.index_job_repository import IndexJobRepository
from app.file.service.file_service import FileService
from app.message.service.message_service import MessageService


@dataclass
class ChatService:
    chat_repository: ChatRepository
    file_service: FileService
    message_service: MessageService
    index_job_repository: IndexJobRepository

    async def create(self, chat_create: ChatCreate) -> Chat:
        chat = Chat(name=chat_create.name)
        return await self.chat_repository.create(chat)

    async def all(self) -> Sequence[Chat]:
        return await self.chat_repository.all()

    async def find_by_name(self, name: str) -> Chat | None:
        return await self.chat_repository.find_by_name(name)

    async def get_by_id(self, chat_id: uuid.UUID) -> Chat:
        return await self.chat_repository.get_by_id(chat_id)

    async def rename(self, chat_id: uuid.UUID, chat_update: ChatUpdate) -> Chat | None:
        return await self.chat_repository.update(chat_id, {"name": chat_update.name})

    async def delete(self, chat_id: uuid.UUID) -> bool:
        jobs = await self.index_job_repository.all_for_chat(chat_id)
        for job in jobs:
            await self.index_job_repository.delete(job.id)
        await self.message_service.delete_by_chat(chat_id)
        await self.file_service.delete_by_chat(chat_id)
        return await self.chat_repository.delete(chat_id)
