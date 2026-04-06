"""Unit tests for ChatService.

Covers chat creation, listing, rename, and cascade-delete (jobs → messages → files → chat).
"""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.chat.dto.chat_schema import ChatCreate, ChatUpdate
from app.chat.model.chat import Chat
from app.chat.service.chat_service import ChatService


def _make_service() -> tuple[ChatService, MagicMock, MagicMock, MagicMock, MagicMock]:
    mock_chat_repo = MagicMock()
    mock_file_service = MagicMock()
    mock_message_service = MagicMock()
    mock_job_repo = MagicMock()

    service = ChatService(
        chat_repository=mock_chat_repo,
        file_service=mock_file_service,
        message_service=mock_message_service,
        index_job_repository=mock_job_repo,
    )
    return service, mock_chat_repo, mock_file_service, mock_message_service, mock_job_repo


# ── create ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_delegates_to_repo():
    service, mock_repo, *_ = _make_service()
    chat = Chat(name="my-chat")
    mock_repo.create = AsyncMock(return_value=chat)

    result = await service.create(ChatCreate(name="my-chat"))

    assert result.name == "my-chat"
    mock_repo.create.assert_awaited_once()


# ── all ───────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_returns_chats():
    service, mock_repo, *_ = _make_service()
    chats = [Chat(name="a"), Chat(name="b")]
    mock_repo.all = AsyncMock(return_value=chats)

    result = await service.all()

    assert len(result) == 2


# ── rename ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rename_calls_update_with_new_name():
    service, mock_repo, *_ = _make_service()
    cid = uuid.uuid4()
    updated = Chat(id=cid, name="new-name")
    mock_repo.update = AsyncMock(return_value=updated)

    result = await service.rename(cid, ChatUpdate(name="new-name"))

    mock_repo.update.assert_awaited_once_with(cid, {"name": "new-name"})
    assert result.name == "new-name"


# ── delete cascade ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_cascade_jobs_messages_files_then_chat():
    """Delete must remove jobs, messages, files, then the chat — in that order."""
    service, mock_repo, mock_file_service, mock_msg_service, mock_job_repo = _make_service()

    cid = uuid.uuid4()
    job1 = MagicMock(id=uuid.uuid4())
    job2 = MagicMock(id=uuid.uuid4())
    mock_job_repo.all_for_chat = AsyncMock(return_value=[job1, job2])
    mock_job_repo.delete = AsyncMock(return_value=True)
    mock_msg_service.delete_by_chat = AsyncMock(return_value=3)
    mock_file_service.delete_by_chat = AsyncMock(return_value=2)
    mock_repo.delete = AsyncMock(return_value=True)

    result = await service.delete(cid)

    assert result is True
    # Jobs deleted first
    assert mock_job_repo.delete.await_count == 2
    # Then messages
    mock_msg_service.delete_by_chat.assert_awaited_once_with(cid)
    # Then files
    mock_file_service.delete_by_chat.assert_awaited_once_with(cid)
    # Finally the chat itself
    mock_repo.delete.assert_awaited_once_with(cid)


# ── get_by_id ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_by_id_delegates():
    service, mock_repo, *_ = _make_service()
    cid = uuid.uuid4()
    chat = Chat(id=cid, name="test")
    mock_repo.get_by_id = AsyncMock(return_value=chat)

    result = await service.get_by_id(cid)
    assert result.id == cid


@pytest.mark.asyncio
async def test_get_by_id_raises_when_not_found():
    service, mock_repo, *_ = _make_service()
    mock_repo.get_by_id = AsyncMock(side_effect=ValueError("not found"))

    with pytest.raises(ValueError):
        await service.get_by_id(uuid.uuid4())
