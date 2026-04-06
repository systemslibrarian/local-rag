"""Unit tests for MessageService and MessageUI helpers.

Covers message creation, ordering, history building, answer formatting,
and the sources-stripping logic.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.dto.ai_schema import AIAnswer, Citation
from app.message.dto.message_enum import MessageType
from app.message.dto.message_schema import MessageCreate
from app.message.model.message import Message
from app.message.service.message_service import MessageService
from app.message.ui.message_ui import MessageUI


# ── Helper ────────────────────────────────────────────────────────────────────

def _msg(text: str, msg_type: MessageType, minutes: int = 0) -> MagicMock:
    m = MagicMock(spec=Message)
    m.text = text
    m.type = msg_type
    m.created_at = datetime(2024, 1, 1, 0, minutes, 0, tzinfo=timezone.utc)
    return m


# ── MessageService.create ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_message_service_create():
    mock_repo = MagicMock()
    msg = Message(text="hello", chat_id=uuid.uuid4(), type=MessageType.USER)
    mock_repo.create = AsyncMock(return_value=msg)
    service = MessageService(message_repository=mock_repo)

    result = await service.create(
        MessageCreate(text="hello", chat_id=uuid.uuid4(), type=MessageType.USER)
    )
    assert result.text == "hello"
    assert result.type == MessageType.USER


# ── MessageService.delete_by_chat ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_by_chat_deletes_all():
    mock_repo = MagicMock()
    cid = uuid.uuid4()
    msgs = [
        MagicMock(id=uuid.uuid4()),
        MagicMock(id=uuid.uuid4()),
    ]
    mock_repo.all = AsyncMock(return_value=msgs)
    mock_repo.delete = AsyncMock(return_value=True)
    service = MessageService(message_repository=mock_repo)

    count = await service.delete_by_chat(cid)
    assert count == 2
    assert mock_repo.delete.await_count == 2


# ── _strip_sources_section ────────────────────────────────────────────────────

def test_strip_sources_section_removes_sources():
    text = "Some answer text.\n\nSources:\n- doc.pdf (page 1): excerpt"
    result = MessageUI._strip_sources_section(text)
    assert result == "Some answer text."


def test_strip_sources_section_noop_when_no_sources():
    text = "Plain answer with no sources block."
    result = MessageUI._strip_sources_section(text)
    assert result == text


# ── _build_history ────────────────────────────────────────────────────────────

def test_build_history_pairs_user_and_system():
    messages = [
        _msg("q1", MessageType.USER, 1),
        _msg("a1\n\nSources:\n- doc.pdf", MessageType.SYSTEM, 2),
        _msg("q2", MessageType.USER, 3),
        _msg("a2", MessageType.SYSTEM, 4),
    ]
    pairs = MessageUI._build_history(messages)
    assert len(pairs) == 2
    assert pairs[0] == ("q1", "a1")        # sources stripped
    assert pairs[1] == ("q2", "a2")


def test_build_history_empty_list():
    assert MessageUI._build_history([]) == []


def test_build_history_single_user_message():
    messages = [_msg("q1", MessageType.USER, 1)]
    assert MessageUI._build_history(messages) == []


def test_build_history_respects_window_limit():
    """If window is smaller than available pairs, only keep the latest ones."""
    messages = []
    for i in range(20):
        messages.append(_msg(f"q{i}", MessageType.USER, i * 2))
        messages.append(_msg(f"a{i}", MessageType.SYSTEM, i * 2 + 1))

    # Patch _HISTORY_WINDOW to 3 for this test
    import app.message.ui.message_ui as module
    original = module._HISTORY_WINDOW
    try:
        module._HISTORY_WINDOW = 3
        pairs = MessageUI._build_history(messages)
        assert len(pairs) == 3
        # Should be the LAST 3 pairs
        assert pairs[-1] == ("q19", "a19")
    finally:
        module._HISTORY_WINDOW = original


def test_build_history_skips_consecutive_same_type():
    """Two consecutive USER messages — should only pair the correct ones."""
    messages = [
        _msg("q1", MessageType.USER, 1),
        _msg("q2", MessageType.USER, 2),      # unpaired user msg
        _msg("q3", MessageType.USER, 3),
        _msg("a3", MessageType.SYSTEM, 4),
    ]
    pairs = MessageUI._build_history(messages)
    assert len(pairs) == 1
    assert pairs[0] == ("q3", "a3")


# ── format_ai_answer ─────────────────────────────────────────────────────────

def test_format_ai_answer_no_citations():
    answer = AIAnswer(answer="The answer.", citations=[])
    result = MessageUI.format_ai_answer(answer)
    assert result == "The answer."
    assert "Sources:" not in result


def test_format_ai_answer_with_citations():
    answer = AIAnswer(
        answer="Capital is Paris.",
        citations=[
            Citation(file_name="geo.pdf", page=3, excerpt="Paris is the capital"),
            Citation(file_name="atlas.pdf", page=None, excerpt=None),
        ],
    )
    result = MessageUI.format_ai_answer(answer)
    assert "Sources:" in result
    assert "geo.pdf (page 3): Paris is the capital" in result
    assert "atlas.pdf" in result


def test_format_ai_answer_roundtrip_with_strip():
    """format_ai_answer → _strip_sources_section should give back just the answer."""
    answer = AIAnswer(
        answer="My answer.",
        citations=[Citation(file_name="a.pdf", page=1, excerpt="ex")],
    )
    formatted = MessageUI.format_ai_answer(answer)
    stripped = MessageUI._strip_sources_section(formatted)
    assert stripped == "My answer."


# ── MessageType enum values ──────────────────────────────────────────────────

def test_message_type_enum_names():
    """Ensure enum member names are USER and SYSTEM (used by values_callable)."""
    assert MessageType.USER.name == "USER"
    assert MessageType.SYSTEM.name == "SYSTEM"


def test_message_type_equality():
    """Enum comparison must use the enum itself, not string or int."""
    assert MessageType.USER == MessageType.USER
    assert MessageType.USER != MessageType.SYSTEM
    # Should NOT equal the string "USER" or int 2
    assert MessageType.USER != "USER"
    assert MessageType.USER != 2
