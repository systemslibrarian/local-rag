"""Extended tests for AIService — covers multi-query retrieval, context formatting,
score filtering, streaming error paths, and history message construction.
"""
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ai.dto.ai_schema import AIAnswer, Citation
from app.ai.service.ai_service import AIService


def _make_service(
    file_ids: list[str] | None = None,
    relevance_scores: list[tuple] | None = None,
) -> tuple[AIService, MagicMock, MagicMock, MagicMock]:
    mock_llm = MagicMock()
    mock_vs = MagicMock()
    mock_fs = MagicMock()
    mock_fs.find_files_ids = AsyncMock(return_value=file_ids or [])
    mock_vs.asimilarity_search_with_relevance_scores = AsyncMock(
        return_value=relevance_scores or []
    )
    service = AIService(llm=mock_llm, vector_store=mock_vs, file_service=mock_fs)
    return service, mock_llm, mock_vs, mock_fs


async def _collect_stream(gen: AsyncGenerator) -> tuple[str, list[Citation]]:
    text = ""
    citations: list[Citation] = []
    async for item in gen:
        if isinstance(item, list):
            citations = item
        else:
            text += item
    return text, citations


def _make_doc(file_name: str, page: int, content: str, file_id: str = "abc", chunk_index: int = 0) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {
        "file_name": file_name,
        "page": page,
        "file_id": file_id,
        "chunk_index": chunk_index,
    }
    doc.page_content = content
    return doc


# ── _format_context ──────────────────────────────────────────────────────────

def test_format_context_includes_source_labels():
    doc = _make_doc("guide.pdf", 3, "Important content here")
    result = AIService._format_context([doc], top_k=5)
    assert "[Source 1: guide.pdf, page 4]" in result  # page is 0-indexed +1
    assert "Important content here" in result


def test_format_context_limits_to_top_k():
    docs = [_make_doc(f"d{i}.pdf", 0, f"text{i}") for i in range(10)]
    result = AIService._format_context(docs, top_k=3)
    assert "[Source 3:" in result
    assert "[Source 4:" not in result


def test_format_context_no_page_label_when_not_int():
    doc = MagicMock()
    doc.metadata = {"file_name": "x.pdf", "source": "x.pdf"}
    doc.page_content = "text"
    result = AIService._format_context([doc], top_k=1)
    assert "page" not in result


# ── _score_filter ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_score_filter_passes_above_threshold():
    doc_good = _make_doc("a.pdf", 0, "good")
    doc_bad = _make_doc("b.pdf", 0, "bad")

    service, _, mock_vs, _ = _make_service(file_ids=["f1"])
    mock_vs.asimilarity_search_with_relevance_scores = AsyncMock(
        return_value=[(doc_good, 0.85), (doc_bad, 0.10)]
    )

    with patch("app.ai.service.ai_service.setting") as ms:
        ms.similarity_threshold = 0.30
        result = await service._score_filter("query", ["f1"], top_k=5)

    assert len(result) == 1
    assert result[0].metadata["file_name"] == "a.pdf"


@pytest.mark.asyncio
async def test_score_filter_uses_k_at_least_20():
    """k should be max(top_k*4, 20) — even for small top_k."""
    service, _, mock_vs, _ = _make_service(file_ids=["f1"])
    mock_vs.asimilarity_search_with_relevance_scores = AsyncMock(return_value=[])

    with patch("app.ai.service.ai_service.setting") as ms:
        ms.similarity_threshold = 0.30
        await service._score_filter("query", ["f1"], top_k=2)

    call_kwargs = mock_vs.asimilarity_search_with_relevance_scores.call_args
    assert call_kwargs.kwargs["k"] == 20  # max(2*4, 20)


@pytest.mark.asyncio
async def test_score_filter_scales_k_with_top_k():
    service, _, mock_vs, _ = _make_service(file_ids=["f1"])
    mock_vs.asimilarity_search_with_relevance_scores = AsyncMock(return_value=[])

    with patch("app.ai.service.ai_service.setting") as ms:
        ms.similarity_threshold = 0.30
        await service._score_filter("query", ["f1"], top_k=10)

    call_kwargs = mock_vs.asimilarity_search_with_relevance_scores.call_args
    assert call_kwargs.kwargs["k"] == 40  # max(10*4, 20)


# ── citation edge cases ─────────────────────────────────────────────────────

def test_citation_truncates_long_excerpt():
    doc = MagicMock()
    doc.metadata = {"file_name": "long.pdf", "page": 0}
    doc.page_content = "x " * 200  # > 180 chars

    service, *_ = _make_service()
    citation = service._citation_from_document(doc)

    assert len(citation.excerpt) <= 180


def test_citation_none_page_when_not_int():
    doc = MagicMock()
    doc.metadata = {"file_name": "no_page.pdf"}
    doc.page_content = "text"

    service, *_ = _make_service()
    citation = service._citation_from_document(doc)

    assert citation.page is None


def test_citation_falls_back_to_source_path():
    doc = MagicMock()
    doc.metadata = {"source": "/path/to/report.pdf"}
    doc.page_content = "text"

    service, *_ = _make_service()
    citation = service._citation_from_document(doc)

    assert citation.file_name == "report.pdf"


# ── stream_query: retrieval error ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_query_handles_retrieval_error():
    service, _, mock_vs, _ = _make_service(file_ids=["abc"])
    mock_vs.asimilarity_search_with_relevance_scores = AsyncMock(
        side_effect=RuntimeError("connection lost")
    )

    with patch("app.ai.service.ai_service.setting") as ms:
        ms.similarity_threshold = 0.30
        text, citations = await _collect_stream(
            service.stream_query("q", uuid.uuid4(), history=[])
        )

    assert "retrieval" in text.lower() or "couldn't" in text.lower()
    assert citations == []


# ── stream_query: generation error ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_query_handles_generation_error():
    good_doc = _make_doc("a.pdf", 0, "content")

    service, mock_llm, mock_vs, _ = _make_service(
        file_ids=["f1"],
        relevance_scores=[(good_doc, 0.90)],
    )

    mock_chain = MagicMock()

    async def _explode(_input):
        raise RuntimeError("LLM down")
        yield  # make it a generator

    mock_chain.astream = _explode

    with (
        patch("app.ai.service.ai_service.setting") as ms,
        patch("app.ai.service.ai_service._PROMPT") as mock_prompt,
        patch("app.ai.service.ai_service.StrOutputParser") as mock_parser,
    ):
        ms.similarity_threshold = 0.30
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        text, citations = await _collect_stream(
            service.stream_query("q", uuid.uuid4(), history=[], use_multi_query=False)
        )

    assert "couldn't generate" in text.lower()
    assert citations == []


# ── stream_query: history messages are built correctly ───────────────────────

@pytest.mark.asyncio
async def test_stream_query_converts_history_to_messages():
    """History pairs should become alternating HumanMessage/AIMessage."""
    good_doc = _make_doc("a.pdf", 0, "content")

    service, mock_llm, mock_vs, _ = _make_service(
        file_ids=["f1"],
        relevance_scores=[(good_doc, 0.90)],
    )

    captured_inputs: list[dict] = []

    async def _capture_astream(inputs):
        captured_inputs.append(inputs)
        yield "answer"

    mock_chain = MagicMock()
    mock_chain.astream = _capture_astream

    with (
        patch("app.ai.service.ai_service.setting") as ms,
        patch("app.ai.service.ai_service._PROMPT") as mock_prompt,
        patch("app.ai.service.ai_service.StrOutputParser") as mock_parser,
    ):
        ms.similarity_threshold = 0.30
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        history = [("q1", "a1"), ("q2", "a2")]
        text, _ = await _collect_stream(
            service.stream_query("q3", uuid.uuid4(), history=history, use_multi_query=False)
        )

    assert len(captured_inputs) == 1
    chain_input = captured_inputs[0]
    assert chain_input["question"] == "q3"
    assert len(chain_input["history"]) == 4  # 2 pairs × 2 messages each


# ── query() non-streaming wrapper ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_wrapper_accumulates_text():
    """Non-streaming query() should concatenate all chunks into answer."""
    good_doc = _make_doc("a.pdf", 0, "content")

    service, mock_llm, mock_vs, _ = _make_service(
        file_ids=["f1"],
        relevance_scores=[(good_doc, 0.90)],
    )

    async def _fake_stream(inputs):
        yield "chunk1 "
        yield "chunk2"

    mock_chain = MagicMock()
    mock_chain.astream = _fake_stream

    with (
        patch("app.ai.service.ai_service.setting") as ms,
        patch("app.ai.service.ai_service._PROMPT") as mock_prompt,
        patch("app.ai.service.ai_service.StrOutputParser") as mock_parser,
    ):
        ms.similarity_threshold = 0.30
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        result = await service.query("q", uuid.uuid4(), use_multi_query=False)

    assert isinstance(result, AIAnswer)
    assert "chunk1" in result.answer
    assert "chunk2" in result.answer
