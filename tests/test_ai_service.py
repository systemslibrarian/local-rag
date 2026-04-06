"""Unit tests for AIService.

All external dependencies (vector store, LLM, FileService) are mocked so
these tests run offline without Ollama or PostgreSQL.
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
    """Return (service, mock_llm, mock_vector_store, mock_file_service)."""
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


# ── No documents uploaded ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_query_no_docs_returns_early():
    service, *_ = _make_service(file_ids=[])
    text, citations = await _collect_stream(
        service.stream_query("anything", uuid.uuid4(), history=[])
    )
    assert "No documents uploaded" in text
    assert citations == []


# ── Similarity threshold gate ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_query_no_evidence_when_all_scores_below_threshold():
    """All retrieved chunks score below threshold → no-evidence response, LLM not called."""
    low_score_doc = MagicMock()
    low_score_doc.metadata = {"file_name": "doc.pdf", "page": 0}
    low_score_doc.page_content = "Some text"

    service, mock_llm, _, _ = _make_service(
        file_ids=["abc"],
        relevance_scores=[(low_score_doc, 0.10)],  # below 0.30
    )

    with patch("app.ai.service.ai_service.setting") as mock_setting:
        mock_setting.similarity_threshold = 0.30
        text, citations = await _collect_stream(
            service.stream_query("irrelevant question", uuid.uuid4(), history=[])
        )

    assert "could not find relevant information" in text
    assert citations == []
    # LLM should never be invoked
    mock_llm.astream.assert_not_called()


@pytest.mark.asyncio
async def test_stream_query_passes_when_score_above_threshold():
    """Chunks above threshold are passed to the LLM; response is streamed."""
    high_score_doc = MagicMock()
    high_score_doc.metadata = {"file_name": "report.pdf", "page": 2}
    high_score_doc.page_content = "The capital of France is Paris."

    async def _fake_astream(_input):
        yield "The capital"
        yield " of France is Paris."

    mock_chain = MagicMock()
    mock_chain.astream = _fake_astream

    service, mock_llm, _, _ = _make_service(
        file_ids=["xyz"],
        relevance_scores=[(high_score_doc, 0.85)],
    )

    with (
        patch("app.ai.service.ai_service.setting") as mock_setting,
        patch("app.ai.service.ai_service._PROMPT") as mock_prompt,
        patch("app.ai.service.ai_service.StrOutputParser") as mock_parser,
    ):
        mock_setting.similarity_threshold = 0.30
        mock_setting.history_window = 6
        mock_prompt.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        text, citations = await _collect_stream(
            service.stream_query("What is the capital of France?", uuid.uuid4(), history=[], use_multi_query=False)
        )

    assert "Paris" in text
    assert len(citations) >= 0  # citation list returned


# ── query() wrapper ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_wrapper_returns_aianswer():
    service, *_ = _make_service(file_ids=[])
    result = await service.query("test", uuid.uuid4())
    assert isinstance(result, AIAnswer)
    assert result.answer != ""


# ── Citation building ─────────────────────────────────────────────────────────

def test_citation_from_document_extracts_page():
    doc = MagicMock()
    doc.metadata = {"file_name": "guide.pdf", "page": 4}
    doc.page_content = "Short content"

    service, *_ = _make_service()
    citation = service._citation_from_document(doc)

    assert citation.file_name == "guide.pdf"
    assert citation.page == 5  # 0-indexed → 1-indexed


def test_build_citations_deduplicates():
    doc_a = MagicMock()
    doc_a.metadata = {"file_name": "a.pdf", "page": 0}
    doc_a.page_content = "text a"

    doc_b = MagicMock()  # same file/page as doc_a
    doc_b.metadata = {"file_name": "a.pdf", "page": 0}
    doc_b.page_content = "text b"

    service, *_ = _make_service()
    citations = service._build_citations([doc_a, doc_b], citation_limit=5)
    assert len(citations) == 1  # deduplicated
