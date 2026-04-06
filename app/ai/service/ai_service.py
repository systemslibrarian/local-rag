import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.vectorstores import VectorStore

from app.ai.dto.ai_schema import AIAnswer, Citation
from app.file.service.file_service import FileService
from internal.config.logging_config import StructuredLogger, timed
from internal.config.setting import setting

_log = StructuredLogger(__name__)

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Generate five different versions of the
given user question to retrieve relevant documents from a vector database.
Provide the alternative questions separated by newlines.
Original question: {question}""",
)

# Grounding prompt — includes chat history slot for conversation memory
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are answering questions about uploaded documents.

Rules:
- Use ONLY the provided context.
- If the context is insufficient, say you could not find the answer in the uploaded documents.
- Keep the answer concise and factual.
- Do not invent citations or details not present in the context.
- You may refer to earlier messages in the conversation to understand follow-up questions.

Context:
{context}""",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


@dataclass
class AIService:
    def __init__(self, llm: BaseChatModel, vector_store: VectorStore, file_service: FileService):
        self.llm = llm
        self.db = vector_store
        self.file_service = file_service

    @staticmethod
    def _citation_from_document(document: Document) -> Citation:
        file_name = document.metadata.get("file_name") or Path(str(document.metadata.get("source", "Document"))).name
        raw_page = document.metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else None
        excerpt = " ".join(document.page_content.split())
        if len(excerpt) > 180:
            excerpt = f"{excerpt[:177]}..."
        return Citation(file_name=str(file_name), page=page, excerpt=excerpt or None)

    def _build_citations(self, documents: list[Document], citation_limit: int) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None]] = set()
        for document in documents:
            citation = self._citation_from_document(document)
            key = (citation.file_name, citation.page)
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
            if len(citations) == citation_limit:
                break
        return citations

    @staticmethod
    def _format_context(documents: list[Document], top_k: int) -> str:
        sections: list[str] = []
        for index, document in enumerate(documents[:top_k], start=1):
            file_name = document.metadata.get("file_name") or Path(str(document.metadata.get("source", "Document"))).name
            raw_page = document.metadata.get("page")
            page_label = f", page {raw_page + 1}" if isinstance(raw_page, int) else ""
            content = " ".join(document.page_content.split())
            sections.append(f"[Source {index}: {file_name}{page_label}]\n{content}")
        return "\n\n".join(sections)

    async def _score_filter(
        self,
        query: str,
        file_ids: list[str],
        top_k: int,
    ) -> list[Document]:
        """Return docs that pass the similarity threshold, ordered best-first."""
        threshold = setting.similarity_threshold
        scored = await self.db.asimilarity_search_with_relevance_scores(
            query,
            k=top_k,
            filter={"file_id": {"$in": file_ids}},
        )
        passed = [(doc, score) for doc, score in scored if score >= threshold]
        _log.debug(
            "similarity_filter",
            total=len(scored),
            passed=len(passed),
            threshold=threshold,
            top_score=round(scored[0][1], 4) if scored else None,
        )
        return [doc for doc, _ in passed]

    async def stream_query(
        self,
        query: str,
        chat_id: uuid.UUID,
        history: list[tuple[str, str]],
        top_k: int = 10,
        citation_limit: int = 4,
        use_multi_query: bool = True,
    ) -> AsyncGenerator[str | list[Citation], None]:
        """
        Async generator that yields str chunks of the answer, then yields
        list[Citation] as the final item once generation is complete.
        Yields AIAnswer.answer as a single string on error paths.
        """
        file_ids = await self.file_service.find_files_ids(chat_id=chat_id)
        if not file_ids:
            yield "No documents uploaded yet. Please upload files first."
            yield []
            return

        # --- Relevance gate ---
        try:
            with timed(_log, "retrieval", chat_id=str(chat_id), top_k=top_k, multi_query=use_multi_query):
                relevant_docs = await self._score_filter(query, file_ids, top_k)
                if not relevant_docs:
                    _log.info("no_evidence", chat_id=str(chat_id))
                    yield "I could not find relevant information in the uploaded documents for your question."
                    yield []
                    return

                if use_multi_query:
                    base_retriever = self.db.as_retriever(
                        search_kwargs={
                            "filter": {"file_id": {"$in": file_ids}},
                            "k": top_k,
                        }
                    )
                    retriever = MultiQueryRetriever.from_llm(
                        base_retriever, self.llm, prompt=query_prompt,
                    )
                    mq_docs = await retriever.ainvoke(query)
                    seen_ids = {id(d) for d in relevant_docs}
                    extra = [d for d in mq_docs if id(d) not in seen_ids]
                    documents = (relevant_docs + extra)[:top_k]
                else:
                    documents = relevant_docs
        except Exception as exc:
            _log.error("retrieval_error", error=str(exc), chat_id=str(chat_id))
            yield "I couldn't complete the retrieval step. Please check Ollama and try again."
            yield []
            return

        # --- Build LangChain history messages ---
        history_messages = []
        for human_text, ai_text in history:
            history_messages.append(HumanMessage(content=human_text))
            history_messages.append(AIMessage(content=ai_text))

        try:
            context = self._format_context(documents, top_k)
            chain = _PROMPT | self.llm | StrOutputParser()
            full_text = ""
            with timed(_log, "generation", chat_id=str(chat_id), num_docs=len(documents)):
                async for chunk in chain.astream({
                    "context": context,
                    "question": query,
                    "history": history_messages,
                }):
                    full_text += chunk
                    yield chunk

            citations = self._build_citations(documents, citation_limit)
            _log.info("query_complete", chat_id=str(chat_id), num_citations=len(citations))
            yield citations
        except Exception as exc:
            _log.error("generation_error", error=str(exc), chat_id=str(chat_id))
            yield "I couldn't generate an answer. Please check Ollama and try again."
            yield []

    async def query(
        self,
        query: str,
        chat_id: uuid.UUID,
        history: list[tuple[str, str]] | None = None,
        top_k: int = 10,
        citation_limit: int = 4,
        use_multi_query: bool = True,
    ) -> AIAnswer:
        """Non-streaming wrapper around stream_query for callers that need a complete AIAnswer."""
        full_text = ""
        citations: list[Citation] = []
        async for item in self.stream_query(
            query=query,
            chat_id=chat_id,
            history=history or [],
            top_k=top_k,
            citation_limit=citation_limit,
            use_multi_query=use_multi_query,
        ):
            if isinstance(item, list):
                citations = item
            else:
                full_text += item
        return AIAnswer(answer=full_text, citations=citations)


