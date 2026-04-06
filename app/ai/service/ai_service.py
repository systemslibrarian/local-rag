import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import VectorStore

from app.ai.dto.ai_schema import AIAnswer, Citation, LLMResponse
from app.file.service.file_service import FileService
from internal.config.logging_config import StructuredLogger, timed

_log = StructuredLogger(__name__)

# Minimum cosine-similarity score for a chunk to be considered relevant.
# Chunks below this threshold are discarded before context assembly.
SIMILARITY_THRESHOLD = 0.30

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("system", "{data}"),
        ("human", "{text}"),
    ]
)

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant and an expert extraction algorithm. Your task is to generate five
        different versions of the given user question to retrieve and extract relevant documents and information from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
)

template = """You are answering questions about uploaded documents.

Rules:
- Use ONLY the provided context.
- If the context is insufficient, say that you could not find the answer in the uploaded documents.
- Keep the answer concise and factual.
- Do not invent citations or details that are not present in the context.

Context:
{context}

Question: {question}
"""

@dataclass
class AIService:
    def __init__(self, llm: BaseChatModel, vector_store: VectorStore, file_service: FileService):
        self.llm_original = llm
        self.llm = llm.with_structured_output(schema=LLMResponse)
        self.db = vector_store
        self.file_service = file_service

    async def query_alternative(self, query: str, chat_id: uuid.UUID) -> str | None:
        documents = await self.file_service.search_documents(query=query, chat_id=chat_id)
        data = "\n\n".join(doc.page_content for doc in documents)
        prompt = prompt_template.invoke({"text": query, "data": data})
        llm_result = await self.llm.ainvoke(prompt)

        return LLMResponse.model_validate(llm_result).answer if llm_result else None

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
        scored = await self.db.asimilarity_search_with_relevance_scores(
            query,
            k=top_k,
            filter={"file_id": {"$in": file_ids}},
        )
        passed = [(doc, score) for doc, score in scored if score >= SIMILARITY_THRESHOLD]
        _log.debug(
            "similarity_filter",
            total=len(scored),
            passed=len(passed),
            threshold=SIMILARITY_THRESHOLD,
            top_score=round(scored[0][1], 4) if scored else None,
        )
        return [doc for doc, _ in passed]

    async def query(
        self,
        query: str,
        chat_id: uuid.UUID,
        top_k: int = 10,
        citation_limit: int = 4,
        use_multi_query: bool = True,
    ) -> AIAnswer:
        prompt = ChatPromptTemplate.from_template(template)
        file_ids = await self.file_service.find_files_ids(chat_id=chat_id)
        if not file_ids:
            return AIAnswer(answer="No documents uploaded yet. Please upload files first.")

        # --- Relevance gate: score-filter before invoking the LLM at all ---
        try:
            with timed(_log, "retrieval", chat_id=str(chat_id), top_k=top_k, multi_query=use_multi_query):
                relevant_docs = await self._score_filter(query, file_ids, top_k)
                if not relevant_docs:
                    _log.info("no_evidence", chat_id=str(chat_id))
                    return AIAnswer(
                        answer="I could not find relevant information in the uploaded documents for your question."
                    )

                if use_multi_query:
                    base_retriever = self.db.as_retriever(
                        search_kwargs={
                            "filter": {"file_id": {"$in": file_ids}},
                            "k": top_k,
                        }
                    )
                    retriever = MultiQueryRetriever.from_llm(
                        base_retriever,
                        self.llm_original,
                        prompt=query_prompt,
                    )
                    # Multi-query may surface additional docs; merge with scored set
                    mq_docs = await retriever.ainvoke(query)
                    # Keep the relevance-scored doc first so context ordering is deterministic,
                    # then append any extra docs multi-query found (up to top_k total).
                    seen_ids = {id(d) for d in relevant_docs}
                    extra = [d for d in mq_docs if id(d) not in seen_ids]
                    documents = (relevant_docs + extra)[:top_k]
                else:
                    documents = relevant_docs
        except Exception as exc:
            _log.error("retrieval_error", error=str(exc), chat_id=str(chat_id))
            return AIAnswer(answer="I couldn't complete the retrieval step. Please check Ollama and try again.")

        try:
            context = self._format_context(documents, top_k)
            chain = prompt | self.llm_original | StrOutputParser()
            with timed(_log, "generation", chat_id=str(chat_id), num_docs=len(documents)):
                response = await chain.ainvoke({"context": context, "question": query})
            answer = response.strip() if response else "I don't know"
            result = AIAnswer(answer=answer, citations=self._build_citations(documents, citation_limit))
            _log.info(
                "query_complete",
                chat_id=str(chat_id),
                num_citations=len(result.citations),
            )
            return result
        except Exception as exc:
            _log.error("generation_error", error=str(exc), chat_id=str(chat_id))
            return AIAnswer(answer="I couldn't generate an answer. Please check Ollama and try again.")

