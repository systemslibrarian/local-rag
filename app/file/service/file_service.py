import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from pdf2image import convert_from_bytes
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.chat.repository.chat_repository import ChatRepository
from app.file.dto.file_schema import FileCreate
from app.file.model.file import File
from app.file.model.index_job import IndexJob
from app.file.repository.file_repository import FileRepository
from app.file.repository.index_job_repository import IndexJobRepository
from internal.config.logging_config import StructuredLogger, timed

_log = StructuredLogger(__name__)


@dataclass
class UploadResult:
    file: File
    created: bool
    message: str
    pages: int
    chunks: int

@dataclass
class FileService:
    chat_repository: ChatRepository
    file_repository: FileRepository
    index_job_repository: IndexJobRepository
    text_specifier: TextSplitter
    vector_store: VectorStore

    @staticmethod
    def build_chunk_id(file_id: uuid.UUID, chunk_index: int) -> str:
        return f"{file_id}:{chunk_index}"

    async def _load_documents_from_bytes(self, pdf_bytes: bytes) -> list[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            tmp_file_path = tmp_file.name
        try:
            loader = PyPDFLoader(file_path=tmp_file_path)
            return loader.load()
        finally:
            os.unlink(tmp_file_path)

    async def _split_file_content(self, file: File) -> tuple[list[Document], list[Document]]:
        data = await self._load_documents_from_bytes(file.content)
        if not data:
            raise ValueError("No readable text found in the uploaded PDF.")
        chunks = self.text_specifier.split_documents(data)
        if not chunks:
            raise ValueError("No chunks were produced from the uploaded PDF.")
        return data, chunks

    async def _index_file(self, file: File) -> tuple[int, int]:
        data, chunks = await self._split_file_content(file)
        chunk_ids: list[str] = []
        for chunk_index, document in enumerate(chunks):
            document.metadata["file_name"] = file.name
            document.metadata["file_id"] = str(file.id)
            document.metadata["chat_id"] = str(file.chat_id)
            document.metadata["chunk_index"] = chunk_index
            chunk_ids.append(self.build_chunk_id(file.id, chunk_index))
        _log.info("indexing_start", file_name=file.name, pages=len(data), chunks=len(chunks))
        with timed(_log, "embedding", file_name=file.name, chunks=len(chunks)):
            await self.vector_store.aadd_documents(chunks, ids=chunk_ids)
        _log.info("indexing_done", file_name=file.name, pages=len(data), chunks=len(chunks))
        return len(data), len(chunks)

    async def _delete_vectors_for_file(self, file: File) -> None:
        try:
            _, chunks = await self._split_file_content(file)
        except Exception:
            return
        chunk_ids = [self.build_chunk_id(file.id, chunk_index) for chunk_index in range(len(chunks))]
        if chunk_ids:
            await self.vector_store.adelete(ids=chunk_ids, collection_only=True)

    @staticmethod
    def pdf_to_image(pdf_bytes: bytes, only_first_page=False) -> list[Image.Image]:
        if only_first_page:
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        else:
            images = convert_from_bytes(pdf_bytes)
        return images

    async def create(self, file_create: FileCreate) -> File:
        f = File(name=file_create.name, chat_id=file_create.chat_id, content=file_create.content)
        return await self.file_repository.create(f)

    async def all(self, conditions: dict | None = None) -> Sequence[File]:
        if conditions is None:
            conditions = {}
        return await self.file_repository.all(conditions=conditions)

    async def get_by_id(self, file_id: uuid.UUID) -> File:
        return await self.file_repository.get_by_id(file_id)

    async def delete(self, file_id: uuid.UUID) -> bool:
        file = await self.get_by_id(file_id)
        await self._delete_vectors_for_file(file)
        return await self.file_repository.delete(file_id)

    async def delete_by_chat(self, chat_id: uuid.UUID) -> int:
        files = await self.all(conditions={"chat_id": chat_id})
        deleted = 0
        for file in files:
            await self.delete(file.id)
            deleted += 1
        return deleted

    async def find_by_chat_and_name(self, chat_id: uuid.UUID, name: str) -> File | None:
        return await self.file_repository.find_by_chat_and_name(chat_id=chat_id, name=name)

    async def has_files(self, chat_id: uuid.UUID) -> bool:
        files = await self.all(conditions={"chat_id": chat_id})
        return bool(files)

    async def list_jobs(self, chat_id: uuid.UUID) -> list[IndexJob]:
        return await self.index_job_repository.all_for_chat(chat_id)

    async def has_active_jobs(self, chat_id: uuid.UUID) -> bool:
        jobs = await self.list_jobs(chat_id)
        return any(job.status in {"queued", "running"} for job in jobs)

    async def clear_finished_jobs(self, chat_id: uuid.UUID) -> int:
        jobs = await self.list_jobs(chat_id)
        deleted = 0
        for job in jobs:
            if job.status in {"completed", "failed"}:
                await self.index_job_repository.delete(job.id)
                deleted += 1
        return deleted

    async def _update_job(self, job_id: uuid.UUID, **data: object) -> IndexJob | None:
        return await self.index_job_repository.update(job_id, data)

    async def _run_upload_job(self, job_id: uuid.UUID, pdf_bytes: bytes, file_name: str, chat_id: uuid.UUID) -> None:
        await self._update_job(job_id, status="running", message="Reading PDF and generating chunks...")
        _log.info("job_start", job_id=str(job_id), file_name=file_name, chat_id=str(chat_id))

        chat = await self.chat_repository.find_by_id(chat_id)
        if chat is None:
            await self._update_job(job_id, status="failed", message="Chat no longer exists.")
            _log.warning("job_aborted", job_id=str(job_id), reason="chat_not_found")
            return

        existing_file = await self.find_by_chat_and_name(chat_id=chat_id, name=file_name)
        if existing_file is not None:
            await self._update_job(
                job_id,
                status="completed",
                file_id=existing_file.id,
                message=f"File '{file_name}' is already indexed for this chat.",
            )
            return

        file = await self.create(FileCreate(name=file_name, chat_id=chat_id, content=pdf_bytes))
        try:
            with timed(_log, "job_index", job_id=str(job_id), file_name=file_name):
                pages, chunks = await self._index_file(file)
            await self._update_job(
                job_id,
                status="completed",
                file_id=file.id,
                pages=pages,
                chunks=chunks,
                message=f"Indexed {file_name} successfully.",
            )
            _log.info("job_done", job_id=str(job_id), file_name=file_name, pages=pages, chunks=chunks)
        except Exception as exc:
            await self.file_repository.delete(file.id)
            await self._update_job(job_id, status="failed", message=f"Failed to index {file_name}: {exc}")
            _log.error("job_failed", job_id=str(job_id), file_name=file_name, error=str(exc))

    async def submit_upload_job(self, file_name: str, pdf_bytes: bytes, chat_id: uuid.UUID) -> IndexJob:
        existing_file = await self.find_by_chat_and_name(chat_id=chat_id, name=file_name)
        if existing_file is not None:
            job = IndexJob(
                chat_id=chat_id,
                file_name=file_name,
                status="completed",
                message=f"File '{file_name}' is already indexed for this chat.",
                file_id=existing_file.id,
                pages=0,
                chunks=0,
            )
            return job

        active_job = await self.index_job_repository.find_active_by_chat_and_name(chat_id=chat_id, file_name=file_name)
        if active_job is not None:
            return active_job

        job = await self.index_job_repository.create(IndexJob(
            chat_id=chat_id,
            file_name=file_name,
            status="queued",
            message=f"Queued {file_name} for indexing.",
            pages=0,
            chunks=0,
        ))
        asyncio.create_task(self._run_upload_job(job.id, pdf_bytes, file_name, chat_id))
        return job

    async def upload_file(self, u_file: UploadedFile, chat_id: uuid.UUID) -> UploadResult:
        existing_file = await self.find_by_chat_and_name(chat_id=chat_id, name=u_file.name)
        if existing_file is not None:
            return UploadResult(
                file=existing_file,
                created=False,
                message=f"File '{u_file.name}' is already indexed for this chat.",
                pages=0,
                chunks=0,
            )

        file = await self.create(FileCreate(name=u_file.name, chat_id=chat_id, content=u_file.getvalue()))
        try:
            pages, chunks = await self._index_file(file)
            return UploadResult(
                file=file,
                created=True,
                message=f"File '{u_file.name}' indexed successfully.",
                pages=pages,
                chunks=chunks,
            )
        except Exception:
            await self.file_repository.delete(file.id)
            raise

    async def reindex(self, file_id: uuid.UUID) -> UploadResult:
        file = await self.get_by_id(file_id)
        await self._delete_vectors_for_file(file)
        pages, chunks = await self._index_file(file)
        return UploadResult(
            file=file,
            created=True,
            message=f"Re-indexed '{file.name}'.",
            pages=pages,
            chunks=chunks,
        )

    async def search_documents(self, query: str, chat_id: uuid.UUID) -> list[Document]:
        file_ids = await self.find_files_ids(chat_id=chat_id)
        if not file_ids:
            return []
        return await self.vector_store.asimilarity_search(query, filter={"file_id": {"$in": file_ids}})

    async def find_files_ids(self, chat_id: uuid.UUID) -> list[str]:
        files = await self.all(conditions={"chat_id": chat_id})
        return [str(file.id) for file in files]
