"""Unit tests for FileService.

Uses tmp_path (pytest fixture) for real disk I/O tests and mocks for
database / vector-store interactions.
"""
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.file.dto.file_schema import FileCreate
from app.file.model.file import File
from app.file.service.file_service import FileService, UploadResult


def _make_service(tmp_path: Path) -> tuple[FileService, MagicMock, MagicMock, MagicMock]:
    """Return (service, mock_file_repo, mock_job_repo, mock_vector_store)."""
    mock_chat_repo = MagicMock()
    mock_file_repo = MagicMock()
    mock_job_repo = MagicMock()
    mock_splitter = MagicMock()
    mock_vs = MagicMock()

    service = FileService(
        chat_repository=mock_chat_repo,
        file_repository=mock_file_repo,
        index_job_repository=mock_job_repo,
        text_specifier=mock_splitter,
        vector_store=mock_vs,
        storage_folder=str(tmp_path / "files"),
    )
    return service, mock_file_repo, mock_job_repo, mock_vs


# ── build_chunk_id ────────────────────────────────────────────────────────────

def test_build_chunk_id_format():
    fid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    result = FileService.build_chunk_id(fid, 3)
    assert result == "12345678-1234-5678-1234-567812345678:3"


# ── _file_storage_path ────────────────────────────────────────────────────────

def test_file_storage_path_creates_dir(tmp_path: Path):
    service, *_ = _make_service(tmp_path)
    fid = uuid.uuid4()
    path = service._file_storage_path(fid)
    assert path.parent.exists()
    assert path.name == f"{fid}.pdf"


# ── create() writes bytes to disk ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_writes_pdf_to_disk(tmp_path: Path):
    service, mock_file_repo, *_ = _make_service(tmp_path)

    pdf_bytes = b"%PDF-1.4 fake content"
    chat_id = uuid.uuid4()

    # Mock the repo to return a File with the same id we assign
    async def fake_create(f: File) -> File:
        return f

    mock_file_repo.create = AsyncMock(side_effect=fake_create)

    file_create = FileCreate(name="test.pdf", chat_id=chat_id, content=pdf_bytes)
    result = await service.create(file_create)

    assert Path(result.storage_path).exists()
    assert Path(result.storage_path).read_bytes() == pdf_bytes


# ── delete() removes disk file ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_removes_disk_file(tmp_path: Path):
    service, mock_file_repo, *_ = _make_service(tmp_path)

    fid = uuid.uuid4()
    storage_path = service._file_storage_path(fid)
    storage_path.write_bytes(b"fake pdf")

    dummy_file = File(
        id=fid,
        name="test.pdf",
        chat_id=uuid.uuid4(),
        storage_path=str(storage_path),
    )

    mock_file_repo.get_by_id = AsyncMock(return_value=dummy_file)
    mock_file_repo.delete = AsyncMock(return_value=True)

    # _delete_vectors_for_file calls _split_file_content — mock it to avoid PDF loading
    service._delete_vectors_for_file = AsyncMock(return_value=None)

    result = await service.delete(fid)

    assert result is True
    assert not storage_path.exists()


# ── submit_upload_job deduplication ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_upload_job_returns_completed_for_duplicate(tmp_path: Path):
    service, mock_file_repo, mock_job_repo, *_ = _make_service(tmp_path)

    existing_file = MagicMock()
    existing_file.id = uuid.uuid4()
    service.find_by_chat_and_name = AsyncMock(return_value=existing_file)

    job = await service.submit_upload_job(
        file_name="report.pdf",
        pdf_bytes=b"bytes",
        chat_id=uuid.uuid4(),
    )

    assert job.status == "completed"
    assert "already indexed" in job.message
    # No new DB record should be created
    mock_job_repo.create.assert_not_called()


# ── reindex() ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reindex_returns_upload_result(tmp_path: Path):
    service, mock_file_repo, *_ = _make_service(tmp_path)

    fid = uuid.uuid4()
    storage_path = service._file_storage_path(fid)
    storage_path.write_bytes(b"fake pdf")

    dummy_file = File(
        id=fid,
        name="report.pdf",
        chat_id=uuid.uuid4(),
        storage_path=str(storage_path),
    )
    mock_file_repo.get_by_id = AsyncMock(return_value=dummy_file)

    service._delete_vectors_for_file = AsyncMock(return_value=None)
    service._index_file = AsyncMock(return_value=(10, 42))  # pages, chunks

    result = await service.reindex(fid)

    assert isinstance(result, UploadResult)
    assert result.pages == 10
    assert result.chunks == 42
    assert result.created is True
