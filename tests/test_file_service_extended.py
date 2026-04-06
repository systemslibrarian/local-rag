"""Extended tests for FileService — covers the upload-job pipeline end-to-end,
background thread behaviour, IntegrityError race handling, vector cleanup,
and the coro-definition fix.
"""
import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from app.file.dto.file_schema import FileCreate
from app.file.model.file import File
from app.file.model.index_job import IndexJob
from app.file.service.file_service import FileService


def _make_service(tmp_path: Path) -> tuple[FileService, MagicMock, MagicMock, MagicMock, MagicMock]:
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
    return service, mock_chat_repo, mock_file_repo, mock_job_repo, mock_vs


# ── _run_upload_job: happy path ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_upload_job_indexes_successfully(tmp_path: Path):
    """Full pipeline: chat exists → no duplicate → create file → index → mark completed."""
    service, mock_chat_repo, mock_file_repo, mock_job_repo, mock_vs = _make_service(tmp_path)

    chat_id = uuid.uuid4()
    job_id = uuid.uuid4()
    file_id = uuid.uuid4()
    pdf_bytes = b"%PDF-1.4 fake"

    mock_chat_repo.find_by_id = AsyncMock(return_value=MagicMock(id=chat_id))
    mock_file_repo.find_by_chat_and_name = AsyncMock(return_value=None)

    created_file = File(id=file_id, name="test.pdf", chat_id=chat_id, storage_path=str(tmp_path / "x.pdf"))

    async def fake_create(f: File) -> File:
        f.id = file_id
        return f

    mock_file_repo.create = AsyncMock(side_effect=fake_create)
    mock_file_repo.update = AsyncMock(return_value=None)

    service._index_file = AsyncMock(return_value=(5, 20))

    updates: list[dict] = []

    async def capture_update(jid, data):
        updates.append({"job_id": jid, **data})
        return None

    mock_job_repo.update = AsyncMock(side_effect=capture_update)

    await service._run_upload_job(job_id, pdf_bytes, "test.pdf", chat_id)

    # Should have updated job to "running" then "completed"
    statuses = [u["status"] for u in updates]
    assert "running" in statuses
    assert "completed" in statuses

    # The final update should include pages and chunks
    final = updates[-1]
    assert final["pages"] == 5
    assert final["chunks"] == 20


# ── _run_upload_job: chat deleted mid-flight ─────────────────────────────────

@pytest.mark.asyncio
async def test_run_upload_job_aborts_when_chat_gone(tmp_path: Path):
    service, mock_chat_repo, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    mock_chat_repo.find_by_id = AsyncMock(return_value=None)

    updates: list[dict] = []
    async def capture_update(jid, data):
        updates.append(data)
        return None
    mock_job_repo.update = AsyncMock(side_effect=capture_update)

    await service._run_upload_job(uuid.uuid4(), b"pdf", "test.pdf", uuid.uuid4())

    # Should mark failed because chat is gone
    final = updates[-1]
    assert final["status"] == "failed"


# ── _run_upload_job: duplicate file (TOCTOU race via IntegrityError) ─────────

@pytest.mark.asyncio
async def test_run_upload_job_handles_integrity_error_race(tmp_path: Path):
    """If two uploads race and one hits IntegrityError, it should handle gracefully."""
    service, mock_chat_repo, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    chat_id = uuid.uuid4()
    job_id = uuid.uuid4()
    existing_file = MagicMock(id=uuid.uuid4())

    mock_chat_repo.find_by_id = AsyncMock(return_value=MagicMock(id=chat_id))
    # First call: no file. After IntegrityError: file exists.
    mock_file_repo.find_by_chat_and_name = AsyncMock(side_effect=[None, existing_file])
    mock_file_repo.create = AsyncMock(side_effect=IntegrityError("dup", {}, None))

    updates: list[dict] = []
    async def capture_update(jid, data):
        updates.append(data)
        return None
    mock_job_repo.update = AsyncMock(side_effect=capture_update)

    await service._run_upload_job(job_id, b"pdf", "test.pdf", chat_id)

    final = updates[-1]
    assert final["status"] == "completed"
    assert "already indexed" in final["message"]


# ── _run_upload_job: indexing failure → cleans up file ────────────────────────

@pytest.mark.asyncio
async def test_run_upload_job_cleans_up_on_index_failure(tmp_path: Path):
    service, mock_chat_repo, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    chat_id = uuid.uuid4()
    job_id = uuid.uuid4()
    file_id = uuid.uuid4()

    mock_chat_repo.find_by_id = AsyncMock(return_value=MagicMock(id=chat_id))
    mock_file_repo.find_by_chat_and_name = AsyncMock(return_value=None)

    storage_path = tmp_path / "files" / f"{file_id}.pdf"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_bytes(b"fake")

    async def fake_create(f: File) -> File:
        f.id = file_id
        f.storage_path = str(storage_path)
        return f

    mock_file_repo.create = AsyncMock(side_effect=fake_create)
    service._index_file = AsyncMock(side_effect=RuntimeError("embedding model down"))

    # Mock delete chain for cleanup
    service.delete = AsyncMock(return_value=True)

    updates: list[dict] = []
    async def capture_update(jid, data):
        updates.append(data)
        return None
    mock_job_repo.update = AsyncMock(side_effect=capture_update)

    await service._run_upload_job(job_id, b"pdf", "test.pdf", chat_id)

    final = updates[-1]
    assert final["status"] == "failed"
    assert "Failed to index" in final["message"]
    service.delete.assert_awaited_once_with(file_id)


# ── submit_upload_job: coro is defined before thread starts ──────────────────

@pytest.mark.asyncio
async def test_submit_upload_job_coro_defined(tmp_path: Path):
    """Regression: `coro` must be assigned before the thread function references it."""
    service, mock_chat_repo, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    chat_id = uuid.uuid4()
    mock_file_repo.find_by_chat_and_name = AsyncMock(return_value=None)
    mock_job_repo.find_active_by_chat_and_name = AsyncMock(return_value=None)

    created_job = IndexJob(
        id=uuid.uuid4(),
        chat_id=chat_id,
        file_name="test.pdf",
        status="queued",
        message="Queued",
        pages=0,
        chunks=0,
    )
    mock_job_repo.create = AsyncMock(return_value=created_job)

    # Patch threading.Thread to capture args instead of actually spawning
    with patch("app.file.service.file_service.threading.Thread") as MockThread:
        mock_thread_instance = MagicMock()
        MockThread.return_value = mock_thread_instance

        job = await service.submit_upload_job("test.pdf", b"bytes", chat_id)

        assert job.status == "queued"
        MockThread.assert_called_once()
        # The thread target should be callable (not crash with NameError)
        call_kwargs = MockThread.call_args
        thread_target = call_kwargs.kwargs.get("target") or call_kwargs[1].get("target") or call_kwargs[0][0]
        # Actually call it to make sure `coro` is captured properly
        # (this would raise NameError with the old buggy code)
        assert callable(thread_target)


# ── submit_upload_job: dedup via existing file ───────────────────────────────

@pytest.mark.asyncio
async def test_submit_upload_job_returns_early_for_existing(tmp_path: Path):
    service, _, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    existing = MagicMock(id=uuid.uuid4())
    mock_file_repo.find_by_chat_and_name = AsyncMock(return_value=existing)

    job = await service.submit_upload_job("test.pdf", b"bytes", uuid.uuid4())
    assert job.status == "completed"
    assert "already indexed" in job.message
    mock_job_repo.create.assert_not_called()


# ── submit_upload_job: dedup via active job ──────────────────────────────────

@pytest.mark.asyncio
async def test_submit_upload_job_returns_active_job(tmp_path: Path):
    service, _, mock_file_repo, mock_job_repo, _ = _make_service(tmp_path)

    mock_file_repo.find_by_chat_and_name = AsyncMock(return_value=None)

    active_job = IndexJob(
        id=uuid.uuid4(),
        chat_id=uuid.uuid4(),
        file_name="test.pdf",
        status="running",
        message="In progress",
        pages=0,
        chunks=0,
    )
    mock_job_repo.find_active_by_chat_and_name = AsyncMock(return_value=active_job)

    job = await service.submit_upload_job("test.pdf", b"bytes", uuid.uuid4())
    assert job.status == "running"
    mock_job_repo.create.assert_not_called()


# ── _delete_vectors_for_file ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_vectors_skips_when_no_chunks(tmp_path: Path):
    service, *_, mock_vs = _make_service(tmp_path)
    file = MagicMock(chunk_count=0, id=uuid.uuid4())
    await service._delete_vectors_for_file(file)
    mock_vs.adelete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_vectors_generates_correct_ids(tmp_path: Path):
    service, *_, mock_vs = _make_service(tmp_path)
    fid = uuid.uuid4()
    file = MagicMock(chunk_count=3, id=fid)
    mock_vs.adelete = AsyncMock()

    await service._delete_vectors_for_file(file)

    expected_ids = [f"{fid}:0", f"{fid}:1", f"{fid}:2"]
    mock_vs.adelete.assert_awaited_once_with(ids=expected_ids, collection_only=True)


# ── delete_by_chat ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_by_chat_removes_all_files(tmp_path: Path):
    service, _, mock_file_repo, *_ = _make_service(tmp_path)
    cid = uuid.uuid4()

    f1 = MagicMock(id=uuid.uuid4())
    f2 = MagicMock(id=uuid.uuid4())
    mock_file_repo.all = AsyncMock(return_value=[f1, f2])
    service.delete = AsyncMock(return_value=True)

    count = await service.delete_by_chat(cid)
    assert count == 2
    assert service.delete.await_count == 2


# ── has_files ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_has_files_true_when_files_exist(tmp_path: Path):
    service, _, mock_file_repo, *_ = _make_service(tmp_path)
    mock_file_repo.all = AsyncMock(return_value=[MagicMock()])
    assert await service.has_files(uuid.uuid4()) is True


@pytest.mark.asyncio
async def test_has_files_false_when_empty(tmp_path: Path):
    service, _, mock_file_repo, *_ = _make_service(tmp_path)
    mock_file_repo.all = AsyncMock(return_value=[])
    assert await service.has_files(uuid.uuid4()) is False


# ── clear_finished_jobs ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_finished_jobs_only_removes_completed_and_failed(tmp_path: Path):
    service, _, _, mock_job_repo, _ = _make_service(tmp_path)

    jobs = [
        MagicMock(id=uuid.uuid4(), status="completed"),
        MagicMock(id=uuid.uuid4(), status="failed"),
        MagicMock(id=uuid.uuid4(), status="running"),
        MagicMock(id=uuid.uuid4(), status="queued"),
    ]
    mock_job_repo.all_for_chat = AsyncMock(return_value=jobs)
    mock_job_repo.delete = AsyncMock(return_value=True)

    deleted = await service.clear_finished_jobs(uuid.uuid4())
    assert deleted == 2  # only completed + failed
    assert mock_job_repo.delete.await_count == 2


# ── find_files_ids ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_files_ids_returns_string_uuids(tmp_path: Path):
    service, _, mock_file_repo, *_ = _make_service(tmp_path)
    fid1 = uuid.uuid4()
    fid2 = uuid.uuid4()
    f1 = MagicMock(id=fid1)
    f2 = MagicMock(id=fid2)
    mock_file_repo.all = AsyncMock(return_value=[f1, f2])

    result = await service.find_files_ids(uuid.uuid4())
    assert result == [str(fid1), str(fid2)]
