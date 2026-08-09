"""Tests for the Phase 7 ChunkRunner (DB streaming) and its Celery task."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.ingestion as ingestion_module
from app.core.exceptions import ChunkConflictError
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import (
    ChunkRepository,
    IngestionJobRepository,
    UploadRepository,
)
from app.services.chunking import ChunkRunner
from app.services.extraction import ExtractionRunner
from app.services.metadata import MetadataRunner
from app.worker.celery_app import celery_app
from tests.support import build_structured_book


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


@pytest.fixture()
def storage(tmp_path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "uploads")


def _extract_and_metadata(
    session: Session,
    storage: LocalStorageProvider,
    pdf_path: Path,
) -> tuple[Upload, IngestionJob, IngestionJob]:
    key = "book.pdf"
    with storage.writer(key) as writer:
        writer.write(pdf_path.read_bytes())
    upload = UploadRepository(session).create(
        Upload(
            original_filename="Al-Mabsut v1.pdf",
            filename=key,
            storage_path=key,
            mime="application/pdf",
            status="completed",
        )
    )
    extraction_job = IngestionJobRepository(session).create_for_upload(upload.id)
    ExtractionRunner(session, storage).run(extraction_job, upload)
    metadata_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload.id, kind="metadata", status="queued")
    )
    MetadataRunner(session).run(metadata_job, upload)
    IngestionJobRepository(session).update_status(
        metadata_job, "completed", progress_percent=100, current_step="metadata"
    )
    session.commit()
    session.expire_all()
    return (
        upload,
        IngestionJobRepository(session).get(extraction_job.id),
        IngestionJobRepository(session).get(metadata_job.id),
    )


def _create_chunking_job(session: Session, upload: Upload) -> IngestionJob:
    return IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload.id, kind="chunking", status="queued")
    )


def test_runner_persists_structure_aware_chunks(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    result = ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    session.expire_all()

    assert result.page_count == 4
    assert result.chunk_count == 4
    assert result.pages_covered == 4
    assert result.token_count > 0

    chunks = ChunkRepository(session).list_by_job(chunk_job.id)
    assert len(chunks) == 4
    assert [chunk.order_index for chunk in chunks] == [0, 1, 2, 3]

    by_page = {chunk.pdf_page_start: chunk for chunk in chunks}
    cover = by_page[1]
    assert cover.kitab is None
    assert cover.bab is None
    assert cover.raw_text.startswith("Al-Mabsut")

    kitab_chunk = by_page[2]
    assert kitab_chunk.kitab == "Kitab al-Taharah"
    assert kitab_chunk.bab is None
    assert kitab_chunk.context_heading == "Kitab: Kitab al-Taharah"
    assert kitab_chunk.raw_text.startswith("Kitab: Kitab al-Taharah")

    bab_chunk = by_page[3]
    assert bab_chunk.kitab == "Kitab al-Taharah"
    assert bab_chunk.bab == "Bab al-Wudu"
    assert bab_chunk.context_heading == "Kitab: Kitab al-Taharah\nBab: Bab al-Wudu"
    assert bab_chunk.printed_page_start == 2
    assert bab_chunk.region == "main"
    assert bab_chunk.lang == "ar"
    assert bab_chunk.raw_text
    assert bab_chunk.token_count > 0


def test_runner_replaces_previous_chunks(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    session.expire_all()

    chunks = ChunkRepository(session).list_by_job(chunk_job.id)
    assert len(chunks) == 4
    assert len({chunk.chunk_id for chunk in chunks}) == 4


def test_runner_updates_job_progress_to_completion(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    session.expire_all()

    done = IngestionJobRepository(session).get(chunk_job.id)
    assert done.status == "structuring"
    assert done.current_step == "chunking"
    assert done.progress_percent == 100
    assert done.started_at is not None


def test_runner_fails_without_metadata_document(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    key = "book.pdf"
    with storage.writer(key) as writer:
        writer.write(pdf.read_bytes())
    upload = UploadRepository(session).create(
        Upload(
            original_filename="Al-Mabsut v1.pdf",
            filename=key,
            storage_path=key,
            mime="application/pdf",
            status="completed",
        )
    )
    extraction_job = IngestionJobRepository(session).create_for_upload(upload.id)
    ExtractionRunner(session, storage).run(extraction_job, upload)
    session.commit()
    session.expire_all()

    chunk_job = _create_chunking_job(session, upload)
    with pytest.raises(ChunkConflictError):
        ChunkRunner(session).run(chunk_job, upload)


def test_chunking_task_completes_job_and_upload(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_chunking_task.delay(chunk_job.id, upload.id, metadata_job.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(chunk_job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "completed"
    assert done_job.progress_percent == 100
    assert done_job.current_step == "chunking"
    assert done_job.errors == []
    assert done_upload.status == "completed"
    assert len(ChunkRepository(session).list_by_job(chunk_job.id)) == 4


def test_chunking_task_records_error_and_fails_job(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    def boom(_job: IngestionJob, _upload: Upload, *, metadata_job_id=None):
        raise RuntimeError("db exploded")

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(
        ingestion_module, "ChunkRunner", lambda _session: type("Broken", (), {"run": boom})()
    )

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_chunking_task.delay(chunk_job.id, upload.id, metadata_job.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(chunk_job.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert any(error.step == "chunking" for error in done_job.errors)


def test_chunking_task_fails_on_missing_metadata(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, extraction_job, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = _create_chunking_job(session, upload)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_chunking_task.delay(
            chunk_job.id, upload.id, "00000000000000000000000000000000"
        )
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(chunk_job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert "metadata" in done_job.error_message
    assert any(error.step == "chunking" for error in done_job.errors)
    assert done_upload.status == "failed"
