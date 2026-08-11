"""Tests for the Phase 8 IndexingRunner (DB streaming → Qdrant) and its Celery task."""

from pathlib import Path

import pytest
from qdrant_client import models
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.ingestion as ingestion_module
from app.core.exceptions import IndexConflictError
from app.core.qdrant import SPARSE_VECTOR_NAME
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import (
    ChunkRepository,
    IngestionJobRepository,
    MetadataRepository,
    UploadRepository,
)
from app.services.chunking import ChunkRunner
from app.services.embedding import DeterministicEmbedder
from app.services.extraction import ExtractionRunner
from app.services.indexing import IndexingRunner
from app.services.metadata import MetadataRunner
from app.worker.celery_app import celery_app
from tests.support import build_structured_book

PAYLOAD_KEYS = {
    "chunk_id",
    "text",
    "book_id",
    "book_name",
    "author",
    "volume",
    "printed_page_start",
    "printed_page_end",
    "pdf_page_start",
    "pdf_page_end",
    "kitab",
    "bab",
    "fasl",
    "topic",
    "region",
    "lang",
    "edition_id",
    "publisher",
    "year",
    "verified",
    "upload_id",
    "job_id",
}


class FakeStore:
    def __init__(self) -> None:
        self.deleted_filters: list[models.Filter] = []
        self.upsert_batches: list[list[models.PointStruct]] = []

    def delete_by_filter(self, filter_: models.Filter) -> None:
        self.deleted_filters.append(filter_)

    def upsert_points(self, points: list[models.PointStruct]) -> None:
        self.upsert_batches.append(points)


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
) -> tuple[Upload, IngestionJob]:
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
    return upload, IngestionJobRepository(session).get(metadata_job.id)


def _chunked_upload(
    session: Session, storage: LocalStorageProvider, pdf_path: Path
) -> tuple[Upload, IngestionJob]:
    upload, metadata_job = _extract_and_metadata(session, storage, pdf_path)
    chunk_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload.id, kind="chunking", status="queued")
    )
    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    IngestionJobRepository(session).update_status(
        chunk_job, "completed", progress_percent=100, current_step="chunking"
    )
    session.commit()
    session.expire_all()
    return upload, IngestionJobRepository(session).get(chunk_job.id)


def _index_job(session: Session, upload: Upload) -> IngestionJob:
    return IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload.id, kind="indexing", status="queued")
    )


def test_runner_indexes_every_chunk_with_contract_payload(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()

    result = IndexingRunner(session, store).run(index_job, upload, chunking_job_id=chunk_job.id)
    session.expire_all()

    assert result.page_count == 4
    assert result.chunk_count == 4
    assert result.vectors_indexed == 4

    points = [point for batch in store.upsert_batches for point in batch]
    assert len(points) == 4
    ids = {point.id for point in points}
    assert ids == {chunk.chunk_id for chunk in ChunkRepository(session).list_by_job(chunk_job.id)}

    point = points[0]
    assert set(point.vector) == {"", SPARSE_VECTOR_NAME}
    assert isinstance(point.vector[SPARSE_VECTOR_NAME], models.SparseVector)
    assert set(point.payload) == PAYLOAD_KEYS
    assert point.payload["upload_id"] == upload.id
    assert point.payload["job_id"] == chunk_job.id
    assert point.payload["book_name"] == "Al-Mabsut"
    assert point.payload["author"] is not None
    assert point.payload["region"] == "main"
    assert point.payload["lang"] == "ar"
    assert point.payload["verified"] is False
    assert point.payload["text"]


def test_runner_deletes_then_upserts_for_reindex(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()

    IndexingRunner(session, store).run(index_job, upload, chunking_job_id=chunk_job.id)
    IndexingRunner(session, store).run(index_job, upload, chunking_job_id=chunk_job.id)

    assert len(store.deleted_filters) == 2
    filter_ = store.deleted_filters[0]
    assert isinstance(filter_.must, list)
    assert filter_.must[0].key == "upload_id"
    assert filter_.must[0].match.value == upload.id


def test_runner_upserts_in_batches(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()

    IndexingRunner(session, store, batch_size=3).run(
        index_job, upload, chunking_job_id=chunk_job.id
    )

    assert len(store.upsert_batches) == 2
    assert sum(len(batch) for batch in store.upsert_batches) == 4


class RecordingEmbedder:
    """Records `embed_batch` sizes and texts for the Phase 15 batching contract."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.texts: list[str] = []
        self._deterministic = DeterministicEmbedder(dim=64)

    def embed(self, text: str):
        raise AssertionError("batched indexing must use embed_batch, not embed")

    def embed_batch(self, texts: list[str]):
        self.batch_sizes.append(len(texts))
        self.texts.extend(texts)
        return [self._deterministic.embed(text) for text in texts]


def test_runner_embeds_in_batches_preserving_order(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()
    embedder = RecordingEmbedder()

    IndexingRunner(session, store, embedder=embedder, embedding_batch_size=2).run(
        index_job, upload, chunking_job_id=chunk_job.id
    )

    # 4 chunks embedded in batches of 2, in chunk order.
    assert embedder.batch_sizes == [2, 2]
    chunks = ChunkRepository(session).list_by_job(chunk_job.id)
    assert embedder.texts == [chunk.raw_text for chunk in chunks]

    # Point order mirrors input chunk order (ordering preserved end to end).
    points = [point for batch in store.upsert_batches for point in batch]
    assert [point.id for point in points] == [chunk.chunk_id for chunk in chunks]


def test_runner_fails_without_chunking_job(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, _ = _extract_and_metadata(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()

    with pytest.raises(IndexConflictError):
        IndexingRunner(session, store).run(index_job, upload)

    assert store.upsert_batches == []
    assert store.deleted_filters == []


def test_runner_fails_without_metadata_document(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, metadata_job = _extract_and_metadata(session, storage, pdf)
    chunk_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload.id, kind="chunking", status="queued")
    )
    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job.id)
    MetadataRepository(session).delete_for_job(metadata_job.id)
    session.expire_all()

    index_job = _index_job(session, upload)
    store = FakeStore()

    with pytest.raises(IndexConflictError):
        IndexingRunner(session, store).run(index_job, upload, chunking_job_id=chunk_job.id)

    assert store.upsert_batches == []
    assert store.deleted_filters == []


def test_runner_updates_job_progress_and_upload_status(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)

    IndexingRunner(session, FakeStore()).run(index_job, upload, chunking_job_id=chunk_job.id)
    session.expire_all()

    done_job = IngestionJobRepository(session).get(index_job.id)
    assert done_job.status == "embedding"
    assert done_job.current_step == "embedding"
    assert done_job.progress_percent == 100
    assert done_job.started_at is not None
    assert UploadRepository(session).get(upload.id).status == "processing"


def test_indexing_task_completes_job_and_upload(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)
    store = FakeStore()

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(ingestion_module, "get_qdrant_store", lambda: store)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_indexing_task.delay(index_job.id, upload.id, chunk_job.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(index_job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "indexed"
    assert done_job.progress_percent == 100
    assert done_job.current_step == "embedding"
    assert done_job.errors == []
    assert done_upload.status == "completed"
    assert sum(len(batch) for batch in store.upsert_batches) == 4


def test_indexing_task_records_error_and_fails_job(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, chunk_job = _chunked_upload(session, storage, pdf)
    index_job = _index_job(session, upload)

    def boom(_job: IngestionJob, _upload: Upload, *, chunking_job_id=None):
        raise RuntimeError("qdrant exploded")

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(
        ingestion_module,
        "IndexingRunner",
        lambda _session, _store: type("Broken", (), {"run": boom})(),
    )

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_indexing_task.delay(index_job.id, upload.id, chunk_job.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(index_job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert any(error.step == "indexing" for error in done_job.errors)
    assert done_upload.status == "failed"


def test_indexing_task_fails_permanently_without_chunks(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, _ = _extract_and_metadata(session, storage, pdf)
    index_job = _index_job(session, upload)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_indexing_task.delay(
            index_job.id, upload.id, "00000000000000000000000000000000"
        )
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(index_job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert "chunking" in done_job.error_message
    assert any(error.step == "indexing" for error in done_job.errors)
    assert done_upload.status == "failed"
