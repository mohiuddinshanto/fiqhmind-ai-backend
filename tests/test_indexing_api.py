"""Tests for the Phase 8 indexing API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
from app.api.v1 import deps
from app.api.v1.endpoints import indexing as indexing_endpoints
from app.core.config import Settings, get_settings
from app.core.postgres import get_db
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob
from app.db.repositories import (
    ChunkRepository,
    IngestionJobRepository,
    UploadRepository,
)
from app.main import app
from app.services.chunking import ChunkRunner
from app.services.extraction import ExtractionRunner
from app.services.metadata import MetadataRunner
from tests.support import build_structured_book


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def delay(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def _no_celery(monkeypatch) -> FakeTask:
    task = FakeTask()
    monkeypatch.setattr(uploads_module, "mark_queued", task)
    monkeypatch.setattr(indexing_endpoints, "process_book_task", task)
    return task


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


@pytest.fixture()
def client(session: Session, storage: LocalStorageProvider, tmp_path) -> TestClient:
    settings = Settings(
        upload_storage_path=str(tmp_path / "uploads"),
        upload_max_size_bytes=1024 * 1024,
        upload_chunk_size=64 * 1024,
    )

    def override_db():
        yield session

    def override_storage():
        return storage

    def override_settings():
        return settings

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[deps.get_storage_dep] = override_storage
    app.dependency_overrides[get_settings] = override_settings  # type: ignore[name-defined]
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload_pdf(client: TestClient, data: bytes, name: str = "Al-Mabsut v1.pdf") -> str:
    response = client.post(
        "/api/v1/uploads",
        files={"files": (name, data, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def _complete_pipeline(
    client: TestClient, session: Session, storage: LocalStorageProvider, upload_id: str
) -> IngestionJob:
    upload = UploadRepository(session).get(upload_id)
    extraction_job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    assert extraction_job is not None
    ExtractionRunner(session, storage).run(extraction_job, upload)
    IngestionJobRepository(session).update_status(
        extraction_job, "completed", progress_percent=100, current_step="extracting"
    )
    upload.status = "completed"
    session.commit()
    session.expire_all()

    metadata_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="metadata", status="queued")
    )
    MetadataRunner(session).run(metadata_job, upload)
    IngestionJobRepository(session).update_status(
        metadata_job, "completed", progress_percent=100, current_step="metadata"
    )
    session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(metadata_job.id)


def _complete_chunking(
    session: Session, storage: LocalStorageProvider, upload_id: str, metadata_job_id: str
) -> IngestionJob:
    upload = UploadRepository(session).get(upload_id)
    chunk_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="chunking", status="queued")
    )
    ChunkRunner(session).run(chunk_job, upload, metadata_job_id=metadata_job_id)
    IngestionJobRepository(session).update_status(
        chunk_job, "completed", progress_percent=100, current_step="chunking"
    )
    session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(chunk_job.id)


def test_start_indexing_queues_job(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    metadata_job = _complete_pipeline(client, session, storage, upload_id)
    _complete_chunking(session, storage, upload_id, metadata_job.id)

    response = client.post(f"/api/v1/indexing/{upload_id}")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["upload_id"] == upload_id

    job = IngestionJobRepository(session).get(body["job_id"])
    assert job is not None
    assert job.kind == "indexing"
    assert job.status == "queued"
    assert job.upload_id == upload_id


def test_start_indexing_dispatches_worker(
    client: TestClient,
    session: Session,
    storage: LocalStorageProvider,
    tmp_path,
    _no_celery: FakeTask,
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    metadata_job = _complete_pipeline(client, session, storage, upload_id)
    chunk_job = _complete_chunking(session, storage, upload_id, metadata_job.id)

    response = client.post(f"/api/v1/indexing/{upload_id}")
    body = response.json()
    assert _no_celery.calls[-1] == (
        (upload_id,),
        {"stage": "indexing", "job_id": body["job_id"], "chunking_job_id": chunk_job.id},
    )


def test_start_indexing_missing_upload_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/indexing/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_start_indexing_requires_completed_chunking(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_pipeline(client, session, storage, upload_id)

    response = client.post(f"/api/v1/indexing/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "indexing_conflict"


def test_start_indexing_conflicts_while_in_flight(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    metadata_job = _complete_pipeline(client, session, storage, upload_id)
    _complete_chunking(session, storage, upload_id, metadata_job.id)

    indexing_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="indexing", status="embedding")
    )
    assert indexing_job.id

    response = client.post(f"/api/v1/indexing/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "indexing_conflict"


def test_start_indexing_reruns_after_terminal_status(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    metadata_job = _complete_pipeline(client, session, storage, upload_id)
    _complete_chunking(session, storage, upload_id, metadata_job.id)

    indexing_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="indexing", status="indexed")
    )
    indexing_job.finished_at = None
    session.commit()

    response = client.post(f"/api/v1/indexing/{upload_id}")
    assert response.status_code == 202
    assert response.json()["job_id"] == indexing_job.id

    refreshed = IngestionJobRepository(session).get(indexing_job.id)
    assert refreshed.status == "queued"
    assert refreshed.progress_percent == 0


def test_get_indexing_status_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/indexing/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_indexing_status_after_complete_run(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    metadata_job = _complete_pipeline(client, session, storage, upload_id)
    chunk_job = _complete_chunking(session, storage, upload_id, metadata_job.id)

    indexing_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="indexing", status="queued")
    )
    IngestionJobRepository(session).update_status(
        indexing_job, "indexed", progress_percent=100, current_step="embedding"
    )
    session.commit()
    session.expire_all()

    response = client.get(f"/api/v1/indexing/{indexing_job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "indexed"
    assert body["upload_id"] == upload_id
    assert body["page_count"] == 4
    assert body["chunks_count"] == 4
    assert body["vectors_indexed"] == 4
    assert body["errors"] == []
    assert len(ChunkRepository(session).list_by_job(chunk_job.id)) == 4
