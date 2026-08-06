"""Tests for the Phase 7 chunking API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
from app.api.v1 import deps
from app.api.v1.endpoints import chunking as chunking_endpoints
from app.core.config import Settings, get_settings
from app.core.postgres import get_db
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob
from app.db.repositories import (
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
    monkeypatch.setattr(chunking_endpoints, "run_chunking_task", task)
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


def _complete_extraction(
    client: TestClient, session: Session, storage: LocalStorageProvider, upload_id: str
) -> IngestionJob:
    upload = UploadRepository(session).get(upload_id)
    job = upload.ingestion_job
    assert job is not None
    ExtractionRunner(session, storage).run(job, upload)
    IngestionJobRepository(session).update_status(
        job, "completed", progress_percent=100, current_step="extracting"
    )
    upload.status = "completed"
    session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(job.id)


def _complete_metadata(
    client: TestClient, session: Session, storage: LocalStorageProvider, upload_id: str
) -> IngestionJob:
    upload = UploadRepository(session).get(upload_id)
    metadata_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="metadata", status="queued")
    )
    MetadataRunner(session).run(metadata_job, upload)
    IngestionJobRepository(session).update_status(
        metadata_job, "completed", progress_percent=100, current_step="metadata"
    )
    upload.status = "completed"
    session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(metadata_job.id)


def test_start_chunking_queues_job(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)
    _complete_metadata(client, session, storage, upload_id)

    response = client.post(f"/api/v1/chunking/{upload_id}")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["upload_id"] == upload_id

    job = IngestionJobRepository(session).get(body["job_id"])
    assert job is not None
    assert job.kind == "chunking"
    assert job.status == "queued"
    assert job.upload_id == upload_id


def test_start_chunking_dispatches_worker(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path,
    _no_celery: FakeTask,
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)
    metadata_job = _complete_metadata(client, session, storage, upload_id)

    response = client.post(f"/api/v1/chunking/{upload_id}")
    body = response.json()
    assert _no_celery.calls[-1] == ((body["job_id"], upload_id, metadata_job.id), {})


def test_start_chunking_missing_upload_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/chunking/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_start_chunking_requires_completed_extraction(client: TestClient, tmp_path) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())

    response = client.post(f"/api/v1/chunking/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chunking_conflict"


def test_start_chunking_requires_completed_metadata(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)

    response = client.post(f"/api/v1/chunking/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chunking_conflict"


def test_start_chunking_conflicts_while_in_flight(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)
    _complete_metadata(client, session, storage, upload_id)

    chunking_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="chunking", status="structuring")
    )
    assert chunking_job.id

    response = client.post(f"/api/v1/chunking/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "chunking_conflict"


def test_get_chunking_status_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/chunking/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_chunking_status_after_complete_run(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)
    metadata_job = _complete_metadata(client, session, storage, upload_id)
    upload = UploadRepository(session).get(upload_id)

    chunking_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="chunking", status="queued")
    )
    ChunkRunner(session).run(chunking_job, upload, metadata_job_id=metadata_job.id)
    IngestionJobRepository(session).update_status(
        chunking_job, "completed", progress_percent=100, current_step="chunking"
    )
    session.commit()
    session.expire_all()

    response = client.get(f"/api/v1/chunking/{chunking_job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["upload_id"] == upload_id
    assert body["page_count"] == 4
    assert body["chunks_count"] == 4
    assert body["pages_covered"] == 4
    assert body["tokens_count"] > 0
    assert body["errors"] == []

    by_page = {item["pdf_page_start"]: item for item in body["chunks"]}
    assert by_page[2]["kitab"] == "Kitab al-Taharah"
    assert by_page[2]["context_heading"] == "Kitab: Kitab al-Taharah"
    assert by_page[3]["bab"] == "Bab al-Wudu"
    assert by_page[3]["printed_page_start"] == 2
    for item in body["chunks"]:
        assert item["chunk_id"]
        assert item["raw_text"]
        assert item["normalized_text"]
        assert item["token_count"] > 0
        assert item["region"] == "main"
        assert item["lang"] == "ar"
