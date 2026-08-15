"""Tests for the Phase 4 extraction API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
from app.api.v1 import deps
from app.api.v1.endpoints import extraction as extraction_endpoints
from app.core.config import Settings, get_settings
from app.core.postgres import get_db
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob
from app.db.repositories import (
    ExtractionRepository,
    IngestionJobRepository,
    UploadRepository,
)
from app.main import app
from app.services.extraction import ExtractionRunner
from tests.support import build_scanned_pdf, build_text_pdf, make_pdf_bytes


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def delay(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def _no_celery(monkeypatch) -> FakeTask:
    task = FakeTask()
    monkeypatch.setattr(uploads_module, "mark_queued", task)
    monkeypatch.setattr(extraction_endpoints, "process_book_task", task)
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


def _upload_pdf(client: TestClient, data: bytes, name: str = "kitab.pdf") -> str:
    response = client.post(
        "/api/v1/uploads",
        files={"files": (name, data, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["id"]


def _run_extraction(
    client: TestClient,
    session: Session,
    storage: LocalStorageProvider,
    upload_id: str,
    *,
    complete: bool = True,
) -> IngestionJob:
    upload = UploadRepository(session).get(upload_id)
    job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    assert job is not None
    ExtractionRunner(session, storage).run(job, upload)
    if complete:
        IngestionJobRepository(session).update_status(
            job, "completed", progress_percent=100, current_step="extracting"
        )
        upload.status = "completed"
        session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(job.id)


def test_start_extraction_queues_job(client: TestClient, session: Session, tmp_path) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())

    response = client.post(f"/api/v1/extraction/{upload_id}")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["upload_id"] == upload_id

    job = IngestionJobRepository(session).get(body["job_id"])
    assert job is not None
    assert job.status == "queued"
    assert job.upload_id == upload_id
    assert ExtractionRepository(session).count_pages(job.id) == 0


def test_start_extraction_dispatches_worker(
    client: TestClient, session: Session, tmp_path, _no_celery: FakeTask
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())

    response = client.post(f"/api/v1/extraction/{upload_id}")
    body = response.json()
    assert _no_celery.calls[-1] == (
        (upload_id,),
        {"stage": "extraction", "job_id": body["job_id"]},
    )


def test_start_extraction_missing_upload_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/extraction/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_start_extraction_missing_stored_file_returns_409(
    client: TestClient, session: Session, storage: LocalStorageProvider
) -> None:
    data = make_pdf_bytes()
    upload_id = _upload_pdf(client, data)
    upload = UploadRepository(session).get(upload_id)
    assert upload is not None and upload.storage_path is not None
    storage.delete(upload.storage_path)

    response = client.post(f"/api/v1/extraction/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "extraction_conflict"


def test_start_extraction_conflicts_while_in_flight(
    client: TestClient, session: Session, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    assert job is not None
    job.status = "extracting"
    session.commit()

    response = client.post(f"/api/v1/extraction/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "extraction_conflict"


def test_get_extraction_status_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/extraction/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_extraction_status_after_complete_run(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    job = _run_extraction(client, session, storage, upload_id)

    response = client.get(f"/api/v1/extraction/{job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["upload_id"] == upload_id
    assert body["page_count"] == 2
    assert body["pages_extracted"] == 2
    assert body["has_text_layer"] is True
    assert body["extraction_confidence"] > 0
    assert body["char_count"] > 0
    assert body["block_count"] == 8
    assert body["errors"] == []


def test_get_extraction_status_scanned_pdf_has_no_text_layer(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_scanned_pdf(tmp_path / "scanned.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    job = _run_extraction(client, session, storage, upload_id)

    response = client.get(f"/api/v1/extraction/{job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["has_text_layer"] is False
    assert body["char_count"] == 0
    assert body["extraction_confidence"] == 0
    assert body["image_count"] == 1


def test_get_extraction_status_lists_errors(client: TestClient, session: Session, tmp_path) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    assert job is not None
    job_repo = IngestionJobRepository(session)
    job_repo.add_error(job, "extracting", "boom")
    job_repo.update_status(job, "failed", error_message="boom")

    response = client.get(f"/api/v1/extraction/{job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error_message"] == "boom"
    assert len(body["errors"]) == 1
    assert body["errors"][0]["message"] == "boom"
    assert body["errors"][0]["step"] == "extracting"
