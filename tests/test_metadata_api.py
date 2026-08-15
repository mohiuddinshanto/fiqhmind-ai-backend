"""Tests for the Phase 6 metadata API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
from app.api.v1 import deps
from app.api.v1.endpoints import metadata as metadata_endpoints
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
    monkeypatch.setattr(metadata_endpoints, "run_metadata_task", task)
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
    job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    assert job is not None
    ExtractionRunner(session, storage).run(job, upload)
    IngestionJobRepository(session).update_status(
        job, "completed", progress_percent=100, current_step="extracting"
    )
    upload.status = "completed"
    session.commit()
    session.expire_all()
    return IngestionJobRepository(session).get(job.id)


def test_start_metadata_queues_job(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)

    response = client.post(f"/api/v1/metadata/{upload_id}")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["upload_id"] == upload_id

    job = IngestionJobRepository(session).get(body["job_id"])
    assert job is not None
    assert job.kind == "metadata"
    assert job.status == "queued"
    assert job.upload_id == upload_id


def test_start_metadata_dispatches_worker(
    client: TestClient,
    session: Session,
    storage: LocalStorageProvider,
    tmp_path,
    _no_celery: FakeTask,
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)

    response = client.post(f"/api/v1/metadata/{upload_id}")
    body = response.json()
    assert _no_celery.calls[-1] == ((body["job_id"], upload_id), {})


def test_start_metadata_missing_upload_returns_404(client: TestClient) -> None:
    response = client.post("/api/v1/metadata/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_start_metadata_requires_completed_extraction(client: TestClient, tmp_path) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())

    response = client.post(f"/api/v1/metadata/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "metadata_conflict"


def test_start_metadata_conflicts_while_in_flight(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    _complete_extraction(client, session, storage, upload_id)

    metadata_job = IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind="metadata", status="structuring")
    )
    assert metadata_job.id

    response = client.post(f"/api/v1/metadata/{upload_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "metadata_conflict"


def test_get_metadata_status_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/metadata/00000000000000000000000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_get_metadata_status_after_complete_run(
    client: TestClient, session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload_id = _upload_pdf(client, pdf.read_bytes())
    job = _complete_extraction(client, session, storage, upload_id)
    upload = UploadRepository(session).get(upload_id)

    MetadataRunner(session).run(job, upload)
    IngestionJobRepository(session).update_status(
        job, "completed", progress_percent=100, current_step="metadata"
    )
    session.commit()
    session.expire_all()

    response = client.get(f"/api/v1/metadata/{job.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["upload_id"] == upload_id
    assert body["page_count"] == 4
    assert body["pages_mapped"] == 4
    assert body["numbering_system"] == "latin"
    assert body["confidence"] > 0.0

    fields = {item["field"]: item for item in body["fields"]}
    assert fields["title"]["value"] == "Al-Mabsut"
    assert fields["volume"]["value"] == "1"
    assert fields["author"]["value"] == "Imam Sarakhsi"
    assert fields["publisher"]["value"] == "Dar al-Kutub"
    assert fields["publication_year"]["value"] == "1998"
    for item in fields.values():
        assert item["value"]
        assert 0.0 < item["confidence"] <= 1.0
        assert item["source"]

    by_pdf = {item["pdf_page"]: item for item in body["page_mapping"]}
    assert by_pdf[2]["printed_page"] == "1"
    assert by_pdf[2]["printed_page_numeric"] == 1
    assert by_pdf[2]["kitab"] == "Kitab al-Taharah"
    assert by_pdf[3]["bab"] == "Bab al-Wudu"
    assert by_pdf[1]["printed_page"] == ""
    assert by_pdf[1]["page_number_uncertain"] is True

    kitabs = [item for item in body["structures"] if item["level"] == "kitab"]
    assert len(kitabs) == 1
    assert kitabs[0]["name"] == "Kitab al-Taharah"
    assert kitabs[0]["page_start"] == 2
    assert kitabs[0]["page_end"] == 4
    assert body["errors"] == []
