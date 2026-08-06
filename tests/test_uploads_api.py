import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
from app.api.v1 import deps
from app.core.config import Settings, get_settings
from app.core.postgres import get_db
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.main import app
from tests.support import make_pdf_bytes, sha256_hex


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def delay(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def _no_celery(monkeypatch) -> None:
    monkeypatch.setattr(uploads_module, "mark_queued", FakeTask())


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
def client(session: Session, tmp_path) -> TestClient:
    storage = LocalStorageProvider(tmp_path / "uploads")
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


def test_single_upload_roundtrip(client: TestClient) -> None:
    data = make_pdf_bytes()
    response = client.post(
        "/api/v1/uploads",
        files={"files": ("kitab.pdf", data, "application/pdf")},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["original_filename"] == "kitab.pdf"
    assert payload["sha256"] == sha256_hex(data)
    assert payload["status"] == "queued"
    assert payload["mime"] == "application/pdf"
    assert payload["size"] == len(data)
    assert payload["filename"] == f"{payload['id']}.pdf"

    upload_id = payload["id"]
    detail = client.get(f"/api/v1/uploads/{upload_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["ingestion_job_id"] is not None
    assert [log["event"] for log in body["logs"]] == ["uploading", "queued"]


def test_single_upload_invalid_pdf_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/uploads",
        files={"files": ("fake.pdf", b"definitely not a pdf", "application/pdf")},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "upload_validation_error"


def test_duplicate_upload_returns_409(client: TestClient) -> None:
    data = make_pdf_bytes()
    first = client.post(
        "/api/v1/uploads",
        files={"files": ("a.pdf", data, "application/pdf")},
    )
    assert first.status_code == 201
    second = client.post(
        "/api/v1/uploads",
        files={"files": ("b.pdf", data, "application/pdf")},
    )
    assert second.status_code == 409
    body = second.json()
    assert body["error"]["code"] == "duplicate_upload"
    assert body["error"]["details"]["existing_upload_id"] == first.json()["id"]


def test_multi_file_batch_mixed_success(client: TestClient) -> None:
    data = make_pdf_bytes()
    response = client.post(
        "/api/v1/uploads",
        files=[
            ("files", ("ok.pdf", data, "application/pdf")),
            ("files", ("bad.pdf", b"garbage", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    results = body["results"]
    assert len(results) == 2
    ok = next(item for item in results if item["filename"] == "ok.pdf")
    bad = next(item for item in results if item["filename"] == "bad.pdf")
    assert ok["success"] is True
    assert ok["upload"]["status"] == "queued"
    assert bad["success"] is False
    assert bad["error"]["code"] == "upload_validation_error"


def test_list_uploads_paginates(client: TestClient) -> None:
    for index in range(3):
        data = make_pdf_bytes(extra=f"page {index}".encode())
        assert (
            client.post(
                "/api/v1/uploads",
                files={"files": (f"f{index}.pdf", data, "application/pdf")},
            ).status_code
            == 201
        )
    response = client.get("/api/v1/uploads", params={"skip": 1, "limit": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2


def test_delete_upload(client: TestClient) -> None:
    data = make_pdf_bytes()
    upload_id = client.post(
        "/api/v1/uploads",
        files={"files": ("kitab.pdf", data, "application/pdf")},
    ).json()["id"]
    response = client.delete(f"/api/v1/uploads/{upload_id}")
    assert response.status_code == 204
    assert client.get(f"/api/v1/uploads/{upload_id}").status_code == 404


def test_upload_requires_files(client: TestClient) -> None:
    response = client.post("/api/v1/uploads", files=[])
    assert response.status_code == 422
