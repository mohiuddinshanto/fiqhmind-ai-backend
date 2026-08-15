from io import BytesIO

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.uploads as uploads_module
import app.tasks.ingestion as ingestion_module
from app.core.config import Settings
from app.core.exceptions import DuplicateUploadError, UploadTooLargeError, UploadValidationError
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import Upload
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.services.uploads import UploadService
from tests.support import make_pdf_bytes, sha256_hex


class FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def delay(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class DispatchFakes:
    """Captures both background dispatches made on an accepted upload."""

    def __init__(self) -> None:
        self.mark_queued = FakeTask()
        self.pipeline = FakeTask()


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
def settings(tmp_path) -> Settings:
    return Settings(
        upload_storage_path=str(tmp_path / "uploads"),
        upload_max_size_bytes=1024 * 1024,
        upload_chunk_size=64 * 1024,
    )


@pytest.fixture()
def service(session: Session, storage: LocalStorageProvider, settings: Settings) -> UploadService:
    return UploadService(session=session, storage=storage, settings=settings)


@pytest.fixture()
def fake_task(monkeypatch) -> DispatchFakes:
    fakes = DispatchFakes()
    monkeypatch.setattr(uploads_module, "mark_queued", fakes.mark_queued)
    monkeypatch.setattr(uploads_module, "start_ingestion_pipeline_task", fakes.pipeline)
    return fakes


def test_receive_valid_pdf(
    service: UploadService, session: Session, fake_task: DispatchFakes
) -> None:
    data = make_pdf_bytes()
    upload = service.receive(
        BytesIO(data), original_filename="kitab.pdf", content_type="application/pdf"
    )

    assert upload.status == "queued"
    assert upload.sha256 == sha256_hex(data)
    assert upload.size == len(data)
    assert upload.mime == "application/pdf"
    assert upload.storage_path == f"{upload.id}.pdf"
    assert upload.page_count is None
    assert service._storage.exists(upload.storage_path)

    job = IngestionJobRepository(session).find_pipeline_job(upload.id)
    assert job is not None
    assert job.status == "uploaded"
    assert job.upload_id == upload.id

    assert [log.event for log in upload.logs] == ["uploading", "queued"]
    assert fake_task.mark_queued.calls == [((job.id, upload.id), {})]
    assert fake_task.pipeline.calls == [((upload.id,), {})]


def test_receive_rejects_bad_magic_bytes(
    service: UploadService, session: Session, storage: LocalStorageProvider
) -> None:
    with pytest.raises(UploadValidationError):
        service.receive(
            BytesIO(b"not a pdf at all"),
            original_filename="fake.pdf",
            content_type="application/pdf",
        )
    assert UploadRepository(session).count() == 0
    assert storage.exists("fake.pdf") is False


def test_receive_rejects_empty_file(service: UploadService, session: Session) -> None:
    with pytest.raises(UploadValidationError):
        service.receive(BytesIO(b""), original_filename="empty.pdf", content_type="application/pdf")
    assert UploadRepository(session).count() == 0


def test_receive_rejects_corrupted_pdf(
    service: UploadService, session: Session, storage: LocalStorageProvider
) -> None:
    with pytest.raises(UploadValidationError):
        service.receive(
            BytesIO(b"%PDF-1.4 truncated without trailer"),
            original_filename="broken.pdf",
            content_type="application/pdf",
        )
    assert UploadRepository(session).count() == 0
    assert len(list(storage._root.iterdir())) == 0


def test_receive_rejects_wrong_mime(service: UploadService, session: Session) -> None:
    with pytest.raises(UploadValidationError):
        service.receive(
            BytesIO(make_pdf_bytes()),
            original_filename="kitab.pdf",
            content_type="text/plain",
        )
    assert UploadRepository(session).count() == 0


def test_receive_rejects_oversized_file(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    small = Settings(upload_storage_path=str(tmp_path / "uploads"), upload_max_size_bytes=10)
    service = UploadService(session=session, storage=storage, settings=small)
    with pytest.raises(UploadTooLargeError):
        service.receive(
            BytesIO(make_pdf_bytes(extra=b"x" * 100)),
            original_filename="big.pdf",
            content_type="application/pdf",
        )
    assert UploadRepository(session).count() == 0


def test_duplicate_detection(
    service: UploadService, session: Session, fake_task: DispatchFakes
) -> None:
    data = make_pdf_bytes()
    first = service.receive(
        BytesIO(data), original_filename="a.pdf", content_type="application/pdf"
    )
    with pytest.raises(DuplicateUploadError) as exc_info:
        service.receive(BytesIO(data), original_filename="b.pdf", content_type="application/pdf")
    assert exc_info.value.details == {"existing_upload_id": first.id}
    # the failed duplicate cleaned itself up
    assert UploadRepository(session).count() == 1
    assert len(list(service._storage._root.iterdir())) == 1


def test_filename_sanitization(service: UploadService, fake_task: DispatchFakes) -> None:
    upload = service.receive(
        BytesIO(make_pdf_bytes()),
        original_filename="../../Radd al-Muhtar.pdf",
        content_type=None,
    )
    assert upload.original_filename == "Radd al-Muhtar.pdf"
    assert upload.filename == f"{upload.id}.pdf"


def test_delete_removes_file_row_and_job(
    service: UploadService, session: Session, fake_task: DispatchFakes
) -> None:
    upload = service.receive(
        BytesIO(make_pdf_bytes()), original_filename="a.pdf", content_type="application/pdf"
    )
    key = upload.storage_path
    job_id = IngestionJobRepository(session).find_pipeline_job(upload.id).id
    service.delete(upload)
    assert service._storage.exists(key) is False
    assert UploadRepository(session).get(upload.id) is None
    assert IngestionJobRepository(session).get(job_id) is None


def test_background_task_marks_job_queued(session: Session, monkeypatch) -> None:
    upload = UploadRepository(session).create(
        Upload(
            original_filename="a.pdf",
            filename="u.pdf",
            storage_path="u.pdf",
            mime="application/pdf",
            status="uploaded",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    assert job.status == "uploaded"

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    ingestion_module.mark_queued(job.id, upload.id)
    session.expire_all()
    assert IngestionJobRepository(session).get(job.id).status == "queued"
