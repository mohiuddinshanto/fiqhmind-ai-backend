from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Upload, UploadLog
from app.db.repositories import IngestionJobRepository, UploadRepository
from tests.support import make_pdf_bytes, sha256_hex


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


def make_upload(**overrides: object) -> Upload:
    fields = {
        "original_filename": "kitab.pdf",
        "filename": "u1.pdf",
        "sha256": sha256_hex(make_pdf_bytes()),
        "size": 42,
        "mime": "application/pdf",
        "storage_path": "u1.pdf",
        "status": "queued",
        "received_bytes": 42,
    }
    fields.update(overrides)
    return Upload(**fields)


def test_upload_repository_crud(session: Session) -> None:
    repo = UploadRepository(session)
    upload = repo.create(make_upload())
    assert repo.get(upload.id) is upload
    assert repo.get_by_sha256(upload.sha256) is upload
    assert repo.get_by_sha256("nope") is None
    assert repo.count() == 1

    upload.status = "processing"
    repo.update(upload)
    assert repo.get(upload.id).status == "processing"

    repo.delete(upload)
    assert repo.get(upload.id) is None
    assert repo.count() == 0


def test_upload_repository_list_newest_first(session: Session) -> None:
    repo = UploadRepository(session)
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 2, 1, tzinfo=UTC)
    first = repo.create(make_upload(original_filename="a.pdf", sha256="a" * 64, created_at=older))
    second = repo.create(make_upload(original_filename="b.pdf", sha256="b" * 64, created_at=newer))
    items = repo.list()
    assert [u.id for u in items] == [second.id, first.id]
    assert repo.count() == 2


def test_record_progress_commits(session: Session) -> None:
    repo = UploadRepository(session)
    upload = repo.create(make_upload())
    repo.record_progress(upload, 1024)
    fresh = repo.get(upload.id)
    assert fresh.received_bytes == 1024


def test_add_log_and_cascade_delete(session: Session) -> None:
    repo = UploadRepository(session)
    upload = repo.create(make_upload())
    log = repo.add_log(upload, "queued", message="accepted")
    session.commit()
    assert isinstance(log, UploadLog)
    assert [item.event for item in repo.get(upload.id).logs] == ["queued"]

    repo.delete_with_artifacts(upload)
    assert repo.get(upload.id) is None
    assert session.get(UploadLog, log.id) is None


def test_delete_with_artifacts_removes_job(session: Session) -> None:
    upload = UploadRepository(session).create(make_upload())
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    session.refresh(upload)
    UploadRepository(session).delete_with_artifacts(upload)
    assert IngestionJobRepository(session).get(job.id) is None
