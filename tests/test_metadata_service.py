"""Tests for the Phase 6 MetadataRunner (DB streaming) and its Celery task."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.ingestion as ingestion_module
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import (
    IngestionJobRepository,
    MetadataRepository,
    UploadRepository,
)
from app.services.extraction import ExtractionRunner
from app.services.metadata import (
    FIELD_AUTHOR,
    FIELD_PUBLICATION_YEAR,
    FIELD_PUBLISHER,
    FIELD_TITLE,
    FIELD_VOLUME,
    NUMBER_SYSTEM_ARABIC,
    NUMBER_SYSTEM_LATIN,
    MetadataRunner,
)
from app.worker.celery_app import celery_app
from tests.support import (
    arabic_font_file,
    build_arabic_numbered_pdf,
    build_structured_book,
)


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


def _extract_upload(
    session: Session,
    storage: LocalStorageProvider,
    pdf_path: Path,
    *,
    filename: str = "Al-Mabsut v1.pdf",
) -> tuple[Upload, IngestionJob]:
    key = "book.pdf"
    with storage.writer(key) as writer:
        writer.write(pdf_path.read_bytes())
    upload = UploadRepository(session).create(
        Upload(
            original_filename=filename,
            filename=key,
            storage_path=key,
            mime="application/pdf",
            status="completed",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    ExtractionRunner(session, storage).run(job, upload)
    session.commit()
    session.expire_all()
    return upload, IngestionJobRepository(session).get(job.id)


def test_runner_persists_document_metadata(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    result = MetadataRunner(session).run(job, upload)
    session.expire_all()

    assert result.page_count == 4
    assert result.pages_mapped == 4
    assert result.numbering_system == NUMBER_SYSTEM_LATIN
    assert result.confidence > 0.0

    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    fields = {item.field: item.value for item in document.fields}
    assert fields[FIELD_TITLE] == "Al-Mabsut"
    assert fields[FIELD_VOLUME] == "1"
    assert fields[FIELD_AUTHOR] == "Imam Sarakhsi"
    assert fields[FIELD_PUBLISHER] == "Dar al-Kutub"
    assert fields[FIELD_PUBLICATION_YEAR] == "1998"
    assert document.numbering_system == NUMBER_SYSTEM_LATIN


def test_runner_persists_page_mapping(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    MetadataRunner(session).run(job, upload)
    session.expire_all()

    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    pages = MetadataRepository(session).list_pages(document)
    assert len(pages) == 4
    by_pdf = {page.pdf_page: page for page in pages}

    assert by_pdf[1].printed_page == ""  # cover has no footer
    assert by_pdf[1].page_number_uncertain is True
    assert by_pdf[2].printed_page == "1"
    assert by_pdf[2].printed_page_numeric == 1
    assert by_pdf[3].printed_page == "2"
    assert by_pdf[4].printed_page == "3"
    assert all(by_pdf[n].pdf_page == n for n in (2, 3, 4))  # pdf page never lost


def test_runner_persists_structures(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    MetadataRunner(session).run(job, upload)
    session.expire_all()

    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    structures = MetadataRepository(session).list_structures(document)
    kitabs = [structure for structure in structures if structure.level == "kitab"]
    babs = [structure for structure in structures if structure.level == "bab"]
    assert len(kitabs) == 1
    assert kitabs[0].name == "Kitab al-Taharah"
    assert kitabs[0].page_start == 2
    assert kitabs[0].page_end == 4
    assert len(babs) == 1
    assert babs[0].name == "Bab al-Wudu"
    assert babs[0].page_start == 3
    assert babs[0].page_end == 4


def test_runner_attaches_sections_to_pages(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    MetadataRunner(session).run(job, upload)
    session.expire_all()

    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    pages = MetadataRepository(session).list_pages(document)
    by_pdf = {page.pdf_page: page for page in pages}
    assert by_pdf[2].kitab == "Kitab al-Taharah"
    assert by_pdf[2].bab is None
    assert by_pdf[3].kitab == "Kitab al-Taharah"
    assert by_pdf[3].bab == "Bab al-Wudu"
    assert by_pdf[4].kitab == "Kitab al-Taharah"
    assert by_pdf[4].bab == "Bab al-Wudu"


def test_runner_updates_job_progress_to_completion(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    MetadataRunner(session).run(job, upload)
    session.expire_all()

    done = IngestionJobRepository(session).get(job.id)
    assert done.progress_percent == 100
    assert done.current_step == "metadata"
    assert done.started_at is not None


def test_runner_replaces_previous_metadata(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    MetadataRunner(session).run(job, upload)
    MetadataRunner(session).run(job, upload)
    session.expire_all()

    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    assert len(document.fields) == len({item.field for item in document.fields})
    assert len(MetadataRepository(session).list_pages(document)) == 4
    assert len(MetadataRepository(session).list_structures(document)) == 2


def test_runner_handles_arabic_numbered_pdf(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    if arabic_font_file() is None:
        pytest.skip("no Arabic-capable font installed on this machine")

    pdf = build_arabic_numbered_pdf(tmp_path / "arabic.pdf")
    upload, job = _extract_upload(session, storage, pdf, filename="Al-Mabsut.pdf")

    result = MetadataRunner(session).run(job, upload)
    session.expire_all()

    assert result.numbering_system == NUMBER_SYSTEM_ARABIC
    document = MetadataRepository(session).get_by_job(job.id)
    assert document is not None
    pages = MetadataRepository(session).list_pages(document)
    assert len(pages) == 3
    by_pdf = {page.pdf_page: page for page in pages}
    assert by_pdf[1].printed_page == "٥"
    assert by_pdf[1].printed_page_numeric == 5
    assert by_pdf[1].page_number_uncertain is False
    assert by_pdf[3].printed_page_numeric == 7


def test_metadata_task_completes_job_and_upload(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_metadata_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "completed"
    assert done_job.progress_percent == 100
    assert done_job.current_step == "metadata"
    assert done_upload.status == "completed"
    assert done_job.errors == []
    assert MetadataRepository(session).get_by_job(job.id) is not None


def test_metadata_task_records_error_and_fails_job(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_structured_book(tmp_path / "book.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    def boom(_job: IngestionJob, _upload: Upload):
        raise RuntimeError("db exploded")

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(
        ingestion_module, "MetadataRunner", lambda _session: type("Broken", (), {"run": boom})()
    )

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_metadata_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert any(error.step == "metadata" for error in done_job.errors)
