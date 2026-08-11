"""Tests for the Phase 4 PDF extraction engine and its DB streaming runner."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.ingestion as ingestion_module
from app.core.exceptions import (
    EncryptedPdfError,
    MalformedPdfError,
    PdfPageLimitError,
)
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.services.extraction import (
    MIN_TEXT_CHARS_PER_PAGE,
    TEXT_CAPACITY_PER_PAGE,
    ExtractionRunner,
    extract_pdf,
)
from app.worker.celery_app import celery_app
from tests.support import (
    build_encrypted_pdf,
    build_image_pdf,
    build_malformed_pdf,
    build_rotated_pdf,
    build_scanned_pdf,
    build_text_pdf,
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


def _make_upload_and_job(
    session: Session,
    storage: LocalStorageProvider,
    pdf_path: Path,
    *,
    status: str = "uploaded",
) -> tuple[Upload, IngestionJob]:
    key = "kitab.pdf"
    with storage.writer(key) as writer:
        writer.write(pdf_path.read_bytes())
    upload = UploadRepository(session).create(
        Upload(
            original_filename="kitab.pdf",
            filename=key,
            storage_path=key,
            mime="application/pdf",
            status=status,
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    return upload, job


def test_extract_pdf_born_digital_text_layer(session: Session, tmp_path) -> None:
    result = extract_pdf(str(build_text_pdf(tmp_path / "text.pdf")))

    assert result.page_count == 2
    assert result.total_chars > 0
    assert result.has_text_layer is True
    assert 0.0 < result.confidence <= 1.0

    page = result.pages[0]
    assert page.number == 1
    assert page.width == 595
    assert page.height == 842
    assert page.rotation == 0
    assert page.has_text is True
    assert page.char_count >= MIN_TEXT_CHARS_PER_PAGE
    assert page.confidence == min(1.0, page.char_count / TEXT_CAPACITY_PER_PAGE)

    assert len(page.blocks) == 4
    first = page.blocks[0]
    assert first.index == 0
    assert first.text == "Bismillah ar-Rahman ar-Rahim"
    assert first.font == "Helvetica"
    assert first.size == 16
    assert len(first.bbox) == 4
    assert round(first.bbox[0]) == 72
    assert first.bbox[1] < 100 < first.bbox[3]  # baseline 100 sits inside glyph bbox
    assert len(first.spans) == 1
    assert first.spans[0].font == "Helvetica"
    assert first.spans[0].size == 16
    assert first.spans[0].text == "Bismillah ar-Rahman ar-Rahim"


def test_extract_pdf_scanned_page_has_no_text_layer(session: Session, tmp_path) -> None:
    result = extract_pdf(str(build_scanned_pdf(tmp_path / "scanned.pdf")))

    assert result.page_count == 1
    assert result.has_text_layer is False
    assert result.confidence == 0.0

    page = result.pages[0]
    assert page.has_text is False
    assert page.char_count == 0
    assert len(page.blocks) == 0
    assert len(page.images) == 1
    assert [round(value) for value in page.images[0].bbox] == [0, 0, 595, 842]


def test_extract_pdf_captures_images_and_drawings(session: Session, tmp_path) -> None:
    result = extract_pdf(str(build_image_pdf(tmp_path / "image.pdf")))
    page = result.pages[0]

    assert len(page.blocks) == 4
    assert len(page.images) == 1
    image = page.images[0]
    assert [round(value) for value in image.bbox] == [72, 300, 172, 400]
    assert image.width == 1
    assert image.height == 1

    assert len(page.drawings) == 1
    drawing = page.drawings[0]
    assert [round(value) for value in drawing.bbox] == [300, 300, 500, 350]
    assert drawing.kind is not None
    assert drawing.stroke_width == 2


def test_extract_pdf_preserves_rotation_swapped_dims(session: Session, tmp_path) -> None:
    result = extract_pdf(str(build_rotated_pdf(tmp_path / "rotated.pdf")))
    page = result.pages[0]

    assert page.rotation == 90
    assert page.width == 842
    assert page.height == 595
    assert page.has_text is True


def test_extract_pdf_raises_on_malformed_file(session: Session, tmp_path) -> None:
    path = build_malformed_pdf(tmp_path / "malformed.pdf")
    with pytest.raises(MalformedPdfError):
        extract_pdf(str(path))


def test_extract_pdf_raises_on_encrypted_file(session: Session, tmp_path) -> None:
    path = build_encrypted_pdf(tmp_path / "encrypted.pdf")
    with pytest.raises(EncryptedPdfError):
        extract_pdf(str(path))


def test_extract_pdf_rejects_page_count_above_cap(session: Session, tmp_path) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    with pytest.raises(PdfPageLimitError):
        extract_pdf(str(pdf), max_pages=1)


def test_runner_rejects_page_count_above_cap(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    with pytest.raises(PdfPageLimitError):
        ExtractionRunner(session, storage, max_pages=1).run(job, upload)


def test_runner_streams_pages_and_blocks_into_db(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    result = ExtractionRunner(session, storage).run(job, upload)

    assert upload.page_count == 2
    assert job.status == "extracting"
    assert job.current_step == "extracting"
    assert job.progress_percent == 100
    assert job.started_at is not None
    assert result.has_text_layer is True

    repo = ExtractionRepository(session)
    assert repo.count_pages(job.id) == 2
    pages = repo.list_pages(job.id)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].has_text is True
    assert pages[0].char_count >= MIN_TEXT_CHARS_PER_PAGE
    assert pages[0].block_count == 4
    assert pages[0].image_count == 0
    assert pages[0].drawing_count == 0
    assert pages[0].confidence > 0.0
    assert pages[1].page_number == 2


def test_runner_persists_blocks_spans_images_drawings(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_image_pdf(tmp_path / "image.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    ExtractionRunner(session, storage).run(job, upload)
    page = ExtractionRepository(session).list_pages(job.id)[0]

    assert page.block_count == 4
    blocks = sorted(page.blocks, key=lambda block: block.block_index)
    assert blocks[0].text == "Bismillah ar-Rahman ar-Rahim"
    assert blocks[0].font == "Helvetica"
    assert blocks[0].font_size == 16
    assert blocks[0].span_count == 1
    assert round(blocks[0].bbox[0]) == 72
    assert blocks[0].bbox[1] < 100 < blocks[0].bbox[3]
    assert len(blocks[0].spans) == 1
    assert blocks[0].spans[0].font_size == 16

    assert len(page.images) == 1
    image = page.images[0]
    assert [round(value) for value in image.bbox] == [72, 300, 172, 400]
    assert image.width == 1
    assert image.height == 1

    assert len(page.drawings) == 1
    drawing = page.drawings[0]
    assert [round(value) for value in drawing.bbox] == [300, 300, 500, 350]
    assert drawing.stroke_width == 2


def test_runner_scanned_pdf_persists_no_text(session: Session, storage, tmp_path) -> None:
    pdf = build_scanned_pdf(tmp_path / "scanned.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    result = ExtractionRunner(session, storage).run(job, upload)
    page = ExtractionRepository(session).list_pages(job.id)[0]

    assert result.has_text_layer is False
    assert page.has_text is False
    assert page.char_count == 0
    assert page.confidence == 0.0
    assert page.image_count == 1


def test_runner_raises_when_stored_file_missing(session: Session, storage) -> None:
    upload = UploadRepository(session).create(
        Upload(
            original_filename="kitab.pdf",
            filename="kitab.pdf",
            storage_path="kitab.pdf",
            mime="application/pdf",
            status="uploaded",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)

    with pytest.raises(MalformedPdfError):
        ExtractionRunner(session, storage).run(job, upload)


def test_runner_raises_on_malformed_file(session: Session, storage, tmp_path) -> None:
    pdf = build_malformed_pdf(tmp_path / "malformed.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    with pytest.raises(MalformedPdfError):
        ExtractionRunner(session, storage).run(job, upload)


def test_runner_raises_on_encrypted_file(session: Session, storage, tmp_path) -> None:
    pdf = build_encrypted_pdf(tmp_path / "encrypted.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    with pytest.raises(EncryptedPdfError):
        ExtractionRunner(session, storage).run(job, upload)


def test_extraction_task_completes_job_and_upload(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(ingestion_module, "get_storage_provider", lambda: storage)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.extract_pdf_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "completed"
    assert done_job.progress_percent == 100
    assert done_upload.status == "completed"
    assert done_upload.page_count == 2
    assert ExtractionRepository(session).count_pages(job.id) == 2


def test_extraction_task_fails_job_on_malformed_pdf(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_malformed_pdf(tmp_path / "malformed.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(ingestion_module, "get_storage_provider", lambda: storage)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.extract_pdf_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert done_upload.status == "failed"
    assert len(done_job.errors) >= 1
