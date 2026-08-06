"""Tests for the Phase 5 LayoutRunner (DB streaming) and its Celery task."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.ingestion as ingestion_module
from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.services.extraction import ExtractionRunner
from app.services.layout import (
    REGION_FOOTER,
    REGION_FOOTNOTE,
    REGION_HEADER,
    REGION_MAIN,
    REGION_MARGIN,
    LayoutRunner,
)
from app.worker.celery_app import celery_app
from tests.support import (
    build_double_column_pdf,
    build_footnote_pdf,
    build_header_footer_pdf,
    build_margin_pdf,
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


def _extract_upload(
    session: Session,
    storage: LocalStorageProvider,
    pdf_path: Path,
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
            status="completed",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    ExtractionRunner(session, storage).run(job, upload)
    session.commit()
    session.expire_all()
    return upload, IngestionJobRepository(session).get(job.id)


def _first_page_blocks(session: Session, job: IngestionJob) -> list:
    pages = ExtractionRepository(session).list_pages(job.id)
    return ExtractionRepository(session).list_blocks(pages[0])


def test_runner_persists_header_and_footer_regions(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_header_footer_pdf(tmp_path / "hf.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    result = LayoutRunner(session).run(job)
    session.expire_all()

    assert result.page_count == 1
    assert result.block_count == 6
    assert result.region_counts[REGION_HEADER] == 1
    assert result.region_counts[REGION_FOOTER] == 1
    assert result.region_counts[REGION_MAIN] == 4

    blocks = _first_page_blocks(session, job)
    header = next(block for block in blocks if block.region == REGION_HEADER)
    footer = next(block for block in blocks if block.region == REGION_FOOTER)
    assert header.reading_order == 0
    assert footer.reading_order == 5
    assert header.classification_reason == "top band (running header position)"
    assert footer.confidence == 0.95


def test_runner_persists_footnote_regions(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_footnote_pdf(tmp_path / "footnote.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    LayoutRunner(session).run(job)
    session.expire_all()

    blocks = _first_page_blocks(session, job)
    footnotes = [block for block in blocks if block.region == REGION_FOOTNOTE]
    mains = [block for block in blocks if block.region == REGION_MAIN]
    assert len(footnotes) == 2
    assert len(mains) == 4
    assert all(block.confidence >= 0.9 for block in footnotes)


def test_runner_persists_margin_regions(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_margin_pdf(tmp_path / "margin.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    LayoutRunner(session).run(job)
    session.expire_all()

    blocks = _first_page_blocks(session, job)
    margins = [block for block in blocks if block.region == REGION_MARGIN]
    assert len(margins) == 2
    assert all(block.classification_reason.startswith("narrow column") for block in margins)


def test_runner_handles_double_column(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_double_column_pdf(tmp_path / "double.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    LayoutRunner(session).run(job)
    session.expire_all()

    blocks = _first_page_blocks(session, job)
    mains = [block for block in blocks if block.region == REGION_MAIN]
    assert len(mains) == 8
    orders = [block.reading_order for block in blocks]
    assert sorted(orders) == list(range(8))
    left = [block for block in blocks if block.bbox[0] < 200]
    right = [block for block in blocks if block.bbox[0] >= 200]
    assert max(block.reading_order for block in left) < min(
        block.reading_order for block in right
    )


def test_runner_updates_job_progress_to_completion(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf", pages=2)
    upload, job = _extract_upload(session, storage, pdf)

    LayoutRunner(session).run(job)
    session.expire_all()

    done = IngestionJobRepository(session).get(job.id)
    assert done.progress_percent == 100
    assert done.current_step == "layout"
    assert ExtractionRepository(session).region_summary(job.id).get("unknown", 0) == 0


def test_layout_task_completes_job_and_upload(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_header_footer_pdf(tmp_path / "hf.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_layout_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    done_upload = UploadRepository(session).get(upload.id)
    assert done_job.status == "completed"
    assert done_job.progress_percent == 100
    assert done_upload.status == "completed"
    assert done_job.errors == []


def test_layout_task_records_error_and_fails_job(
    session: Session, storage: LocalStorageProvider, tmp_path, monkeypatch
) -> None:
    pdf = build_text_pdf(tmp_path / "text.pdf")
    upload, job = _extract_upload(session, storage, pdf)

    def boom(_job: IngestionJob):
        raise RuntimeError("db exploded")

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(ingestion_module, "get_session_factory", lambda: factory)
    monkeypatch.setattr(ingestion_module, "LayoutRunner", lambda _session: type(
        "Broken", (), {"run": boom}
    )())

    celery_app.conf.task_always_eager = True
    try:
        ingestion_module.run_layout_task.delay(job.id, upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    done_job = IngestionJobRepository(session).get(job.id)
    assert done_job.status == "failed"
    assert done_job.error_message is not None
    assert any(error.step == "layout" for error in done_job.errors)
