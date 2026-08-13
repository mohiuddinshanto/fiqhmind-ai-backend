"""Phase 15 checkpointing/resumability tests for `ExtractionRunner`.

A crash mid-book must not re-extract pages that were already checkpoint-committed
to Postgres: `run()` skips persisted page numbers (resume), and page rows are
committed every `checkpoint_pages` pages rather than once per page.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.services.extraction import ExtractionRunner, _parse_page
from tests.support import build_text_pdf


@pytest.fixture()
def session() -> Generator[Session, None, None]:
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
            status="uploaded",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    return upload, job


def test_runner_resumes_from_partial_pages(session: Session, storage, tmp_path) -> None:
    """Already-committed pages are skipped; only missing pages are extracted."""
    pdf = build_text_pdf(tmp_path / "book.pdf", pages=3)
    upload, job = _make_upload_and_job(session, storage, pdf)

    # Simulate a committed checkpoint: pages 1 and 2 already in Postgres.
    repo = ExtractionRepository(session)
    for number in (1, 2):
        repo.create_page(
            job_id=job.id,
            page_number=number,
            width=595,
            height=842,
            rotation=0,
            has_text=True,
            char_count=40,
            block_count=1,
            image_count=0,
            drawing_count=0,
            confidence=0.5,
        )
    session.commit()

    result = ExtractionRunner(session, storage, max_workers=4).run(job, upload)

    pages = repo.list_pages(job.id)
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert repo.count_pages(job.id) == 3
    assert job.progress_percent == 100
    # Returned summary covers the whole book, not just the resumed tail.
    assert [page.number for page in result.pages] == [1, 2, 3]
    assert pages[2].char_count > 0  # page 3 actually parsed, not blank


def test_runner_does_not_reparse_persisted_pages(session: Session, storage, tmp_path) -> None:
    """The parse callback must only see pages that are not yet persisted."""
    pdf = build_text_pdf(tmp_path / "book.pdf", pages=3)
    upload, job = _make_upload_and_job(session, storage, pdf)

    repo = ExtractionRepository(session)
    repo.create_page(
        job_id=job.id,
        page_number=2,
        width=595,
        height=842,
        rotation=0,
        has_text=True,
        char_count=40,
        block_count=1,
        image_count=0,
        drawing_count=0,
        confidence=0.5,
    )
    session.commit()

    parsed: list[int] = []

    def parse_fn(page, page_number):
        parsed.append(page_number)
        return _parse_page(page, page_number)

    ExtractionRunner(session, storage, max_workers=4, parse_fn=parse_fn).run(job, upload)

    assert sorted(parsed) == [1, 3]
    assert [p.page_number for p in repo.list_pages(job.id)] == [1, 2, 3]


def test_runner_commits_checkpoints_not_every_page(
    storage, tmp_path
) -> None:
    """With `checkpoint_pages`, commits happen in batches, not once per page."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    session = testing_session()

    pdf = build_text_pdf(tmp_path / "book.pdf", pages=6)
    upload, job = _make_upload_and_job(session, storage, pdf)

    commits: list[int] = []

    @event.listens_for(session, "after_commit")
    def _count_commit(_) -> None:
        commits.append(1)

    ExtractionRunner(session, storage, max_workers=4, checkpoint_pages=2).run(job, upload)

    # 6 pages, checkpoint every 2 → 3 batch commits (plus begin/progress/final
    # = 6 total), strictly fewer than the per-page baseline (1 + 1 + 6 + 1 = 9).
    assert ExtractionRepository(session).count_pages(job.id) == 6
    assert len(commits) < 9
