"""Phase 15 §713 page-level parallel extraction tests.

Overlap is proven with threading events — no arbitrary sleeps: a blocking parse
function signals entry and waits for a release, so both pages being in the
parse step at once proves the extraction ran concurrently across workers.
"""

import threading
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.storage import LocalStorageProvider
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.services.extraction import ExtractionRunner, _parse_page, _partition_pages
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


def test_partition_pages_splits_contiguous_slices() -> None:
    assert _partition_pages(4, 4) == [[0], [1], [2], [3]]
    assert _partition_pages(5, 4) == [[0, 1], [2, 3], [4]]
    assert _partition_pages(2, 4) == [[0], [1]]
    assert _partition_pages(6, 2) == [[0, 1, 2], [3, 4, 5]]
    assert _partition_pages(0, 4) == []


def test_page_parsing_overlaps_across_workers(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    """Both pages are in the parse step at once → real parallel extraction."""
    pdf = build_text_pdf(tmp_path / "book.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    started = [threading.Event() for _ in range(2)]
    finished = [threading.Event() for _ in range(2)]
    release = threading.Event()

    def parse_fn(page, page_number):
        started[page_number - 1].set()
        release.wait(timeout=10)
        info = _parse_page(page, page_number)
        finished[page_number - 1].set()
        return info

    outcome: dict = {}

    def _run() -> None:
        try:
            outcome["result"] = ExtractionRunner(
                session, storage, max_workers=4, parse_fn=parse_fn
            ).run(job, upload)
        except Exception as exc:  # pragma: no cover - surfaced via outcome
            outcome["error"] = exc

    thread = threading.Thread(target=_run)
    thread.start()

    assert started[0].wait(timeout=10), "page 1 never reached the parse step"
    assert started[1].wait(timeout=10), "page 2 never reached the parse step"
    # Both entered the parse step while neither returned -> they overlapped.
    assert not finished[0].is_set()
    assert not finished[1].is_set()

    release.set()
    thread.join(timeout=15)
    assert not thread.is_alive()
    assert "error" not in outcome

    result = outcome["result"]
    assert [page.number for page in result.pages] == [1, 2]
    pages = ExtractionRepository(session).list_pages(job.id)
    assert [page.page_number for page in pages] == [1, 2]
    assert job.progress_percent == 100


def test_parallel_extraction_preserves_deterministic_page_order(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "book.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    result = ExtractionRunner(session, storage, max_workers=4).run(job, upload)

    assert result.page_count == 2
    assert [page.number for page in result.pages] == [1, 2]
    pages = ExtractionRepository(session).list_pages(job.id)
    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.has_text for page in pages)


def test_parallel_extraction_records_failed_page_keeps_others(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    """A single broken page is recorded (not swallowed) without losing the rest."""
    pdf = build_text_pdf(tmp_path / "book.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    def parse_fn(page, page_number):
        if page_number == 2:
            raise RuntimeError("page exploded")
        return _parse_page(page, page_number)

    result = ExtractionRunner(session, storage, max_workers=4, parse_fn=parse_fn).run(job, upload)

    assert [page.number for page in result.pages] == [1, 2]
    pages = ExtractionRepository(session).list_pages(job.id)
    assert len(pages) == 2
    assert pages[0].has_text is True
    assert pages[0].error_message is None
    assert pages[1].has_text is False
    assert "page exploded" in (pages[1].error_message or "")


def test_parallel_extraction_single_worker_still_completes(
    session: Session, storage: LocalStorageProvider, tmp_path
) -> None:
    pdf = build_text_pdf(tmp_path / "book.pdf")
    upload, job = _make_upload_and_job(session, storage, pdf)

    result = ExtractionRunner(session, storage, max_workers=1).run(job, upload)

    assert result.page_count == 2
    pages = ExtractionRepository(session).list_pages(job.id)
    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.has_text for page in pages)
