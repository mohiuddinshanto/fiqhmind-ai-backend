"""Tests for the Phase 4 extraction repository (page/block/span/image/drawing rows)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Upload
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository


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
def job_id(session: Session) -> str:
    upload = UploadRepository(session).create(
        Upload(
            original_filename="a.pdf",
            filename="a.pdf",
            storage_path="a.pdf",
            mime="application/pdf",
            status="uploaded",
        )
    )
    job = IngestionJobRepository(session).create_for_upload(upload.id)
    return job.id


def test_create_and_read_pages_with_children(session: Session, job_id: str) -> None:
    repo = ExtractionRepository(session)

    page = repo.create_page(
        job_id=job_id,
        page_number=1,
        width=595.0,
        height=842.0,
        rotation=0,
        has_text=True,
        char_count=120,
        block_count=1,
        image_count=1,
        drawing_count=1,
        confidence=0.03,
    )
    block = repo.add_block(
        page,
        block_index=0,
        bbox=[72.0, 100.0, 290.0, 118.0],
        text="Bismillah ar-Rahman ar-Rahim",
        font="Helvetica",
        font_size=16.0,
    )
    block.span_count = 1
    span = repo.add_span(
        block,
        span_index=0,
        text="Bismillah ar-Rahman ar-Rahim",
        font="Helvetica",
        font_size=16.0,
        bbox=[72.0, 100.0, 290.0, 118.0],
        flags=0,
    )
    repo.add_image(
        page,
        image_index=0,
        bbox=[72.0, 300.0, 172.0, 400.0],
        width=1,
        height=1,
        xref=5,
    )
    repo.add_drawing(
        page,
        drawing_index=0,
        bbox=[300.0, 300.0, 500.0, 350.0],
        kind="fs",
        stroke_width=2.0,
    )
    session.commit()

    loaded = repo.get_page(job_id, 1)
    assert loaded is not None
    assert loaded.has_text is True
    assert loaded.char_count == 120
    assert loaded.width == 595.0
    assert loaded.block_count == 1

    loaded_block = loaded.blocks[0]
    assert loaded_block.span_count == 1
    assert loaded_block.spans[0].text == "Bismillah ar-Rahman ar-Rahim"
    assert span.text == loaded_block.spans[0].text

    assert len(loaded.images) == 1
    assert loaded.images[0].width == 1
    assert len(loaded.drawings) == 1
    assert loaded.drawings[0].stroke_width == 2.0


def test_job_summary_aggregates(session: Session, job_id: str) -> None:
    repo = ExtractionRepository(session)
    for number, chars in ((1, 100), (2, 300), (3, 0)):
        repo.create_page(
            job_id=job_id,
            page_number=number,
            width=595.0,
            height=842.0,
            rotation=0,
            has_text=chars >= 20,
            char_count=chars,
            block_count=1 if chars else 0,
            image_count=0,
            drawing_count=0,
            confidence=min(1.0, chars / 4000) if chars >= 20 else 0.0,
        )
    session.commit()

    summary = repo.job_summary(job_id)
    assert summary["page_count"] == 3
    assert summary["char_count"] == 400
    assert summary["block_count"] == 2
    assert summary["text_pages"] == 2
    assert summary["image_count"] == 0
    assert summary["drawing_count"] == 0
    assert 0.0 < summary["confidence"] < 1.0


def test_job_summary_empty_job(session: Session, job_id: str) -> None:
    summary = ExtractionRepository(session).job_summary(job_id)
    assert summary == {
        "page_count": 0,
        "char_count": 0,
        "block_count": 0,
        "image_count": 0,
        "drawing_count": 0,
        "text_pages": 0,
        "confidence": 0.0,
    }


def test_delete_for_job_removes_all_rows(session: Session, job_id: str) -> None:
    repo = ExtractionRepository(session)
    page = repo.create_page(
        job_id=job_id,
        page_number=1,
        width=595.0,
        height=842.0,
        rotation=0,
        has_text=True,
        char_count=100,
        block_count=1,
        image_count=0,
        drawing_count=0,
        confidence=0.1,
    )
    block = repo.add_block(
        page, block_index=0, bbox=[1, 2, 3, 4], text="x", font="Helvetica", font_size=12.0
    )
    block.span_count = 1
    repo.add_span(
        block, span_index=0, text="x", font="Helvetica", font_size=12.0, bbox=[1, 2, 3, 4]
    )
    session.commit()

    repo.delete_for_job(job_id)

    assert repo.count_pages(job_id) == 0
    assert repo.list_pages(job_id) == []


def test_job_delete_cascades_to_extracted_pages(session: Session, job_id: str) -> None:
    repo = ExtractionRepository(session)
    repo.create_page(
        job_id=job_id,
        page_number=1,
        width=595.0,
        height=842.0,
        rotation=0,
        has_text=False,
        char_count=0,
        block_count=0,
        image_count=0,
        drawing_count=0,
        confidence=0.0,
    )
    session.commit()

    job = IngestionJobRepository(session).get(job_id)
    assert job is not None
    IngestionJobRepository(session).delete(job)

    assert repo.count_pages(job_id) == 0
