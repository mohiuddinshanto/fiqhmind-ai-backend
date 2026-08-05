from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Book, Chunk, Edition, Page, Role, User, Volume
from app.db.models import Session as AuthSession
from app.db.repositories import (
    BookRepository,
    ChunkRepository,
    IngestionJobRepository,
    SessionRepository,
    UserRepository,
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
def corpus(session: Session) -> dict[str, str]:
    book = Book(title="Al-Hidayah", author="al-Marghinani", status="published")
    session.add(book)
    edition = Edition(book=book, publisher="Dar Ihya", edition_number=1)
    session.add(edition)
    volume = Volume(edition=edition, volume_number=1)
    session.add(volume)
    page = Page(volume=volume, printed_page=5, pdf_page=6)
    session.add(page)
    session.commit()
    return {"book": book.id, "edition": edition.id, "volume": volume.id, "page": page.id}


def test_user_repository_crud(session: Session) -> None:
    repo = UserRepository(session)
    role = Role(name="user")
    session.add(role)
    session.commit()

    user = repo.create(
        User(email="reader@fiqhmind.test", password_hash="hash", role_id=role.id)
    )
    assert repo.get(user.id) == user
    assert repo.get_by_email("reader@fiqhmind.test") is user
    assert repo.get_by_email("missing@fiqhmind.test") is None

    user.full_name = "Reader"
    repo.update(user)
    assert repo.get(user.id).full_name == "Reader"

    repo.delete(user)
    assert repo.get(user.id) is None


def test_session_repository(session: Session) -> None:
    user_repo = UserRepository(session)
    user = user_repo.create(User(email="s@fiqhmind.test", password_hash="hash"))
    repo = SessionRepository(session)
    auth_session = repo.create(
        AuthSession(user_id=user.id, refresh_token_hash="tok", expires_at=datetime.utcnow())
    )
    assert repo.get_by_refresh_token_hash("tok") is auth_session
    repo.revoke(auth_session)
    assert auth_session.revoked_at is not None
    assert repo.list_active_for_user(user.id) == []


def test_book_repository(session: Session, corpus: dict[str, str]) -> None:
    repo = BookRepository(session)
    book = repo.get(corpus["book"])
    assert book is not None
    assert repo.get_by_title_author("Al-Hidayah", "al-Marghinani") is book
    assert [e.id for e in repo.get_editions(book)] == [corpus["edition"]]
    assert [v.id for v in repo.get_volumes(book)] == [corpus["volume"]]
    assert [b.id for b in repo.list_public()] == [corpus["book"]]


def test_chunk_repository(session: Session, corpus: dict[str, str]) -> None:
    repo = ChunkRepository(session)
    chunk = repo.create(
        Chunk(
            chunk_id="c" * 64,
            book_id=corpus["book"],
            edition_id=corpus["edition"],
            volume_id=corpus["volume"],
            page_start_id=corpus["page"],
            page_end_id=corpus["page"],
            raw_text="نص تجريبي",
            printed_page_start=5,
            printed_page_end=5,
        )
    )
    assert repo.get_by_hash("c" * 64) is chunk
    assert [c.chunk_id for c in repo.list_by_book(corpus["book"])] == [chunk.chunk_id]
    assert [c.chunk_id for c in repo.list_by_volume(corpus["volume"])] == [chunk.chunk_id]
    assert [c.chunk_id for c in repo.list_by_page(corpus["volume"], 5)] == [chunk.chunk_id]


def test_ingestion_job_repository_lifecycle(session: Session, corpus: dict[str, str]) -> None:
    repo = IngestionJobRepository(session)
    job = repo.create_for_book(corpus["book"], kind="initial")
    assert job.status == "uploaded"

    repo.update_status(job, "extracting", progress_percent=10, current_step="extract")
    assert job.status == "extracting"
    assert job.progress_percent == 10

    repo.update_status(job, "indexed", progress_percent=100)
    assert job.status == "indexed"
    assert job.finished_at is not None
    assert repo.list_active() == []

    with pytest.raises(ValueError):
        repo.update_status(job, "bogus")
