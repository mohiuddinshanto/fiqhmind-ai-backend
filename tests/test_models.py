import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import (
    Book,
    Chunk,
    Edition,
    IngestionError,
    IngestionJob,
    Page,
    Role,
    User,
    Volume,
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


def make_corpus(session: Session) -> tuple[Book, Edition, Volume, Page]:
    book = Book(title="Radd al-Muhtar", author="Ibn Abidin", status="published")
    session.add(book)
    edition = Edition(book=book, publisher="Dar al-Fikr", edition_number=1)
    session.add(edition)
    volume = Volume(edition=edition, volume_number=1, title="Taharah")
    session.add(volume)
    page = Page(volume=volume, printed_page=12, pdf_page=14)
    session.add(page)
    session.commit()
    return book, edition, volume, page


def test_user_role_relationship(session: Session) -> None:
    role = Role(name="admin", description="Administrator")
    user = User(email="admin@fiqhmind.test", password_hash="hash", full_name="Admin", role=role)
    session.add(user)
    session.commit()
    session.refresh(user)

    assert user.role is role
    assert role.users == [user]


def test_user_email_unique(session: Session) -> None:
    session.add(User(email="same@fiqhmind.test", password_hash="h1"))
    session.commit()
    session.add(User(email="same@fiqhmind.test", password_hash="h2"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_book_value_equality() -> None:
    first = Book(title="Al-Hidayah", author="al-Marghinani")
    second = Book(title="Al-Hidayah", author="al-Marghinani")
    other = Book(title="Al-Hidayah", author="Someone Else")
    assert first == second
    assert first != other
    assert len({first, second, other}) == 2


def test_book_status_check_constraint(session: Session) -> None:
    session.add(Book(title="Bad", author="Author", status="bogus"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_edition_volume_unique(session: Session) -> None:
    book, edition, _, _ = make_corpus(session)
    session.add(Volume(edition_id=edition.id, volume_number=1))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    assert book.id


def test_corpus_chain(session: Session) -> None:
    book, edition, volume, page = make_corpus(session)
    assert volume.edition_id == edition.id
    assert page.volume_id == volume.id
    assert edition.book_id == book.id


def test_chunk_region_check_constraint(session: Session) -> None:
    book, edition, volume, _ = make_corpus(session)
    session.add(
        Chunk(
            chunk_id="a" * 64,
            book_id=book.id,
            edition_id=edition.id,
            volume_id=volume.id,
            raw_text="نص",
            region="bogus",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_chunk_page_anchors(session: Session) -> None:
    book, edition, volume, page = make_corpus(session)
    chunk = Chunk(
        chunk_id="b" * 64,
        book_id=book.id,
        edition_id=edition.id,
        volume_id=volume.id,
        page_start_id=page.id,
        page_end_id=page.id,
        raw_text="نص اختبار",
        region="main",
        verified=True,
    )
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    assert chunk.page_start.printed_page == 12
    assert chunk.book.title == "Radd al-Muhtar"


def test_ingestion_job_and_error(session: Session) -> None:
    book, _, _, _ = make_corpus(session)
    job = IngestionJob(book_id=book.id, kind="initial", status="uploaded")
    session.add(job)
    session.commit()
    error = IngestionError(job_id=job.id, step="extracting", message="bad pdf")
    session.add(error)
    session.commit()
    session.refresh(job)
    assert job.errors == [error]
