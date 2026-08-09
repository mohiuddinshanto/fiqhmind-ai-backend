from sqlalchemy import select

from app.db.models import Book, Edition, Volume
from app.db.repositories.base import RepositoryBase


class BookRepository(RepositoryBase[Book]):
    model = Book

    def get_by_title_author(self, title: str, author: str) -> Book | None:
        return self._session.scalar(select(Book).where(Book.title == title, Book.author == author))

    def list_public(self, *, skip: int = 0, limit: int = 100) -> list[Book]:
        return list(
            self._session.scalars(
                select(Book)
                .where(Book.status == "published")
                .order_by(Book.title)
                .offset(skip)
                .limit(limit)
            )
        )

    def get_editions(self, book: Book) -> list[Edition]:
        return list(
            self._session.scalars(
                select(Edition).where(Edition.book_id == book.id).order_by(Edition.edition_number)
            )
        )

    def get_volumes(self, book: Book) -> list[Volume]:
        return list(
            self._session.scalars(
                select(Volume)
                .join(Edition, Edition.id == Volume.edition_id)
                .where(Edition.book_id == book.id)
                .order_by(Edition.edition_number, Volume.volume_number)
            )
        )
