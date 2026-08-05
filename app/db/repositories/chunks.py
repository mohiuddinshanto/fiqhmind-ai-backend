from sqlalchemy import select

from app.db.models import Chunk, Page
from app.db.repositories.base import RepositoryBase


class ChunkRepository(RepositoryBase[Chunk]):
    model = Chunk

    def get_by_hash(self, chunk_id: str) -> Chunk | None:
        return self._session.get(Chunk, chunk_id)

    def list_by_book(self, book_id: str, *, skip: int = 0, limit: int = 100) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk)
                .where(Chunk.book_id == book_id)
                .order_by(Chunk.created_at)
                .offset(skip)
                .limit(limit)
            )
        )

    def list_by_volume(self, volume_id: str) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk)
                .where(Chunk.volume_id == volume_id)
                .order_by(Chunk.printed_page_start)
            )
        )

    def list_by_page(self, volume_id: str, printed_page: int) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk).join(
                    Page,
                    (Page.id == Chunk.page_start_id) | (Page.id == Chunk.page_end_id),
                ).where(
                    Page.volume_id == volume_id,
                    Page.printed_page == printed_page,
                )
            )
        )
