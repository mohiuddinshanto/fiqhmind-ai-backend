from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from app.db.models import Chunk, Page
from app.db.repositories.base import RepositoryBase

if TYPE_CHECKING:
    from app.services.chunking import ChunkResult


class ChunkRepository(RepositoryBase[Chunk]):
    model = Chunk

    def get_by_hash(self, chunk_id: str) -> Chunk | None:
        return self._session.get(Chunk, chunk_id)

    def list_by_job(
        self, job_id: str, *, skip: int = 0, limit: int = 100
    ) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk)
                .where(Chunk.job_id == job_id)
                .order_by(Chunk.order_index)
                .offset(skip)
                .limit(limit)
            )
        )

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

    def delete_for_job(self, job_id: str) -> None:
        """Remove every chunk produced by a chunking job."""
        self._session.execute(delete(Chunk).where(Chunk.job_id == job_id))
        self._session.commit()

    def save_for_job(self, job_id: str, chunks: list[ChunkResult]) -> None:
        """Replace a job's chunks in one transaction (delete-then-insert)."""
        self.delete_for_job(job_id)
        for chunk in chunks:
            self._session.add(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    job_id=job_id,
                    printed_page_start=chunk.printed_page_start,
                    printed_page_end=chunk.printed_page_end,
                    pdf_page_start=chunk.pdf_page_start,
                    pdf_page_end=chunk.pdf_page_end,
                    kitab=chunk.kitab,
                    bab=chunk.bab,
                    fasl=chunk.fasl,
                    topic=chunk.topic,
                    context_heading=chunk.context_heading,
                    order_index=chunk.order_index,
                    region=chunk.region,
                    lang=chunk.lang,
                    raw_text=chunk.raw_text,
                    normalized_text=chunk.normalized_text,
                    token_count=chunk.token_count,
                    verified=False,
                    needs_review=False,
                )
            )
        self._session.commit()
