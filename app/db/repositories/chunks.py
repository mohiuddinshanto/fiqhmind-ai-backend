from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select

from app.db.models import Chunk, Page
from app.db.repositories.base import RepositoryBase

if TYPE_CHECKING:
    from app.services.chunking import ChunkResult


class ChunkRepository(RepositoryBase[Chunk]):
    model = Chunk

    def get_by_hash(self, chunk_id: str) -> Chunk | None:
        return self._session.get(Chunk, chunk_id)

    def list_by_job(self, job_id: str, *, skip: int = 0, limit: int = 100) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk)
                .where(Chunk.job_id == job_id)
                .order_by(Chunk.order_index)
                .offset(skip)
                .limit(limit)
            )
        )

    def list_all_by_job(self, job_id: str) -> list[Chunk]:
        """Every chunk of a job in order, no limit (indexing needs the full set)."""
        return list(
            self._session.scalars(
                select(Chunk).where(Chunk.job_id == job_id).order_by(Chunk.order_index)
            )
        )

    def list_all_ids(self) -> list[str]:
        """Every chunk_id in the corpus (weekly index health orphan detection)."""
        return list(self._session.scalars(select(Chunk.chunk_id)))

    def list_duplicate_normalized_texts(self) -> list[tuple[str, int]]:
        """Normalized texts that appear in more than one chunk (duplicate detection).

        `chunk_id` is content-addressed (sha256 of the normalized text), so a
        duplicate row means the same passage was chunked twice — typically a
        stale chunking run that `save_for_job` did not fully replace.
        """
        rows = self._session.execute(
            select(Chunk.normalized_text, func.count(Chunk.chunk_id))
            .where(Chunk.normalized_text.is_not(None))
            .group_by(Chunk.normalized_text)
            .having(func.count(Chunk.chunk_id) > 1)
        ).all()
        return [(text, int(count)) for text, count in rows]

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
                select(Chunk).where(Chunk.volume_id == volume_id).order_by(Chunk.printed_page_start)
            )
        )

    def list_by_page(self, volume_id: str, printed_page: int) -> list[Chunk]:
        return list(
            self._session.scalars(
                select(Chunk)
                .join(
                    Page,
                    (Page.id == Chunk.page_start_id) | (Page.id == Chunk.page_end_id),
                )
                .where(
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
