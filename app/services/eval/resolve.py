"""Resolve gold citation anchors to concrete chunk ids (Phase 17).

The gold set stores human-readable anchors (book + volume + printed page). To
compute retrieval metrics (Recall@K, MRR, nDCG@10) the harness must map those
anchors to the actual chunk ids served by the vector store. This resolver walks
the relational corpus (Book → Edition → Volume → Page → Chunk) exactly the way
the ingestion pipeline materializes it, so the eval and the pipeline can never
drift apart.
"""

from __future__ import annotations

import re

from sqlalchemy import select

from app.db.models import Book, Chunk, Edition, Volume
from app.services.eval.gold import GoldItem


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def resolve_expected_chunk_ids(item: GoldItem, session) -> list[str]:
    """Return the chunk ids matching every expected citation of a gold item.

    Matching rules (mirror the payload contract): the book is matched by title
    (case/space-insensitive) — or by `book_id` when the item is scoped to one
    book — then the volume by its number, then any chunk whose printed page
    range contains the cited page. Unmatched citations yield no ids; the caller
    decides how to treat items with no resolvable gold chunks.
    """
    found: set[str] = set()
    for citation in item.expected_citations:
        chunks = _chunks_for_citation(
            session,
            book_id=item.book_id,
            book_title=citation.book,
            volume=citation.volume,
            page=citation.page,
        )
        found.update(chunks)
    return list(found)


def _chunks_for_citation(
    session,
    *,
    book_id: str | None,
    book_title: str | None,
    volume: str | None,
    page: int | None,
) -> list[str]:
    if book_id is not None:
        book = session.get(Book, book_id)
    elif book_title is not None:
        book = session.scalar(select(Book).where(Book.title.ilike(f"%{book_title}%")))
    else:
        book = None

    if book is None:
        return []

    chunk_query = select(Chunk).where(Chunk.book_id == book.id)
    if volume is not None:
        volume_id = session.scalar(
            select(Volume.id)
            .join(Edition, Edition.id == Volume.edition_id)
            .where(Edition.book_id == book.id, Volume.volume_number == _int_or_0(volume))
        )
        if volume_id is None:
            return []
        chunk_query = chunk_query.where(Chunk.volume_id == volume_id)
    if page is not None:
        chunk_query = chunk_query.where(
            Chunk.printed_page_start <= page, Chunk.printed_page_end >= page
        )

    return [chunk.chunk_id for chunk in session.scalars(chunk_query)]


def _int_or_0(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
