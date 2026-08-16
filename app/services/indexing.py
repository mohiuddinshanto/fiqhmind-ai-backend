"""Vector indexing (Phase 8 — Vector Database).

`IndexingRunner` turns a chunking job's `Chunk` rows into Qdrant points and
upserts them into the `fiqh_chunks` collection with the ARCHITECTURE payload
contract (book/volume/page anchors, hierarchy, region, edition, verified).
The embedding model behind the vectors is the Phase 8 `Embedder` interface
(see `app/services/embedding.py`); the runner never talks to the model directly.

Idempotency matches the repository pattern: before indexing, every point for
the upload is deleted by an `upload_id` payload filter, then all chunks are
re-upserted (point id = `point_id_for_chunk(chunk_id)`, so identical content
overwrites the same point). Re-runs therefore never leave stale vectors behind.
Qdrant has no transactions, so a failed run retries the whole delete-then-upsert
cycle.
"""

from dataclasses import dataclass
from datetime import datetime

import structlog
from qdrant_client import models
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import IndexConflictError
from app.core.qdrant import SPARSE_VECTOR_NAME, QdrantStore, point_id_for_chunk
from app.db.models import Chunk, IngestionJob, MetadataDocument, Upload
from app.db.repositories import ChunkRepository, IngestionJobRepository, MetadataRepository
from app.services.cache import CacheService
from app.services.embedding import (
    Embedder,
    Embedding,
    build_cached_embedder,
    get_embedder,
)

logger = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


@dataclass
class IndexingResult:
    page_count: int
    chunk_count: int
    vectors_indexed: int


def _upload_filter(upload_id: str) -> models.Filter:
    """Match every point that was indexed for an upload (idempotent re-index)."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="upload_id",
                match=models.MatchValue(value=upload_id),
            )
        ]
    )


def build_chunk_payload(
    chunk: Chunk,
    *,
    upload_id: str,
    job_id: str | None,
    book_name: str | None,
    author: str | None,
    volume: str | None,
    edition: str | None,
    publisher: str | None,
    year: str | None,
) -> dict:
    """The ARCHITECTURE payload contract plus upload/job scope for re-indexing."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.raw_text,
        "book_id": chunk.book_id,
        "book_name": book_name,
        "author": author,
        "volume": volume,
        "printed_page_start": chunk.printed_page_start,
        "printed_page_end": chunk.printed_page_end,
        "pdf_page_start": chunk.pdf_page_start,
        "pdf_page_end": chunk.pdf_page_end,
        "kitab": chunk.kitab,
        "bab": chunk.bab,
        "fasl": chunk.fasl,
        "topic": chunk.topic,
        "region": chunk.region,
        "lang": chunk.lang,
        "edition_id": chunk.edition_id,
        "publisher": publisher,
        "year": year,
        "verified": chunk.verified,
        "upload_id": upload_id,
        "job_id": job_id,
    }


class IndexingRunner:
    """Streams a chunking job's chunks through the embedder into Qdrant."""

    def __init__(
        self,
        session: Session,
        store: QdrantStore,
        embedder: Embedder | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_batch_size: int | None = None,
        cache: CacheService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._session = session
        self._store = store
        self._settings = settings or get_settings()
        embedder = embedder or get_embedder(self._settings)
        if cache is not None:
            embedder = build_cached_embedder(embedder, cache, self._settings)
        self._embedder = embedder
        self._batch_size = max(int(batch_size), 1)
        self._embedding_batch_size = max(
            int(embedding_batch_size or self._settings.embedding_batch_size), 1
        )
        self._chunk_repo = ChunkRepository(session)
        self._metadata_repo = MetadataRepository(session)

    def run(
        self,
        job: IngestionJob,
        upload: Upload,
        *,
        chunking_job_id: str | None = None,
    ) -> IndexingResult:
        """Embed and index `upload`'s chunks (from its chunking job) under `job`."""
        chunks = self._load_chunks(upload, chunking_job_id)
        document = self._load_metadata_document(upload)
        self._begin(job, upload)

        meta = self._payload_meta(document)
        points = self._build_points(chunks, upload, chunking_job_id, meta)
        self._store.delete_by_filter(_upload_filter(upload.id))

        total = len(points)
        for start in range(0, total, self._batch_size):
            batch = points[start : start + self._batch_size]
            self._store.upsert_points(batch)
            job.progress_percent = min(99, int((start + len(batch)) / total * 100))
            self._session.commit()
            logger.info(
                "indexing_progress",
                job_id=job.id,
                upload_id=upload.id,
                batch=start + len(batch),
                total=total,
            )

        job.progress_percent = 100
        job.current_step = "embedding"
        self._session.commit()

        return IndexingResult(
            page_count=upload.page_count or 0,
            chunk_count=total,
            vectors_indexed=total,
        )

    def _load_chunks(self, upload: Upload, chunking_job_id: str | None) -> list[Chunk]:
        if chunking_job_id is None and upload.id:
            chunking_job = IngestionJobRepository(self._session).find_for_upload(
                upload.id, "chunking"
            )
            chunking_job_id = chunking_job.id if chunking_job else None
        chunks = (
            self._chunk_repo.list_all_by_job(chunking_job_id) if chunking_job_id else []
        )
        if not chunks:
            raise IndexConflictError(
                "indexing requires a completed chunking job with chunks (no chunks found)"
            )
        return chunks

    def _load_metadata_document(self, upload: Upload) -> MetadataDocument:
        metadata_job = IngestionJobRepository(self._session).find_for_upload(upload.id, "metadata")
        document = (
            self._metadata_repo.get_by_job(metadata_job.id) if metadata_job is not None else None
        )
        if document is None:
            raise IndexConflictError(
                "indexing requires a completed metadata extraction (metadata document is missing)"
            )
        return document

    def _payload_meta(self, document: MetadataDocument) -> dict[str, str | None]:
        return {field.field: field.value for field in self._metadata_repo.list_fields(document)}

    def _to_point(
        self,
        chunk: Chunk,
        upload: Upload,
        chunking_job_id: str | None,
        meta: dict[str, str | None],
        embedding: Embedding,
    ) -> models.PointStruct:
        payload = build_chunk_payload(
            chunk,
            upload_id=upload.id,
            job_id=chunking_job_id,
            book_name=meta.get("title"),
            author=meta.get("author"),
            volume=meta.get("volume"),
            edition=meta.get("edition"),
            publisher=meta.get("publisher"),
            year=meta.get("publication_year"),
        )
        return models.PointStruct(
            id=point_id_for_chunk(chunk.chunk_id),
            vector={
                "": embedding.dense,
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=embedding.sparse.indices,
                    values=embedding.sparse.values,
                ),
            },
            payload=payload,
        )

    def _build_points(
        self,
        chunks: list[Chunk],
        upload: Upload,
        chunking_job_id: str | None,
        meta: dict[str, str | None],
    ) -> list[models.PointStruct]:
        """Embed chunks in `embedding_batch_size` batches, preserving input order.

        `embed_batch` returns one vector per input text in the same order, so the
        point list maps 1:1 onto the chunk list (Phase 15 batched embedding).
        """
        points: list[models.PointStruct] = []
        for start in range(0, len(chunks), self._embedding_batch_size):
            batch_chunks = chunks[start : start + self._embedding_batch_size]
            embeddings = self._embedder.embed_batch([chunk.raw_text for chunk in batch_chunks])
            for chunk, embedding in zip(batch_chunks, embeddings):
                points.append(self._to_point(chunk, upload, chunking_job_id, meta, embedding))
        return points

    def _begin(self, job: IngestionJob, upload: Upload) -> None:
        job.status = "embedding"
        job.current_step = "embedding"
        job.progress_percent = 0
        job.started_at = datetime.utcnow()
        upload.status = "processing"
        self._session.commit()
