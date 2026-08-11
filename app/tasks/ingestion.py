"""Celery tasks for the ingestion pipeline (Phases 3 + 4 + 5 + 6 + 7 + 8).

Phase 3 enqueues accepted uploads (`mark_queued`). Phase 4 adds the PDF
extraction worker (`extract_pdf_task`). Phase 5 adds the layout worker
(`run_layout_task`). Phase 6 adds the metadata worker (`run_metadata_task`)
which extracts bibliographic/structural metadata and the page mapping.
Phase 7 adds the chunking worker (`run_chunking_task`) which turns that
metadata + page mapping into structure-aware, page-anchored chunks.
Phase 8 adds the indexing worker (`run_indexing_task`) which embeds and
upserts those chunks into the Qdrant vector store.
"""

import structlog
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    ChunkConflictError,
    EncryptedPdfError,
    IndexConflictError,
    MalformedPdfError,
    PdfPageLimitError,
)
from app.core.postgres import get_session_factory
from app.core.qdrant import get_qdrant_store
from app.core.redis import get_redis
from app.core.storage import get_storage_provider
from app.db.models import IngestionJob, Upload
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.services.cache import CacheService
from app.services.chunking import ChunkRunner
from app.services.extraction import ExtractionRunner
from app.services.indexing import IndexingRunner
from app.services.layout import LayoutRunner
from app.services.metadata import MetadataRunner
from app.worker.celery_app import celery_app

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3


@celery_app.task(name="app.tasks.ingestion.mark_queued")
def mark_queued(job_id: str, upload_id: str) -> None:
    """Mark an accepted upload's ingestion job as queued for processing."""
    session = get_session_factory()()
    try:
        job_repo = IngestionJobRepository(session)
        job = job_repo.get(job_id)
        if job is not None and job.status == "uploaded":
            job_repo.update_status(job, "queued")
            logger.info("ingestion_job_queued", job_id=job_id, upload_id=upload_id)

        upload_repo = UploadRepository(session)
        upload = upload_repo.get(upload_id)
        if upload is not None and upload.status == "uploaded":
            upload.status = "queued"
            upload_repo.update(upload)
    finally:
        session.close()


@celery_app.task(
    name="app.tasks.ingestion.extract_pdf_task",
    bind=True,
    max_retries=MAX_RETRIES,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def extract_pdf_task(self, job_id: str, upload_id: str) -> None:
    """Run PyMuPDF extraction for an upload, persisting page rows with progress.

    Permanent failures (malformed/encrypted PDF) fail the job immediately.
    Transient failures are retried with exponential backoff and recorded in
    `ingestion_errors` for auditability.
    """
    session = get_session_factory()()
    job = None
    upload = None
    try:
        job_repo = IngestionJobRepository(session)
        upload_repo = UploadRepository(session)
        job = job_repo.get(job_id)
        upload = upload_repo.get(upload_id)
        if job is None or upload is None:
            logger.warning("extraction_job_missing", job_id=job_id, upload_id=upload_id)
            return

        runner = ExtractionRunner(
            session,
            get_storage_provider(),
            max_pages=get_settings().upload_max_pages,
        )
        result = runner.run(job, upload)
        job_repo.update_status(job, "completed", progress_percent=100, current_step="extracting")
        upload.status = "completed"
        session.commit()
        logger.info(
            "extraction_completed",
            job_id=job_id,
            upload_id=upload_id,
            page_count=result.page_count,
            has_text_layer=result.has_text_layer,
            confidence=round(result.confidence, 4),
        )
    except (MalformedPdfError, EncryptedPdfError, PdfPageLimitError) as exc:
        session.rollback()
        _record_error_and_fail(session, job, upload, exc.message, permanent=True)
        logger.warning("extraction_permanent_failure", job_id=job_id, error=exc.message)
    except Exception as exc:  # noqa: BLE001 - transient failures are retried
        session.rollback()
        if job is not None:
            IngestionJobRepository(session).add_error(job, "extracting", str(exc))
        if self.request.retries < MAX_RETRIES:
            logger.warning(
                "extraction_retrying", job_id=job_id, retry=self.request.retries + 1, error=str(exc)
            )
            raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
        _record_error_and_fail(session, job, upload, str(exc), permanent=True)
        logger.warning("extraction_retries_exhausted", job_id=job_id, error=str(exc))
    finally:
        session.close()


@celery_app.task(
    name="app.tasks.ingestion.run_layout_task",
    bind=True,
    max_retries=MAX_RETRIES,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_layout_task(self, job_id: str, upload_id: str) -> None:
    """Run Phase 5 layout analysis on stored page rows, persisting results.

    The layout engine is deterministic and cheap; failures are almost always
    transient (DB connection), so retries with backoff cover them.
    """
    session = get_session_factory()()
    job = None
    upload = None
    try:
        job_repo = IngestionJobRepository(session)
        upload_repo = UploadRepository(session)
        job = job_repo.get(job_id)
        upload = upload_repo.get(upload_id)
        if job is None or upload is None:
            logger.warning("layout_job_missing", job_id=job_id, upload_id=upload_id)
            return

        result = LayoutRunner(session).run(job)
        job_repo.update_status(job, "completed", progress_percent=100, current_step="layout")
        upload.status = "completed"
        session.commit()
        logger.info(
            "layout_completed",
            job_id=job_id,
            upload_id=upload_id,
            page_count=result.page_count,
            block_count=result.block_count,
        )
    except Exception as exc:  # noqa: BLE001 - transient failures are retried
        session.rollback()
        if job is not None:
            IngestionJobRepository(session).add_error(job, "layout", str(exc))
        if self.request.retries < MAX_RETRIES:
            logger.warning(
                "layout_retrying", job_id=job_id, retry=self.request.retries + 1, error=str(exc)
            )
            raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
        _record_error_and_fail(session, job, upload, str(exc), step="layout", permanent=True)
        logger.warning("layout_retries_exhausted", job_id=job_id, error=str(exc))
    finally:
        session.close()


@celery_app.task(
    name="app.tasks.ingestion.run_metadata_task",
    bind=True,
    max_retries=MAX_RETRIES,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_metadata_task(self, job_id: str, upload_id: str) -> None:
    """Run Phase 6 metadata extraction on stored page rows, persisting results.

    Metadata extraction is deterministic and cheap; failures are almost always
    transient (DB connection), so retries with backoff cover them.
    """
    session = get_session_factory()()
    job = None
    upload = None
    try:
        job_repo = IngestionJobRepository(session)
        upload_repo = UploadRepository(session)
        job = job_repo.get(job_id)
        upload = upload_repo.get(upload_id)
        if job is None or upload is None:
            logger.warning("metadata_job_missing", job_id=job_id, upload_id=upload_id)
            return

        result = MetadataRunner(session).run(job, upload)
        job_repo.update_status(job, "completed", progress_percent=100, current_step="metadata")
        upload.status = "completed"
        session.commit()
        logger.info(
            "metadata_completed",
            job_id=job_id,
            upload_id=upload_id,
            page_count=result.page_count,
            pages_mapped=result.pages_mapped,
            fields_count=result.fields_count,
            structures_count=result.structures_count,
            numbering_system=result.numbering_system,
            confidence=result.confidence,
        )
    except Exception as exc:  # noqa: BLE001 - transient failures are retried
        session.rollback()
        if job is not None:
            IngestionJobRepository(session).add_error(job, "metadata", str(exc))
        if self.request.retries < MAX_RETRIES:
            logger.warning(
                "metadata_retrying", job_id=job_id, retry=self.request.retries + 1, error=str(exc)
            )
            raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
        _record_error_and_fail(session, job, upload, str(exc), step="metadata", permanent=True)
        logger.warning("metadata_retries_exhausted", job_id=job_id, error=str(exc))
    finally:
        session.close()


def _retry_countdown(retries: int) -> int:
    return min(2**retries * 10, 120)


@celery_app.task(
    name="app.tasks.ingestion.run_indexing_task",
    bind=True,
    max_retries=MAX_RETRIES,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_indexing_task(self, job_id: str, upload_id: str, chunking_job_id: str) -> None:
    """Run Phase 8 vector indexing: embed + upsert chunks into Qdrant.

    Indexing depends on the Qdrant store and the configured embedder, so most
    failures are transient (store unreachable, network) and are retried with
    backoff. A missing chunking run or metadata document is a permanent failure
    and fails the job immediately.
    """
    session = get_session_factory()()
    job = None
    upload = None
    try:
        job_repo = IngestionJobRepository(session)
        upload_repo = UploadRepository(session)
        job = job_repo.get(job_id)
        upload = upload_repo.get(upload_id)
        if job is None or upload is None:
            logger.warning("indexing_job_missing", job_id=job_id, upload_id=upload_id)
            return

        result = IndexingRunner(
            session,
            get_qdrant_store(),
            cache=CacheService(get_redis()),
        ).run(
            job, upload, chunking_job_id=chunking_job_id
        )
        job_repo.update_status(job, "indexed", progress_percent=100, current_step="embedding")
        upload.status = "completed"
        session.commit()
        logger.info(
            "indexing_completed",
            job_id=job_id,
            upload_id=upload_id,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            vectors_indexed=result.vectors_indexed,
        )
    except IndexConflictError as exc:
        session.rollback()
        _record_error_and_fail(session, job, upload, exc.message, permanent=True, step="indexing")
        logger.warning("indexing_permanent_failure", job_id=job_id, error=exc.message)
    except Exception as exc:  # noqa: BLE001 - transient failures are retried
        session.rollback()
        if job is not None:
            IngestionJobRepository(session).add_error(job, "indexing", str(exc))
        if self.request.retries < MAX_RETRIES:
            logger.warning(
                "indexing_retrying", job_id=job_id, retry=self.request.retries + 1, error=str(exc)
            )
            raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
        _record_error_and_fail(session, job, upload, str(exc), step="indexing", permanent=True)
        logger.warning("indexing_retries_exhausted", job_id=job_id, error=str(exc))
    finally:
        session.close()


@celery_app.task(
    name="app.tasks.ingestion.run_chunking_task",
    bind=True,
    max_retries=MAX_RETRIES,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_chunking_task(self, job_id: str, upload_id: str, metadata_job_id: str) -> None:
    """Run Phase 7 structure-aware chunking on stored pages + metadata.

    Chunking is deterministic and cheap; failures are almost always transient
    (DB connection), so retries with backoff cover them. A missing metadata
    document is a permanent failure and fails the job immediately.
    """
    session = get_session_factory()()
    job = None
    upload = None
    try:
        job_repo = IngestionJobRepository(session)
        upload_repo = UploadRepository(session)
        job = job_repo.get(job_id)
        upload = upload_repo.get(upload_id)
        if job is None or upload is None:
            logger.warning("chunking_job_missing", job_id=job_id, upload_id=upload_id)
            return

        result = ChunkRunner(session).run(job, upload, metadata_job_id=metadata_job_id)
        job_repo.update_status(job, "completed", progress_percent=100, current_step="chunking")
        upload.status = "completed"
        session.commit()
        logger.info(
            "chunking_completed",
            job_id=job_id,
            upload_id=upload_id,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            pages_covered=result.pages_covered,
            token_count=result.token_count,
        )
    except ChunkConflictError as exc:
        session.rollback()
        _record_error_and_fail(session, job, upload, exc.message, permanent=True, step="chunking")
        logger.warning("chunking_permanent_failure", job_id=job_id, error=exc.message)
    except Exception as exc:  # noqa: BLE001 - transient failures are retried
        session.rollback()
        if job is not None:
            IngestionJobRepository(session).add_error(job, "chunking", str(exc))
        if self.request.retries < MAX_RETRIES:
            logger.warning(
                "chunking_retrying", job_id=job_id, retry=self.request.retries + 1, error=str(exc)
            )
            raise self.retry(exc=exc, countdown=_retry_countdown(self.request.retries))
        _record_error_and_fail(session, job, upload, str(exc), step="chunking", permanent=True)
        logger.warning("chunking_retries_exhausted", job_id=job_id, error=str(exc))
    finally:
        session.close()


def _record_error_and_fail(
    session: Session,
    job: IngestionJob | None,
    upload: Upload | None,
    message: str,
    *,
    permanent: bool,
    step: str = "extracting",
) -> None:
    """Record an error row and mark the job (and upload) failed when permanent."""
    job_repo = IngestionJobRepository(session)
    if job is not None:
        job_repo.add_error(job, step, message)
        if permanent:
            job_repo.update_status(job, "failed", error_message=message)
    if upload is not None and permanent:
        upload.status = "failed"
        upload.error_message = message
        session.commit()
