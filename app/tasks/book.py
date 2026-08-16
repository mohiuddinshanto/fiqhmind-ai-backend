"""Per-book ingestion orchestration (Phase 15 §713).

The extraction/chunking/indexing leaf tasks stay unchanged; this module wires
them into a per-book task graph:

    extraction → metadata → chunking → group(embed, index)

`build_book_graph` composes the full graph, `build_stage_graph` the sub-graph
an endpoint needs, and `process_book_task` is the dispatch entry point that
endpoints call instead of leaf tasks directly. `start_ingestion_pipeline_task`
creates the stage jobs and dispatches the full graph automatically when a PDF
upload is accepted. `fail_book_jobs` is wired as each leaf's `link_error`
errback so a raising child fails every queued/in-flight job of the book instead
of leaving them stuck; permanent child failures (which the leaf tasks record
without raising) keep the existing job-status semantics.
"""

import structlog
from celery import chain, group

from app.core.postgres import get_session_factory
from app.db.models import IngestionJob
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.tasks.ingestion import (
    embed_chunks_task,
    extract_pdf_task,
    run_chunking_task,
    run_indexing_task,
    run_metadata_task,
)
from app.worker.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _book_errback(upload_id: str, job_id: str):
    """The graph-wide failure hook: fail every active job for the upload."""
    return fail_book_jobs.s(upload_id=upload_id, failed_job_id=job_id)


def build_extraction_stage(job_id: str, upload_id: str):
    """Sub-graph for the extraction endpoint: the extraction leaf task."""
    return chain(
        extract_pdf_task.s(job_id, upload_id).set(
            link_error=_book_errback(upload_id, job_id)
        )
    )


def build_chunking_stage(job_id: str, upload_id: str, metadata_job_id: str):
    """Sub-graph for the chunking endpoint: the chunking leaf task."""
    return chain(
        run_chunking_task.s(job_id, upload_id, metadata_job_id).set(
            link_error=_book_errback(upload_id, job_id)
        )
    )


def build_indexing_stage(job_id: str, upload_id: str, chunking_job_id: str):
    """Sub-graph for the indexing endpoint: embed ∥ index in parallel.

    Embedding pre-warms the vector cache while indexing embeds (cache hits)
    and upserts to Qdrant, so the two worker pools (`embed`, `index`) overlap.
    """
    errback = _book_errback(upload_id, job_id)
    return group(
        embed_chunks_task.si(upload_id, chunking_job_id).set(link_error=errback),
        run_indexing_task.si(job_id, upload_id, chunking_job_id).set(link_error=errback),
    )


def build_stage_graph(
    stage: str,
    *,
    job_id: str,
    upload_id: str,
    metadata_job_id: str | None = None,
    chunking_job_id: str | None = None,
):
    """The task-graph sub-graph an endpoint needs for `stage`."""
    if stage == "extraction":
        return build_extraction_stage(job_id, upload_id)
    if stage == "chunking":
        return build_chunking_stage(job_id, upload_id, metadata_job_id or "")
    if stage == "indexing":
        return build_indexing_stage(job_id, upload_id, chunking_job_id or "")
    raise ValueError(f"unknown ingestion stage: {stage}")


def build_book_graph(
    *,
    extraction_job_id: str,
    upload_id: str,
    chunking_job_id: str,
    indexing_job_id: str,
    metadata_job_id: str,
):
    """The full per-book graph: extraction → metadata → chunking → group(embed, index).

    The metadata stage runs between extraction and chunking because chunking
    depends on a completed metadata run (`run_chunking_task` requires the
    `metadata_job_id`).
    """
    return chain(
        extract_pdf_task.s(extraction_job_id, upload_id).set(
            link_error=_book_errback(upload_id, extraction_job_id)
        ),
        run_metadata_task.si(metadata_job_id, upload_id).set(
            link_error=_book_errback(upload_id, metadata_job_id)
        ),
        run_chunking_task.si(chunking_job_id, upload_id, metadata_job_id).set(
            link_error=_book_errback(upload_id, chunking_job_id)
        ),
        build_indexing_stage(indexing_job_id, upload_id, chunking_job_id),
    )


@celery_app.task(name="app.tasks.book.fail_book_jobs")
def fail_book_jobs(*err_args: object, upload_id: str, failed_job_id: str) -> None:
    """Errback: fail every queued/in-flight job of the book after a child fails.

    `err_args` carries Celery's (request, exc, traceback) regardless of how
    the errback is wired. `list_active_for_upload` only returns non-terminal
    jobs, so the source leaf's own job (already failed when the leaf recorded
    a permanent error) is failed again here only if it is still in flight.
    Failing the upload keeps the book's lifecycle coherent: a dead pipeline
    never reports the upload as completed.
    """
    session = get_session_factory()()
    try:
        message = f"ingestion pipeline failed upstream (job {failed_job_id})"
        repo = IngestionJobRepository(session)
        for job in repo.list_active_for_upload(upload_id):
            repo.update_status(job, "failed", error_message=message)
        upload = UploadRepository(session).get(upload_id)
        if upload is not None:
            upload.status = "failed"
            upload.error_message = message
        session.commit()
        logger.warning(
            "process_book_failed",
            upload_id=upload_id,
            failed_job_id=failed_job_id,
        )
    finally:
        session.close()


@celery_app.task(name="app.tasks.book.process_book")
def process_book_task(
    upload_id: str,
    *,
    stage: str,
    job_id: str,
    metadata_job_id: str | None = None,
    chunking_job_id: str | None = None,
) -> None:
    """Dispatch the per-book task-graph stage for an upload (endpoint entry point)."""
    session = get_session_factory()()
    try:
        upload = UploadRepository(session).get(upload_id)
        if upload is None:
            logger.warning("process_book_upload_missing", upload_id=upload_id, stage=stage)
            return
        graph = build_stage_graph(
            stage,
            job_id=job_id,
            upload_id=upload_id,
            metadata_job_id=metadata_job_id,
            chunking_job_id=chunking_job_id,
        )
        graph.apply_async()
        logger.info("process_book_dispatched", upload_id=upload_id, stage=stage, job_id=job_id)
    finally:
        session.close()


@celery_app.task(name="app.tasks.book.start_ingestion_pipeline")
def start_ingestion_pipeline_task(upload_id: str) -> None:
    """Create the stage jobs and dispatch the full pipeline graph for an upload.

    Called automatically after a PDF upload is accepted (in addition to
    `mark_queued`), so an upload no longer waits for a manual stage endpoint.
    Reuses the upload's existing `initial` job as the extraction job and creates
    the `metadata`, `chunking` and `indexing` job rows when missing, mirroring
    what the manual stage endpoints would create. Existing stage jobs are reused
    so a re-run never duplicates rows.
    """
    session = get_session_factory()()
    try:
        job_repo = IngestionJobRepository(session)
        initial_job = job_repo.find_pipeline_job(upload_id)
        if initial_job is None:
            logger.warning("start_pipeline_missing_extraction_job", upload_id=upload_id)
            return

        metadata_job = job_repo.find_for_upload(upload_id, "metadata")
        if metadata_job is None:
            metadata_job = job_repo.create(
                IngestionJob(upload_id=upload_id, kind="metadata", status="queued")
            )
        chunking_job = job_repo.find_for_upload(upload_id, "chunking")
        if chunking_job is None:
            chunking_job = job_repo.create(
                IngestionJob(upload_id=upload_id, kind="chunking", status="queued")
            )
        indexing_job = job_repo.find_for_upload(upload_id, "indexing")
        if indexing_job is None:
            indexing_job = job_repo.create(
                IngestionJob(upload_id=upload_id, kind="indexing", status="queued")
            )
        session.commit()

        graph = build_book_graph(
            extraction_job_id=initial_job.id,
            upload_id=upload_id,
            chunking_job_id=chunking_job.id,
            indexing_job_id=indexing_job.id,
            metadata_job_id=metadata_job.id,
        )
        graph.apply_async()
        logger.info(
            "ingestion_pipeline_dispatched",
            upload_id=upload_id,
            metadata_job_id=metadata_job.id,
            chunking_job_id=chunking_job.id,
            indexing_job_id=indexing_job.id,
        )
    finally:
        session.close()
