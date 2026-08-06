import structlog
from fastapi import APIRouter, status

from app.api.v1.deps import DbSession
from app.core.exceptions import ChunkConflictError, NotFoundError
from app.db.models import IngestionJob
from app.db.repositories import (
    ChunkRepository,
    IngestionJobRepository,
    MetadataRepository,
    UploadRepository,
)
from app.schemas.chunking import (
    ChunkErrorItem,
    ChunkOut,
    ChunkStartResponse,
    ChunkStatusResponse,
)
from app.tasks.ingestion import run_chunking_task

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chunking"])

# Statuses that mean this upload is already being processed — starting another
# chunking run would corrupt the in-flight job. Terminal statuses can be re-run.
_IN_FLIGHT_STATUSES = {
    "processing",
    "extracting",
    "sanitizing",
    "ocr",
    "ocr_correcting",
    "structuring",
    "chunking",
    "embedding",
}


@router.post(
    "/chunking/{upload_id}",
    response_model=ChunkStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_chunking(upload_id: str, session: DbSession) -> ChunkStartResponse:
    """Run Phase 7 structure-aware chunking on an extracted, metadata-mapped upload."""
    upload = UploadRepository(session).get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")

    job_repo = IngestionJobRepository(session)
    pipeline_job = job_repo.find_pipeline_job(upload_id)
    if pipeline_job is None or pipeline_job.status != "completed":
        raise ChunkConflictError(
            "chunking requires a completed extraction job (status: "
            f"{pipeline_job.status if pipeline_job else 'none'})"
        )

    metadata_job = job_repo.find_for_upload(upload_id, "metadata")
    if (
        metadata_job is None
        or metadata_job.status != "completed"
        or MetadataRepository(session).get_by_job(metadata_job.id) is None
    ):
        raise ChunkConflictError(
            "chunking requires a completed metadata extraction job (status: "
            f"{metadata_job.status if metadata_job else 'none'})"
        )

    job = job_repo.find_for_upload(upload_id, "chunking")
    if job is not None and job.status in _IN_FLIGHT_STATUSES:
        raise ChunkConflictError(f"chunking already in progress (job status: {job.status})")

    if job is None:
        job = job_repo.create(
            IngestionJob(upload_id=upload_id, kind="chunking", status="queued")
        )
    else:
        job.status = "queued"
        job.progress_percent = 0
        job.current_step = None
        job.error_message = None
        job.finished_at = None
        job_repo.update(job)

    upload.status = "queued"
    upload.error_message = None
    session.commit()

    try:
        run_chunking_task.delay(job.id, upload.id, metadata_job.id)
    except Exception:  # noqa: BLE001 - job row survives for later reconciliation
        logger.warning("chunking_dispatch_failed", job_id=job.id, upload_id=upload.id)

    return ChunkStartResponse(
        job_id=job.id,
        upload_id=upload.id,
        status=job.status,
        message="chunking queued",
    )


@router.get("/chunking/{job_id}", response_model=ChunkStatusResponse)
def get_chunking_status(job_id: str, session: DbSession) -> ChunkStatusResponse:
    """Return the status of a chunking job plus its produced chunks."""
    job = IngestionJobRepository(session).get(job_id)
    if job is None:
        raise NotFoundError("chunking job not found")

    repo = ChunkRepository(session)
    chunks = repo.list_by_job(job_id)
    upload = UploadRepository(session).get(job.upload_id) if job.upload_id else None
    errors = sorted(job.errors, key=lambda item: item.created_at, reverse=True)[:5]

    return ChunkStatusResponse(
        job_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        page_count=(upload.page_count if upload and upload.page_count else 0),
        chunks_count=len(chunks),
        pages_covered=len(
            {
                page
                for chunk in chunks
                for page in (chunk.pdf_page_start, chunk.pdf_page_end)
                if page is not None
            }
        ),
        tokens_count=sum(chunk.token_count for chunk in chunks),
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        errors=[
            ChunkErrorItem(
                step=item.step,
                message=item.message,
                details=item.details,
                created_at=item.created_at,
            )
            for item in errors
        ],
        chunks=[
            ChunkOut(
                chunk_id=chunk.chunk_id,
                order_index=chunk.order_index,
                printed_page_start=chunk.printed_page_start,
                printed_page_end=chunk.printed_page_end,
                pdf_page_start=chunk.pdf_page_start,
                pdf_page_end=chunk.pdf_page_end,
                kitab=chunk.kitab,
                bab=chunk.bab,
                fasl=chunk.fasl,
                topic=chunk.topic,
                context_heading=chunk.context_heading,
                region=chunk.region,
                lang=chunk.lang,
                raw_text=chunk.raw_text,
                normalized_text=chunk.normalized_text,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ],
    )
