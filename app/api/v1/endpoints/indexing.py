import structlog
from fastapi import APIRouter, status

from app.api.v1.deps import DbSession
from app.core.exceptions import IndexConflictError, NotFoundError
from app.db.models import IngestionJob
from app.db.repositories import (
    ChunkRepository,
    IngestionJobRepository,
    UploadRepository,
)
from app.schemas.indexing import (
    IndexErrorItem,
    IndexStartResponse,
    IndexStatusResponse,
)
from app.tasks.ingestion import run_indexing_task

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["indexing"])

# Statuses that mean this upload is already being processed — starting another
# indexing run would corrupt the in-flight job. Terminal statuses can be re-run.
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
    "/indexing/{upload_id}",
    response_model=IndexStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_indexing(upload_id: str, session: DbSession) -> IndexStartResponse:
    """Embed and index a chunked upload's chunks into the vector store."""
    upload = UploadRepository(session).get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")

    job_repo = IngestionJobRepository(session)
    chunking_job = job_repo.find_for_upload(upload_id, "chunking")
    if (
        chunking_job is None
        or chunking_job.status != "completed"
        or not ChunkRepository(session).list_by_job(chunking_job.id)
    ):
        raise IndexConflictError(
            "indexing requires a completed chunking job with chunks (status: "
            f"{chunking_job.status if chunking_job else 'none'})"
        )

    job = job_repo.find_for_upload(upload_id, "indexing")
    if job is not None and job.status in _IN_FLIGHT_STATUSES:
        raise IndexConflictError(f"indexing already in progress (job status: {job.status})")

    if job is None:
        job = job_repo.create(IngestionJob(upload_id=upload_id, kind="indexing", status="queued"))
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
        run_indexing_task.delay(job.id, upload.id, chunking_job.id)
    except Exception:  # noqa: BLE001 - job row survives for later reconciliation
        logger.warning("indexing_dispatch_failed", job_id=job.id, upload_id=upload.id)

    return IndexStartResponse(
        job_id=job.id,
        upload_id=upload.id,
        status=job.status,
        message="indexing queued",
    )


@router.get("/indexing/{job_id}", response_model=IndexStatusResponse)
def get_indexing_status(job_id: str, session: DbSession) -> IndexStatusResponse:
    """Return the status of an indexing job plus its indexed-chunk counts."""
    job = IngestionJobRepository(session).get(job_id)
    if job is None:
        raise NotFoundError("indexing job not found")

    upload = UploadRepository(session).get(job.upload_id) if job.upload_id else None
    errors = sorted(job.errors, key=lambda item: item.created_at, reverse=True)[:5]

    # The indexing job references its input via the payload; the number of
    # chunks it indexed is reported from the chunking job it consumed.
    chunking_job = (
        IngestionJobRepository(session).find_for_upload(job.upload_id, "chunking")
        if job.upload_id
        else None
    )
    chunks_count = len(ChunkRepository(session).list_by_job(chunking_job.id)) if chunking_job else 0

    return IndexStatusResponse(
        job_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        page_count=(upload.page_count if upload and upload.page_count else 0),
        chunks_count=chunks_count,
        vectors_indexed=chunks_count if job.status == "indexed" else 0,
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        errors=[
            IndexErrorItem(
                step=item.step,
                message=item.message,
                details=item.details,
                created_at=item.created_at,
            )
            for item in errors
        ],
    )
