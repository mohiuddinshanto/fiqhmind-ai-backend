import structlog
from fastapi import APIRouter, status

from app.api.v1.deps import DbSession
from app.core.exceptions import LayoutConflictError, NotFoundError
from app.db.models import IngestionJob
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.schemas.layout import (
    LayoutErrorItem,
    LayoutStartResponse,
    LayoutStatusResponse,
)
from app.tasks.ingestion import run_layout_task

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["layout"])

# Statuses that mean this upload is already being processed — starting another
# layout run would corrupt the in-flight job. Terminal statuses can be re-run.
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
    "/layout/{upload_id}",
    response_model=LayoutStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_layout(upload_id: str, session: DbSession) -> LayoutStartResponse:
    """Run Phase 5 layout analysis on an already-extracted upload."""
    upload = UploadRepository(session).get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")

    job_repo = IngestionJobRepository(session)
    extraction_job = job_repo.find_pipeline_job(upload_id)
    if extraction_job is None or extraction_job.status != "completed":
        raise LayoutConflictError(
            "layout requires a completed extraction job (status: "
            f"{extraction_job.status if extraction_job else 'none'})"
        )

    job = job_repo.find_for_upload(upload_id, "layout")
    if job is not None and job.status in _IN_FLIGHT_STATUSES:
        raise LayoutConflictError(f"layout already in progress (job status: {job.status})")

    if job is None:
        job = job_repo.create(IngestionJob(upload_id=upload_id, kind="layout", status="queued"))
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
        run_layout_task.delay(job.id, upload.id)
    except Exception:  # noqa: BLE001 - job row survives for later reconciliation
        logger.warning("layout_dispatch_failed", job_id=job.id, upload_id=upload.id)

    return LayoutStartResponse(
        job_id=job.id,
        upload_id=upload.id,
        status=job.status,
        message="layout analysis queued",
    )


@router.get("/layout/{job_id}", response_model=LayoutStatusResponse)
def get_layout_status(job_id: str, session: DbSession) -> LayoutStatusResponse:
    """Return the status of a layout job (progress, region counts)."""
    job = IngestionJobRepository(session).get(job_id)
    if job is None:
        raise NotFoundError("layout job not found")

    repo = ExtractionRepository(session)
    summary = repo.job_summary(job_id)
    upload = UploadRepository(session).get(job.upload_id) if job.upload_id else None
    total_pages = upload.page_count if upload and upload.page_count else summary["page_count"]
    errors = sorted(job.errors, key=lambda item: item.created_at, reverse=True)[:5]

    return LayoutStatusResponse(
        job_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        page_count=int(total_pages),
        pages_processed=int(summary["page_count"]),
        block_count=int(summary["block_count"]),
        region_counts=repo.region_summary(job_id),
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        errors=[
            LayoutErrorItem(
                step=item.step,
                message=item.message,
                details=item.details,
                created_at=item.created_at,
            )
            for item in errors
        ],
    )
