from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, status

from app.api.v1.deps import DbSession, get_storage_dep
from app.core.exceptions import ExtractionConflictError, NotFoundError
from app.core.storage import StorageProvider
from app.db.models import IngestionJob
from app.db.repositories import ExtractionRepository, IngestionJobRepository, UploadRepository
from app.schemas.extraction import (
    ExtractionErrorItem,
    ExtractionStartResponse,
    ExtractionStatusResponse,
)
from app.tasks.ingestion import extract_pdf_task

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["extraction"])

# Statuses that mean "this upload is already being processed" — starting
# extraction again would corrupt the run. Terminal statuses can be re-run.
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
    "/extraction/{upload_id}",
    response_model=ExtractionStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_extraction(
    upload_id: str,
    session: DbSession,
    storage: Annotated[StorageProvider, Depends(get_storage_dep)],
) -> ExtractionStartResponse:
    """Start PDF extraction for an upload. Queues the background worker job."""
    upload = UploadRepository(session).get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")
    if not upload.storage_path or not storage.exists(upload.storage_path):
        raise ExtractionConflictError("stored file is missing for this upload")

    job = upload.ingestion_job
    if job is not None and job.status in _IN_FLIGHT_STATUSES:
        raise ExtractionConflictError(
            f"extraction already in progress (job status: {job.status})"
        )

    job_repo = IngestionJobRepository(session)
    if job is None:
        job = job_repo.create(
            IngestionJob(upload_id=upload_id, kind="extraction", status="queued")
        )
    else:
        ExtractionRepository(session).delete_for_job(job.id)
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
        extract_pdf_task.delay(job.id, upload.id)
    except Exception:  # noqa: BLE001 - job row survives for later reconciliation
        logger.warning("extraction_dispatch_failed", job_id=job.id, upload_id=upload.id)

    return ExtractionStartResponse(
        job_id=job.id,
        upload_id=upload.id,
        status=job.status,
        message="extraction queued",
    )


@router.get("/extraction/{job_id}", response_model=ExtractionStatusResponse)
def get_extraction_status(job_id: str, session: DbSession) -> ExtractionStatusResponse:
    """Return the status of an extraction job (progress, text-layer, confidence)."""
    job = IngestionJobRepository(session).get(job_id)
    if job is None:
        raise NotFoundError("extraction job not found")

    repo = ExtractionRepository(session)
    summary = repo.job_summary(job_id)
    upload = UploadRepository(session).get(job.upload_id) if job.upload_id else None
    total_pages = (upload.page_count if upload and upload.page_count else summary["page_count"])
    errors = sorted(job.errors, key=lambda item: item.created_at, reverse=True)[:5]

    return ExtractionStatusResponse(
        job_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        page_count=int(total_pages),
        pages_extracted=int(summary["page_count"]),
        has_text_layer=summary["text_pages"] > 0,
        extraction_confidence=round(summary["confidence"], 4),
        char_count=int(summary["char_count"]),
        block_count=int(summary["block_count"]),
        image_count=int(summary["image_count"]),
        drawing_count=int(summary["drawing_count"]),
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        errors=[
            ExtractionErrorItem(
                step=item.step,
                message=item.message,
                details=item.details,
                created_at=item.created_at,
            )
            for item in errors
        ],
    )
