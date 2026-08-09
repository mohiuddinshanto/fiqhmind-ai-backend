import structlog
from fastapi import APIRouter, status

from app.api.v1.deps import DbSession
from app.core.exceptions import MetadataConflictError, NotFoundError
from app.db.models import IngestionJob
from app.db.repositories import IngestionJobRepository, MetadataRepository, UploadRepository
from app.schemas.metadata import (
    MetadataErrorItem,
    MetadataFieldOut,
    MetadataStartResponse,
    MetadataStatusResponse,
    MetadataStructureOut,
    PageMetadataOut,
)
from app.tasks.ingestion import run_metadata_task

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["metadata"])

# Statuses that mean this upload is already being processed — starting another
# metadata run would corrupt the in-flight job. Terminal statuses can be re-run.
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
    "/metadata/{upload_id}",
    response_model=MetadataStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_metadata(upload_id: str, session: DbSession) -> MetadataStartResponse:
    """Run Phase 6 metadata extraction on an already-extracted upload."""
    upload = UploadRepository(session).get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")

    job_repo = IngestionJobRepository(session)
    pipeline_job = job_repo.find_pipeline_job(upload_id)
    if pipeline_job is None or pipeline_job.status != "completed":
        raise MetadataConflictError(
            "metadata extraction requires a completed extraction job (status: "
            f"{pipeline_job.status if pipeline_job else 'none'})"
        )

    job = job_repo.find_for_upload(upload_id, "metadata")
    if job is not None and job.status in _IN_FLIGHT_STATUSES:
        raise MetadataConflictError(
            f"metadata extraction already in progress (job status: {job.status})"
        )

    if job is None:
        job = job_repo.create(IngestionJob(upload_id=upload_id, kind="metadata", status="queued"))
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
        run_metadata_task.delay(job.id, upload.id)
    except Exception:  # noqa: BLE001 - job row survives for later reconciliation
        logger.warning("metadata_dispatch_failed", job_id=job.id, upload_id=upload.id)

    return MetadataStartResponse(
        job_id=job.id,
        upload_id=upload.id,
        status=job.status,
        message="metadata extraction queued",
    )


@router.get("/metadata/{job_id}", response_model=MetadataStatusResponse)
def get_metadata_status(job_id: str, session: DbSession) -> MetadataStatusResponse:
    """Return the status of a metadata job plus the extracted metadata itself."""
    job = IngestionJobRepository(session).get(job_id)
    if job is None:
        raise NotFoundError("metadata job not found")

    repo = MetadataRepository(session)
    document = repo.get_by_job(job_id)
    upload = UploadRepository(session).get(job.upload_id) if job.upload_id else None
    errors = sorted(job.errors, key=lambda item: item.created_at, reverse=True)[:5]

    fields = repo.list_fields(document) if document else []
    pages = repo.list_pages(document) if document else []
    structures = repo.list_structures(document) if document else []

    return MetadataStatusResponse(
        job_id=job.id,
        upload_id=job.upload_id,
        status=job.status,
        current_step=job.current_step,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        page_count=(
            upload.page_count
            if upload and upload.page_count
            else (document.page_count if document else 0)
        ),
        pages_mapped=len(pages) if pages else (document.page_count if document else 0),
        fields_count=len(fields),
        structures_count=len(structures),
        numbering_system=(document.numbering_system if document else "none"),
        confidence=round(document.confidence, 4) if document else 0.0,
        fields=[
            MetadataFieldOut(
                field=item.field,
                value=item.value,
                confidence=item.confidence,
                source=item.source,
            )
            for item in fields
        ],
        page_mapping=[
            PageMetadataOut(
                pdf_page=item.pdf_page,
                printed_page=item.printed_page,
                printed_page_numeric=item.printed_page_numeric,
                numbering_system=item.numbering_system,
                page_number_uncertain=item.page_number_uncertain,
                confidence=item.confidence,
                source=item.source,
                kitab=item.kitab,
                bab=item.bab,
                fasl=item.fasl,
            )
            for item in pages
        ],
        structures=[
            MetadataStructureOut(
                level=item.level,
                name=item.name,
                page_start=item.page_start,
                page_end=item.page_end,
                confidence=item.confidence,
                source=item.source,
            )
            for item in structures
        ],
        started_at=job.started_at,
        finished_at=job.finished_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        errors=[
            MetadataErrorItem(
                step=item.step,
                message=item.message,
                details=item.details,
                created_at=item.created_at,
            )
            for item in errors
        ],
    )
