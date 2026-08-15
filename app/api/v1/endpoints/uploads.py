from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.api.v1.deps import DbSession, get_storage_dep
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DuplicateUploadError,
    NotFoundError,
    UploadTooLargeError,
    UploadValidationError,
)
from app.core.storage import StorageProvider
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.schemas.uploads import (
    UploadBatchItem,
    UploadBatchResponse,
    UploadDetailResponse,
    UploadErrorDetail,
    UploadListResponse,
    UploadLogResponse,
    UploadResponse,
)
from app.services.uploads import UploadService, sanitize_filename

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["uploads"])

_UPLOAD_ERRORS = (UploadValidationError, DuplicateUploadError, UploadTooLargeError)


@router.post("/uploads", status_code=status.HTTP_201_CREATED)
def create_upload(
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
    session: DbSession,
    storage: Annotated[StorageProvider, Depends(get_storage_dep)],
    settings: Annotated[Settings, Depends(get_settings)],
    response: Response,
) -> UploadResponse | UploadBatchResponse:
    """Upload one or more PDFs. Single-file failures raise a typed error.

    Multi-file requests return a per-file batch; each item carries its own
    success flag and error detail.
    """
    if not files:
        raise UploadValidationError("no files provided")
    if len(files) > settings.upload_max_files_per_request:
        raise UploadValidationError(
            f"too many files: at most {settings.upload_max_files_per_request} per request"
        )

    service = UploadService(session=session, storage=storage, settings=settings)
    results: list[UploadBatchItem] = []
    failures: list[BaseException] = []

    for upload_file in files:
        filename = sanitize_filename(upload_file.filename or "document.pdf")
        try:
            upload = service.receive(
                upload_file.file,
                original_filename=filename,
                content_type=upload_file.content_type,
            )
            results.append(
                UploadBatchItem(
                    filename=filename,
                    success=True,
                    upload=UploadResponse.model_validate(upload),
                )
            )
        except _UPLOAD_ERRORS as exc:
            failures.append(exc)
            results.append(
                UploadBatchItem(
                    filename=filename,
                    success=False,
                    error=UploadErrorDetail(
                        code=exc.code, message=exc.message, details=exc.details
                    ),
                )
            )

    if len(results) == 1:
        if failures:
            raise failures[0]
        item = results[0]
        assert item.upload is not None
        return item.upload

    response.status_code = status.HTTP_201_CREATED if not failures else status.HTTP_200_OK
    return UploadBatchResponse(results=results)


@router.get("/uploads", response_model=UploadListResponse)
def list_uploads(
    session: DbSession,
    skip: int = 0,
    limit: int = 100,
) -> UploadListResponse:
    """List uploads, newest first. `skip`/`limit` paginate the result."""
    repo = UploadRepository(session)
    bounded = max(0, min(limit, 100))
    items = repo.list(skip=max(0, skip), limit=bounded)
    return UploadListResponse(
        items=[UploadResponse.model_validate(upload) for upload in items],
        total=repo.count(),
    )


@router.get("/uploads/{upload_id}", response_model=UploadDetailResponse)
def get_upload(upload_id: str, session: DbSession) -> UploadDetailResponse:
    """Return a single upload with its lifecycle logs and ingestion job."""
    repo = UploadRepository(session)
    upload = repo.get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")
    detail = UploadDetailResponse.model_validate(upload)
    detail.logs = [UploadLogResponse.model_validate(log) for log in upload.logs]
    pipeline_job = IngestionJobRepository(session).find_pipeline_job(upload_id)
    detail.ingestion_job_id = pipeline_job.id if pipeline_job is not None else None
    return detail


@router.delete("/uploads/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_upload(
    upload_id: str,
    storage: Annotated[StorageProvider, Depends(get_storage_dep)],
    session: DbSession,
) -> None:
    """Delete an upload: storage object, row, logs and linked ingestion job."""
    repo = UploadRepository(session)
    upload = repo.get(upload_id)
    if upload is None:
        raise NotFoundError("upload not found")
    UploadService(session=session, storage=storage, settings=get_settings()).delete(upload)
