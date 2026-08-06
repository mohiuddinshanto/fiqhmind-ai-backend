from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    sha256: str | None
    size: int | None
    mime: str | None
    page_count: int | None
    storage_path: str | None
    status: str
    received_bytes: int
    error_message: str | None
    uploaded_at: datetime
    updated_at: datetime


class UploadLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    upload_id: str
    event: str
    message: str | None
    details: dict | None
    created_at: datetime


class UploadDetailResponse(UploadResponse):
    logs: list[UploadLogResponse] = []
    ingestion_job_id: str | None = None


class UploadErrorDetail(BaseModel):
    code: str
    message: str
    details: dict | None = None


class UploadBatchItem(BaseModel):
    filename: str
    success: bool
    upload: UploadResponse | None = None
    error: UploadErrorDetail | None = None


class UploadBatchResponse(BaseModel):
    results: list[UploadBatchItem]


class UploadListResponse(BaseModel):
    items: list[UploadResponse]
    total: int
