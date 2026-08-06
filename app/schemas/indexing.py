from datetime import datetime

from pydantic import BaseModel


class IndexStartResponse(BaseModel):
    job_id: str
    upload_id: str
    status: str
    message: str


class IndexErrorItem(BaseModel):
    step: str
    message: str
    details: dict | None = None
    created_at: datetime


class IndexStatusResponse(BaseModel):
    job_id: str
    upload_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    page_count: int
    chunks_count: int
    vectors_indexed: int
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    errors: list[IndexErrorItem]
