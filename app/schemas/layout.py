from datetime import datetime

from pydantic import BaseModel


class LayoutStartResponse(BaseModel):
    job_id: str
    upload_id: str
    status: str
    message: str


class LayoutErrorItem(BaseModel):
    step: str
    message: str
    details: dict | None = None
    created_at: datetime


class LayoutStatusResponse(BaseModel):
    job_id: str
    upload_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    page_count: int
    pages_processed: int
    block_count: int
    region_counts: dict[str, int]
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    errors: list[LayoutErrorItem]
