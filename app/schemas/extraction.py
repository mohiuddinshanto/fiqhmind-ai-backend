from datetime import datetime

from pydantic import BaseModel


class ExtractionStartResponse(BaseModel):
    job_id: str
    upload_id: str
    status: str
    message: str


class ExtractionErrorItem(BaseModel):
    step: str
    message: str
    details: dict | None = None
    created_at: datetime


class ExtractionStatusResponse(BaseModel):
    job_id: str
    upload_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    page_count: int
    pages_extracted: int
    has_text_layer: bool
    extraction_confidence: float
    char_count: int
    block_count: int
    image_count: int
    drawing_count: int
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    errors: list[ExtractionErrorItem]
