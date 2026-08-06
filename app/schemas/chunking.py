from datetime import datetime

from pydantic import BaseModel


class ChunkStartResponse(BaseModel):
    job_id: str
    upload_id: str
    status: str
    message: str


class ChunkErrorItem(BaseModel):
    step: str
    message: str
    details: dict | None = None
    created_at: datetime


class ChunkOut(BaseModel):
    chunk_id: str
    order_index: int
    printed_page_start: int | None
    printed_page_end: int | None
    pdf_page_start: int | None
    pdf_page_end: int | None
    kitab: str | None
    bab: str | None
    fasl: str | None
    topic: str | None
    context_heading: str | None
    region: str
    lang: str
    raw_text: str
    normalized_text: str | None
    token_count: int


class ChunkStatusResponse(BaseModel):
    job_id: str
    upload_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    page_count: int
    chunks_count: int
    pages_covered: int
    tokens_count: int
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    errors: list[ChunkErrorItem]
    chunks: list[ChunkOut]
