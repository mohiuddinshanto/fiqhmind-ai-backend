from datetime import datetime

from pydantic import BaseModel


class MetadataStartResponse(BaseModel):
    job_id: str
    upload_id: str
    status: str
    message: str


class MetadataFieldOut(BaseModel):
    field: str
    value: str | None
    confidence: float
    source: str


class PageMetadataOut(BaseModel):
    pdf_page: int
    printed_page: str
    printed_page_numeric: int | None
    numbering_system: str
    page_number_uncertain: bool
    confidence: float
    source: str
    kitab: str | None = None
    bab: str | None = None
    fasl: str | None = None


class MetadataStructureOut(BaseModel):
    level: str
    name: str
    page_start: int
    page_end: int | None
    confidence: float
    source: str


class MetadataErrorItem(BaseModel):
    step: str
    message: str
    details: dict | None = None
    created_at: datetime


class MetadataStatusResponse(BaseModel):
    job_id: str
    upload_id: str | None
    status: str
    current_step: str | None
    progress_percent: int
    error_message: str | None
    page_count: int
    pages_mapped: int
    fields_count: int
    structures_count: int
    numbering_system: str
    confidence: float
    fields: list[MetadataFieldOut]
    page_mapping: list[PageMetadataOut]
    structures: list[MetadataStructureOut]
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    errors: list[MetadataErrorItem]
