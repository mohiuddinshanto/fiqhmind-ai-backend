from pydantic import BaseModel, ConfigDict, Field


class RetrievalSearchRequest(BaseModel):
    """Phase 9 step 5 request: query + optional metadata filters + top-N."""

    query: str = Field(description="The user's question (bn/ar/en)")
    book_id: str | None = None
    volume: str | None = None
    region: str | None = Field(
        default=None, pattern="^(main|footnote|margin|header|footer|unknown)$"
    )
    verified: bool | None = None
    top_n: int | None = Field(
        default=None, ge=1, le=8, description="Context size; defaults to the configured top-N"
    )


class RetrievalChunk(BaseModel):
    """One evidence chunk with its citation anchor and rerank provenance."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    text: str
    book_name: str | None = None
    volume: str | None = None
    printed_page_start: int | None = None
    printed_page_end: int | None = None
    kitab: str | None = None
    bab: str | None = None
    fasl: str | None = None
    topic: str | None = None
    region: str | None = None
    lang: str | None = None
    verified: bool = False
    rerank_score: float
    arabic_score: float | None = None
    original_score: float | None = None


class RetrievalSearchResponse(BaseModel):
    """Phase 9 step 8 output: compressed evidence + pipeline provenance."""

    query: str
    canonical_arabic_query: str
    language: str
    translated: bool
    candidates: list[str]
    evidence_sufficient: bool
    chunks: list[RetrievalChunk]
