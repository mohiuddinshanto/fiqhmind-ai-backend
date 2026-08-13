"""Chat / answer-generation schemas (Phase 10).

The answer shape mirrors ARCHITECTURE §Phase 10 "Answer format": a structured
JSON object with `answer_language`, `explanation`, `arabic_quotes`, `citations`,
`confidence`, `refusal`, `caveats` and `related`. The same models are used for
the SSE `done` payload and for the persisted `chat_history.answer` JSON.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Phase 10 step 1 request: the question plus the same optional filters as retrieval."""

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
    answer_language: str = Field(default="bn", pattern="^(bn|ar|en)$")
    # Phase 15 M3: true token streaming. When True, POST /chat streams LLM
    # tokens as they are produced (`start -> token* -> done | error`) instead of
    # waiting for the full validated answer. Both modes share the same QA cache.
    stream: bool = Field(
        default=False,
        description="Stream LLM tokens live (true streaming) instead of a post-hoc replay",
    )


class Explanation(BaseModel):
    type: str
    html: str


class ArabicQuote(BaseModel):
    text: str
    translation: str | None = None
    region: str | None = None


class Citation(BaseModel):
    chunk_id: str
    book: str | None = None
    volume: str | None = None
    page: str | None = None
    edition: str | None = None
    chapter: str | None = None


class Refusal(BaseModel):
    reason: str
    closest_evidence: list[str] = Field(default_factory=list)


class Confidence(BaseModel):
    level: str  # high | medium | low — always computed server-side
    retrieval_score: float
    source_agreement: str  # consensus | conflict | unknown
    rationale: str


class ChatAnswer(BaseModel):
    """The full grounded answer; `confidence` is always set by the server."""

    answer_language: str
    explanation: Explanation
    arabic_quotes: list[ArabicQuote] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence
    refusal: Refusal | None = None
    caveats: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)
