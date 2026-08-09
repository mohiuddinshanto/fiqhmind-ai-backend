"""Chat endpoint (Phase 10): grounded answer over SSE.

`POST /api/v1/chat` runs the Phase 9 retrieval pipeline, then the Phase 10
generation loop, persists the turn to `chat_history`, and streams the answer as
typed Server-Sent Events in the ARCHITECTURE §Phase 10 order:

    meta → sources → delta → quote → citation → confidence → done

The full answer is generated and validated *before* the stream starts, so the
client only ever sees a validated, grounded answer. If an unexpected error
occurs mid-stream, an `error` event is emitted before the stream closes.
"""

import json
from collections.abc import AsyncIterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.deps import DbSession, get_store_dep
from app.core.qdrant import QdrantStore
from app.db.repositories import ChatHistoryRepository
from app.schemas.chat import ChatAnswer, ChatRequest
from app.schemas.retrieval import RetrievalChunk
from app.services.generation.service import get_generator
from app.services.hybrid_search import PayloadFilter
from app.services.retrieval import RetrievalResult, RetrievalRunner

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

_DELTA_TOKENS = 4


@router.post("/chat")
def chat_question(
    request: ChatRequest,
    session: DbSession,
    store: Annotated[QdrantStore, Depends(get_store_dep)],
) -> StreamingResponse:
    """Retrieve evidence, generate a grounded answer, stream it as SSE."""
    runner = RetrievalRunner(session, store)
    retrieval = runner.search(
        request.query,
        filters=PayloadFilter(
            book_id=request.book_id,
            volume=request.volume,
            region=request.region,
            verified=request.verified,
        ),
        top_n=request.top_n,
    )
    answer = get_generator().generate(retrieval, answer_language=request.answer_language)

    ChatHistoryRepository(session).add(
        question=retrieval.query,
        normalized_query=retrieval.canonical_arabic_query,
        answer_language=answer.answer_language,
        answer=answer.model_dump(mode="json"),
        sources=[
            RetrievalChunk.model_validate(chunk).model_dump(mode="json")
            for chunk in retrieval.chunks
        ],
        confidence=answer.confidence.level,
        refusal=answer.refusal.reason if answer.refusal else None,
    )

    return StreamingResponse(
        _stream_events(request, retrieval, answer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_events(
    request: ChatRequest, retrieval: RetrievalResult, answer: ChatAnswer
) -> AsyncIterator[str]:
    try:
        yield _sse(
            "meta",
            {
                "query": retrieval.query,
                "language": retrieval.language,
                "answer_language": answer.answer_language,
                "latency_budget_ms": 3000,
            },
        )
        yield _sse(
            "sources",
            {
                "count": len(retrieval.chunks),
                "evidence_sufficient": retrieval.evidence_sufficient,
                "chunks": [
                    RetrievalChunk.model_validate(chunk).model_dump(mode="json")
                    for chunk in retrieval.chunks
                ],
            },
        )

        for delta in _split_explanation(answer.explanation.html):
            yield _sse("delta", {"text": delta})

        for quote in answer.arabic_quotes:
            yield _sse("quote", quote.model_dump(mode="json"))

        for citation in answer.citations:
            yield _sse("citation", citation.model_dump(mode="json"))

        yield _sse("confidence", answer.confidence.model_dump(mode="json"))
        yield _sse(
            "done",
            {
                "answer_language": answer.answer_language,
                "refusal": answer.refusal.model_dump(mode="json") if answer.refusal else None,
                "caveats": answer.caveats,
                "related": answer.related,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive mid-stream failure
        logger.exception("sse_stream_failed")
        yield _sse("error", {"code": "stream_error", "message": str(exc)})


def _split_explanation(html: str) -> list[str]:
    """Deterministically chunk the explanation into word-group deltas."""
    words = html.split(" ")
    return [
        " ".join(words[i : i + _DELTA_TOKENS]) + " "
        for i in range(0, max(len(words), 1), _DELTA_TOKENS)
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
