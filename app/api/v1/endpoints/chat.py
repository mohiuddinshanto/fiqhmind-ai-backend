"""Chat endpoint (Phase 10): grounded answer over SSE.

`POST /api/v1/chat` runs the Phase 9 retrieval pipeline, then the Phase 10
generation loop, persists the turn to `chat_history`, and streams the answer as
typed Server-Sent Events in the ARCHITECTURE §Phase 10 order:

    meta → sources → delta → quote → citation → confidence → done

The full answer is generated and validated *before* the stream starts, so the
client only ever sees a validated, grounded answer. If an unexpected error
occurs mid-stream, an `error` event is emitted before the stream closes.

Phase 15: the validated answer (plus the retrieval context it was built from) is
cached under the QA answer cache keyed by the full request scope, so repeated
questions skip retrieval + generation entirely. Cache failures degrade to the
uncached path and never break the request.
"""

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.deps import (
    DbSession,
    get_cache_service,
    get_store_dep,
    require_rate_limit,
)
from app.core.config import get_settings
from app.core.qdrant import QdrantStore
from app.db.repositories import ChatHistoryRepository
from app.schemas.chat import ChatAnswer, ChatRequest
from app.schemas.retrieval import RetrievalChunk
from app.services.cache import CacheService
from app.services.generation.service import get_generator
from app.services.hybrid_search import PayloadFilter
from app.services.retrieval import (
    RetrievalResult,
    RetrievalRunner,
    _retrieval_from_dict,
    _retrieval_to_dict,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

_DELTA_TOKENS = 4


@router.post("/chat")
def chat_question(
    request: ChatRequest,
    session: DbSession,
    store: Annotated[QdrantStore, Depends(get_store_dep)],
    cache: Annotated[CacheService, Depends(get_cache_service)],
    _: Annotated[None, Depends(require_rate_limit("chat"))] = None,
) -> StreamingResponse:
    """Retrieve evidence, generate a grounded answer, stream it as SSE."""
    cache_key = _qa_cache_key(request)
    cached = _cached_answer(cache, cache_key)
    if cached is not None:
        retrieval, answer = cached
    else:
        runner = RetrievalRunner(session, store, cache=cache)
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
        cache.set(
            cache_key,
            {
                "retrieval": _retrieval_to_dict(retrieval),
                "answer": answer.model_dump(mode="json"),
            },
            ttl_seconds=get_settings().cache_qa_ttl_seconds,
        )

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


def _qa_cache_key(request: ChatRequest) -> str:
    """Scope-hashed QA cache key: every input that shapes the answer is included."""
    scope = json.dumps(
        [
            request.query,
            request.book_id,
            request.volume,
            request.region,
            request.verified,
            request.top_n,
            request.answer_language,
        ],
        ensure_ascii=False,
    )
    return f"qa:v1:{hashlib.sha256(scope.encode('utf-8')).hexdigest()}"


def _cached_answer(
    cache: CacheService, cache_key: str
) -> tuple[RetrievalResult, ChatAnswer] | None:
    """Read the QA cache; a corrupt/partial entry is recomputed, never fatal."""
    cached = cache.get(cache_key)
    if cached is None:
        return None
    try:
        retrieval = _retrieval_from_dict(cached["retrieval"])
        answer = ChatAnswer.model_validate(cached["answer"])
    except Exception as exc:  # noqa: BLE001 - corrupt cache is recomputed
        logger.warning("qa_cache_read_failed", key=cache_key, error=str(exc))
        return None
    return retrieval, answer


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
