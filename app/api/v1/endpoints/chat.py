"""Chat endpoint (Phase 10 + Phase 15 M3): grounded answer over SSE.

`POST /api/v1/chat` runs the Phase 9 retrieval pipeline, then the generation
loop, persists the turn to `chat_history`, and streams the answer as typed
Server-Sent Events.

Phase 10 mode (`stream=false`, default) generates + validates the full answer
*before* the stream starts and replays it in the ARCHITECTURE §Phase 10 order:

    meta → sources → delta → quote → citation → confidence → done

Phase 15 M3 mode (`stream=true`) streams LLM tokens as they are produced:

    start → token* → done | error

The `done` payload is always the fully validated, structured answer; a cache
hit replays it through the same protocol without retrieval/generation, and a
cache miss caches + persists history only after validation (a failed stream
never writes a partial cache entry).

Phase 15: the validated answer (plus the retrieval context it was built from) is
cached under the QA answer cache keyed by the full request scope, so repeated
questions skip retrieval + generation entirely. Cache failures degrade to the
uncached path and never break the request.
"""

import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

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
from app.services.generation.service import (
    StreamAnswer,
    StreamToken,
    get_generator,
)
from app.services.hybrid_search import PayloadFilter
from app.services.retrieval import (
    RetrievalResult,
    RetrievalRunner,
    RetrievedChunk,
    _retrieval_from_dict,
    _retrieval_to_dict,
)

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

_DELTA_TOKENS = 4
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# Client-visible excerpts stay short so the UI never renders the raw corpus.
_SOURCE_EXCERPT_LIMIT = 200
# Stable client-facing message for the SSE `error` event. The detailed exception
# is logged server-side; never leak stack traces, provider internals, database
# details, filesystem paths or secrets to the client.
_SAFE_ERROR_MESSAGE = "The answer could not be generated. Please try again."


@router.post("/chat")
def chat_question(
    request: ChatRequest,
    session: DbSession,
    store: Annotated[QdrantStore, Depends(get_store_dep)],
    cache: Annotated[CacheService, Depends(get_cache_service)],
    _: Annotated[None, Depends(require_rate_limit("chat"))] = None,
) -> StreamingResponse:
    """Retrieve evidence, then stream the grounded answer as SSE.

    Phase 10 mode (`stream=false`, default): the full answer is generated and
    validated *before* the stream starts, then replayed as typed events in the
    ARCHITECTURE order (`meta -> sources -> delta -> quote -> citation ->
    confidence -> done`).

    Phase 15 M3 mode (`stream=true`): LLM tokens stream as they are produced
    (`start -> token* -> done | error`). A cache hit replays the cached answer
    through the same protocol without retrieval or generation; a cache miss
    accumulates the stream, validates it, caches the final structured answer,
    and persists history exactly once. Both modes share the same QA cache key.
    """
    cache_key = _qa_cache_key(request)
    cached = _cached_answer(cache, cache_key)
    if cached is not None:
        retrieval, answer = cached
        _persist_history(session, retrieval, answer)
        if request.stream:
            return StreamingResponse(
                _stream_cached_answer(request, retrieval, answer),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        return StreamingResponse(
            _stream_events(request, retrieval, answer),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

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

    if request.stream:
        return StreamingResponse(
            _stream_generated_answer(request, retrieval, session, cache, cache_key),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    answer = get_generator().generate(retrieval, answer_language=request.answer_language)
    if answer.refusal is None:
        cache.set(
            cache_key,
            {
                "retrieval": _retrieval_to_dict(retrieval),
                "answer": answer.model_dump(mode="json"),
            },
            ttl_seconds=get_settings().cache_qa_ttl_seconds,
        )
    _persist_history(session, retrieval, answer)
    return StreamingResponse(
        _stream_events(request, retrieval, answer),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
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


def _persist_history(session: Session, retrieval: RetrievalResult, answer: ChatAnswer) -> None:
    """Persist one chat_history row for a completed turn (cache hit or miss)."""
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


def _source_summary(chunk: RetrievedChunk) -> dict[str, object]:
    """Compact client-facing summary of one retrieved chunk (no full text)."""
    text = (chunk.text or "").strip().replace("\n", " ")
    excerpt = (
        text
        if len(text) <= _SOURCE_EXCERPT_LIMIT
        else text[:_SOURCE_EXCERPT_LIMIT].rstrip() + "…"
    )
    return {
        "chunk_id": chunk.chunk_id,
        "book_name": chunk.book_name,
        "volume": chunk.volume,
        "printed_page_start": chunk.printed_page_start,
        "topic": chunk.topic,
        "kitab": chunk.kitab,
        "bab": chunk.bab,
        "region": chunk.region,
        "verified": chunk.verified,
        "excerpt": excerpt,
    }


def _stream_cached_answer(
    request: ChatRequest, retrieval: RetrievalResult, answer: ChatAnswer
) -> Iterator[str]:
    """Replay a cached answer through the streaming protocol (no retrieval/generation).

    The `done` payload carries the full validated structured answer so the
    client can render quotes/citations/confidence exactly like a fresh stream.
    """
    try:
        yield _sse(
            "start",
            {
                "query": retrieval.query,
                "language": retrieval.language,
                "answer_language": answer.answer_language,
                "latency_budget_ms": get_settings().generation_latency_budget_ms,
            },
        )
        for delta in _split_explanation(answer.explanation.html):
            yield _sse("token", {"text": delta})

        yield _sse("done", answer.model_dump(mode="json"))
    except Exception:  # pragma: no cover - defensive mid-stream failure
        logger.exception("sse_cache_replay_failed")
        yield _sse("error", {"code": "stream_error", "message": _SAFE_ERROR_MESSAGE})


def _stream_generated_answer(
    request: ChatRequest,
    retrieval: RetrievalResult,
    session: Session,
    cache: CacheService,
    cache_key: str,
) -> Iterator[str]:
    """Stream live LLM tokens; cache + persist only the validated answer.

    On a cache miss the raw stream is accumulated, validated (schema +
    citation coverage), and only the final structured answer is written to the
    QA cache and chat_history — a failed generation never writes a partial
    entry. Failures emit an `error` event instead of a `done`.
    """
    try:
        yield _sse(
            "start",
            {
                "query": retrieval.query,
                "language": retrieval.language,
                "answer_language": request.answer_language,
                "latency_budget_ms": get_settings().generation_latency_budget_ms,
            },
        )

        answer: ChatAnswer | None = None
        for event in get_generator().stream_answer(
            retrieval, answer_language=request.answer_language
        ):
            if isinstance(event, StreamToken):
                yield _sse("token", {"text": event.text})
            elif isinstance(event, StreamAnswer):
                answer = event.answer
        if answer is None:
            raise RuntimeError("generation stream finished without an answer")

        if answer.refusal is None:
            cache.set(
                cache_key,
                {
                    "retrieval": _retrieval_to_dict(retrieval),
                    "answer": answer.model_dump(mode="json"),
                },
                ttl_seconds=get_settings().cache_qa_ttl_seconds,
            )
        _persist_history(session, retrieval, answer)
        yield _sse("done", answer.model_dump(mode="json"))
    except Exception:
        logger.exception("chat_stream_failed")
        yield _sse("error", {"code": "generation_error", "message": _SAFE_ERROR_MESSAGE})


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
                # Compact summaries only — the full chunk payload stays in the
                # database/`sources` field and is never streamed to the client.
                "chunks": [_source_summary(chunk) for chunk in retrieval.chunks],
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
    except Exception:  # pragma: no cover - defensive mid-stream failure
        logger.exception("sse_stream_failed")
        yield _sse("error", {"code": "stream_error", "message": _SAFE_ERROR_MESSAGE})


def _split_explanation(html: str) -> list[str]:
    """Deterministically chunk the explanation into word-group deltas."""
    words = html.split(" ")
    return [
        " ".join(words[i : i + _DELTA_TOKENS]) + " "
        for i in range(0, max(len(words), 1), _DELTA_TOKENS)
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
