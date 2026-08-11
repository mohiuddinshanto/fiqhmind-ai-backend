from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from app.api.v1.deps import DbSession, get_store_dep, require_rate_limit
from app.core.qdrant import QdrantStore
from app.schemas.retrieval import (
    RetrievalChunk,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)
from app.services.hybrid_search import PayloadFilter
from app.services.retrieval import RetrievalRunner

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
def retrieval_search(
    request: RetrievalSearchRequest,
    session: DbSession,
    store: Annotated[QdrantStore, Depends(get_store_dep)],
    _: Annotated[None, Depends(require_rate_limit("search"))] = None,
) -> RetrievalSearchResponse:
    """Phase 9 pipeline: preprocess → translate → expand → hybrid → rerank → compress.

    Metadata filters (book_id/volume/region/verified) are applied at query time
    inside Qdrant, per ARCHITECTURE §Phase 9. LLM answer generation is Phase 10;
    this endpoint returns the compressed evidence and an `evidence_sufficient`
    verdict for the caller to act on.
    """
    runner = RetrievalRunner(session, store)
    result = runner.search(
        request.query,
        filters=PayloadFilter(
            book_id=request.book_id,
            volume=request.volume,
            region=request.region,
            verified=request.verified,
        ),
        top_n=request.top_n,
    )
    return RetrievalSearchResponse(
        query=result.query,
        canonical_arabic_query=result.canonical_arabic_query,
        language=result.language,
        translated=result.translated,
        candidates=result.candidates,
        evidence_sufficient=result.evidence_sufficient,
        chunks=[RetrievalChunk.model_validate(chunk) for chunk in result.chunks],
    )
