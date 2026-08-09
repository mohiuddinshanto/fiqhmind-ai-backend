"""Hybrid search (Phase 8 — Vector Database).

ARCHITECTURE §Phase 8 "Hybrid Search (Dense + Sparse)": run the dense and
sparse queries in parallel and fuse with **Reciprocal Rank Fusion** — the
de-facto robust fusion, no score calibration needed:

    score = Σ 1/(k + rank_i),  k ≈ 60

`alpha` weights the two signals: for exact fiqh terms (زكاة, سفر) lexical
weighted higher; for paraphrased questions dense weighted higher. Default
balanced (alpha = 0.5), refined later by the Phase 17 eval harness.

`rrf_fuse` and `build_payload_filter` are pure (no I/O); `HybridSearchService`
is the thin wiring: embed the query, run both searches in parallel, fuse.
Payload filters (book/volume/region/verified) are applied at query time, not
post-filtered, per ARCHITECTURE §Phase 9 "metadata filters".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import structlog
from qdrant_client import models
from qdrant_client.http.models.models import ScoredPoint

from app.core.config import get_settings
from app.core.qdrant import QdrantStore
from app.services.embedding import Embedder, get_embedder

logger = structlog.get_logger(__name__)

# k in 1/(k + rank). ARCHITECTURE: k ≈ 60.
RRF_K = 60


@dataclass(frozen=True)
class SearchHit:
    """One fused result: a chunk id plus its payload and RRF score."""

    chunk_id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PayloadFilter:
    """Metadata filters applied at query time (all conditions are AND-ed)."""

    book_id: str | None = None
    volume: str | None = None
    region: str | None = None
    verified: bool | None = None


def rrf_fuse(
    dense: list[ScoredPoint],
    sparse: list[ScoredPoint],
    *,
    k: int = RRF_K,
    alpha: float = 0.5,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse two ranked lists of scored points with Reciprocal Rank Fusion.

    `alpha` blends dense vs sparse weight: `score = alpha·Σ1/(k+rank_d) +
    (1-alpha)·Σ1/(k+rank_s)`. A point present in both lists keeps its higher
    (summed) score — the same chunk is never duplicated.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    for rank, point in enumerate(dense, start=1):
        chunk_id = str(point.id)
        scores[chunk_id] += alpha / (k + rank)
        if point.payload:
            payloads.setdefault(chunk_id, point.payload)
    for rank, point in enumerate(sparse, start=1):
        chunk_id = str(point.id)
        scores[chunk_id] += (1.0 - alpha) / (k + rank)
        if point.payload:
            payloads.setdefault(chunk_id, point.payload)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        SearchHit(chunk_id=chunk_id, score=score, payload=payloads.get(chunk_id, {}))
        for chunk_id, score in ordered
    ]


def build_payload_filter(
    *,
    book_id: str | None = None,
    volume: str | None = None,
    region: str | None = None,
    verified: bool | None = None,
) -> models.Filter | None:
    """Build a Qdrant Filter from the exposed metadata filters (None → no filter)."""
    conditions: list[models.Condition] = []
    if book_id is not None:
        conditions.append(
            models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))
        )
    if volume is not None:
        conditions.append(
            models.FieldCondition(key="volume", match=models.MatchValue(value=volume))
        )
    if region is not None:
        conditions.append(
            models.FieldCondition(key="region", match=models.MatchValue(value=region))
        )
    if verified is not None:
        conditions.append(
            models.FieldCondition(key="verified", match=models.MatchValue(value=verified))
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


class HybridSearchService:
    """Runs dense + sparse in parallel and fuses with RRF (Phase 8)."""

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder | None = None,
        *,
        k: int | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder or get_embedder()
        self._k = k or get_settings().vector_rrf_k

    def search(
        self,
        query: str,
        *,
        limit: int = 40,
        alpha: float = 0.5,
        filters: PayloadFilter | None = None,
    ) -> list[SearchHit]:
        """Embed `query` and return the fused top-`limit` hits.

        This is the Phase 8 primitive; query normalization, translation and
        reranking belong to the Phase 9 retrieval pipeline.
        """
        embedding = self._embedder.embed(query)
        filter_kwargs = filters.__dict__ if filters is not None else {}
        query_filter = build_payload_filter(**filter_kwargs)
        sparse = models.SparseVector(
            indices=embedding.sparse.indices,
            values=embedding.sparse.values,
        )
        dense_hits = self._store.search_dense(embedding.dense, limit=limit, filter_=query_filter)
        sparse_hits = self._store.search_sparse(sparse, limit=limit, filter_=query_filter)
        return rrf_fuse(dense_hits, sparse_hits, k=self._k, alpha=alpha, limit=limit)
