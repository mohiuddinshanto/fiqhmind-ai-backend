"""Tests for the Phase 8 hybrid search layer (RRF fusion + payload filters)."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.services.hybrid_search import (
    HybridSearchService,
    PayloadFilter,
    build_payload_filter,
    rrf_fuse,
)


def _point(point_id: str, payload: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, score=1.0, payload=payload or {})


DENSE = [_point("a"), _point("b"), _point("c")]
SPARSE = [_point("b"), _point("c"), _point("d")]


def test_rrf_fuse_ranks_by_reciprocal_rank_sum() -> None:
    hits = rrf_fuse(DENSE, SPARSE, k=60, alpha=0.5)

    assert [hit.chunk_id for hit in hits] == ["b", "c", "a", "d"]
    assert hits[0].score == pytest.approx(0.5 / 61 + 0.5 / 62)
    assert hits[1].score == pytest.approx(0.5 / 63 + 0.5 / 62)


def test_rrf_fuse_never_duplicates_a_chunk() -> None:
    hits = rrf_fuse(DENSE, SPARSE, k=60, alpha=0.5)

    ids = [hit.chunk_id for hit in hits]
    assert len(ids) == len(set(ids))


def test_rrf_fuse_alpha_weights_dense_only() -> None:
    hits = rrf_fuse(DENSE, SPARSE, k=60, alpha=1.0)

    assert [hit.chunk_id for hit in hits] == ["a", "b", "c", "d"]


def test_rrf_fuse_alpha_weights_sparse_only() -> None:
    hits = rrf_fuse(DENSE, SPARSE, k=60, alpha=0.0)

    assert [hit.chunk_id for hit in hits] == ["b", "c", "d", "a"]


def test_rrf_fuse_respects_limit() -> None:
    hits = rrf_fuse(DENSE, SPARSE, k=60, alpha=0.5, limit=2)

    assert [hit.chunk_id for hit in hits] == ["b", "c"]


def test_rrf_fuse_carries_payloads() -> None:
    dense = [_point("a", {"region": "main"}), _point("b", {"region": "footer"})]
    sparse = [_point("b", {"region": "footer"})]

    hits = rrf_fuse(dense, sparse, k=60, alpha=0.5)

    assert hits[0].payload == {"region": "footer"}
    assert hits[1].payload == {"region": "main"}


def test_rrf_fuse_keys_on_payload_chunk_id_not_point_id() -> None:
    """Qdrant point ids are UUIDs (`point_id_for_chunk`); the canonical chunk id
    lives in the payload, so fusion must key on it, not `point.id`."""
    dense = [
        _point("uuid-point-1", {"chunk_id": "a", "region": "main"}),
        _point("uuid-point-2", {"chunk_id": "b", "region": "footer"}),
    ]
    sparse = [
        _point("uuid-point-2", {"chunk_id": "b", "region": "footer"}),
        _point("uuid-point-3", {"chunk_id": "c", "region": "main"}),
    ]

    hits = rrf_fuse(dense, sparse, k=60, alpha=0.5)

    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]


def test_rrf_fuse_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        rrf_fuse(DENSE, SPARSE, alpha=1.5)
    with pytest.raises(ValueError):
        rrf_fuse(DENSE, SPARSE, alpha=-0.1)


def test_rrf_fuse_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        rrf_fuse(DENSE, SPARSE, k=0)


def test_build_payload_filter_returns_none_when_empty() -> None:
    assert build_payload_filter() is None


def test_build_payload_filter_single_condition() -> None:
    filter_ = build_payload_filter(book_id="book-1")

    assert filter_ is not None
    assert isinstance(filter_.must, list)
    assert len(filter_.must) == 1
    assert filter_.must[0].key == "book_id"
    assert filter_.must[0].match.value == "book-1"


def test_build_payload_filter_combines_conditions() -> None:
    filter_ = build_payload_filter(book_id="b", volume="v2", region="main", verified=True)

    assert filter_ is not None
    assert isinstance(filter_.must, list)
    keys = {condition.key for condition in filter_.must}
    assert keys == {"book_id", "volume", "region", "verified"}


def test_build_payload_filter_verified_is_match_value() -> None:
    filter_ = build_payload_filter(verified=True)

    assert filter_ is not None
    assert isinstance(filter_.must, list)
    assert filter_.must[0].match.value is True


class FakeStore:
    def __init__(self) -> None:
        self.dense_calls: list[tuple[list[float], int, models.Filter | None]] = []
        self.sparse_calls: list[tuple[models.SparseVector, int, models.Filter | None]] = []

    def search_dense(self, vector, *, limit, filter_=None):
        self.dense_calls.append((vector, limit, filter_))
        return [_point("a"), _point("b")]

    def search_sparse(self, vector, *, limit, filter_=None):
        self.sparse_calls.append((vector, limit, filter_))
        return [_point("b"), _point("c")]


class RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, text: str):
        self.queries.append(text)
        return SimpleNamespace(
            dense=[0.1] * 4,
            sparse=SimpleNamespace(indices=[1, 2], values=[1.0, 1.0]),
        )


def _store() -> FakeStore:
    return FakeStore()


def test_hybrid_search_runs_both_vectors_and_fuses() -> None:
    store = _store()
    embedder = RecordingEmbedder()
    service = HybridSearchService(store, embedder=embedder, k=10)

    hits = service.search("washing before prayer", limit=5, alpha=0.5)

    assert embedder.queries == ["washing before prayer"]
    assert len(store.dense_calls) == 1
    assert len(store.sparse_calls) == 1
    assert store.dense_calls[0][1] == 5
    assert store.sparse_calls[0][1] == 5
    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]


def test_hybrid_search_forwards_filters_to_both_queries() -> None:
    store = _store()
    service = HybridSearchService(store, embedder=RecordingEmbedder(), k=10)
    filters = PayloadFilter(book_id="book-1", verified=True)

    service.search("query", filters=filters)

    expected = build_payload_filter(book_id="book-1", verified=True)
    assert store.dense_calls[0][2] == expected
    assert store.sparse_calls[0][2] == expected


def test_hybrid_search_without_filters_passes_none() -> None:
    store = _store()
    service = HybridSearchService(store, embedder=RecordingEmbedder(), k=10)

    service.search("query")

    assert store.dense_calls[0][2] is None
    assert store.sparse_calls[0][2] is None


class BlockingStore:
    """Blocks both searches until released; each signals when it was entered.

    If the dense and sparse searches are truly concurrent, *both* signals are
    set while neither search has returned — no sleeps needed, the barriers
    decide.
    """

    def __init__(self) -> None:
        self.dense_started = threading.Event()
        self.sparse_started = threading.Event()
        self.release = threading.Event()

    def search_dense(self, vector, *, limit, filter_=None):
        self.dense_started.set()
        self.release.wait(timeout=5)
        return [_point("a"), _point("b")]

    def search_sparse(self, vector, *, limit, filter_=None):
        self.sparse_started.set()
        self.release.wait(timeout=5)
        return [_point("b"), _point("c")]


def test_hybrid_search_runs_dense_and_sparse_concurrently() -> None:
    """Phase 15 §706: dense + sparse are in flight at the same time (barriers)."""
    store = BlockingStore()
    service = HybridSearchService(store, embedder=RecordingEmbedder(), k=10)
    results: dict[str, list] = {}

    def _run() -> None:
        results["hits"] = service.search("washing before prayer", limit=5)

    thread = threading.Thread(target=_run)
    thread.start()

    assert store.dense_started.wait(timeout=5)
    assert store.sparse_started.wait(timeout=5)  # both entered before either returned
    store.release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [hit.chunk_id for hit in results["hits"]] == ["b", "a", "c"]


def test_hybrid_search_async_api_returns_fused_results() -> None:
    store = _store()
    service = HybridSearchService(store, embedder=RecordingEmbedder(), k=10)

    hits = asyncio.run(service.search_async("washing before prayer", limit=5))

    assert len(store.dense_calls) == 1
    assert len(store.sparse_calls) == 1
    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]
