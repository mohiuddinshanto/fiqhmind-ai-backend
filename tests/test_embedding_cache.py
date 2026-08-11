"""Tests for the Phase 15 CachingEmbedder (vector cache)."""

import fakeredis
import pytest

from app.services.cache import CacheService
from app.services.embedding import (
    CachingEmbedder,
    DeterministicEmbedder,
    Embedding,
    _embedding_cache_key,
    _embedding_from_json,
    _embedding_to_json,
)


@pytest.fixture()
def cache() -> CacheService:
    server = fakeredis.FakeServer()
    return CacheService(fakeredis.FakeStrictRedis(server=server))


class CountingEmbedder:
    """Records every embedded text so tests can assert cache misses only."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls: list[str] = []

    def embed(self, text: str) -> Embedding:
        self.calls.append(text)
        return DeterministicEmbedder(dim=self.dim).embed(text)

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return [self.embed(text) for text in texts]


def test_embed_roundtrip_through_cache(cache: CacheService) -> None:
    wrapped = CachingEmbedder(CountingEmbedder(), cache, ttl_seconds=60, key_dim=64)

    first = wrapped.embed("ما حكم الوضوء")
    second = wrapped.embed("ما حكم الوضوء")

    assert first.dense == second.dense
    assert first.sparse == second.sparse


def test_embed_computes_once_per_text(cache: CacheService) -> None:
    underlying = CountingEmbedder()
    wrapped = CachingEmbedder(underlying, cache, ttl_seconds=60, key_dim=64)

    wrapped.embed("text")
    wrapped.embed("text")

    assert underlying.calls == ["text"]


def test_cached_embedding_matches_uncached_vector(cache: CacheService) -> None:
    plain = DeterministicEmbedder(dim=64)
    wrapped = CachingEmbedder(CountingEmbedder(dim=64), cache, ttl_seconds=60, key_dim=64)

    cached = wrapped.embed("fiqh text")
    expected = plain.embed("fiqh text")

    # Reusing the cached vector must not change vector correctness.
    assert cached.dense == expected.dense
    assert cached.sparse == expected.sparse


def test_embed_batch_preserves_order_and_caches(cache: CacheService) -> None:
    underlying = CountingEmbedder()
    wrapped = CachingEmbedder(underlying, cache, ttl_seconds=60, key_dim=64)

    first = wrapped.embed_batch(["a", "b", "c"])
    second = wrapped.embed_batch(["a", "b", "c"])

    assert [embedding.dense for embedding in first] == [
        embedding.dense for embedding in second
    ]
    assert underlying.calls == ["a", "b", "c"]


def test_embed_batch_computes_only_misses(cache: CacheService) -> None:
    underlying = CountingEmbedder()
    wrapped = CachingEmbedder(underlying, cache, ttl_seconds=60, key_dim=64)

    wrapped.embed_batch(["a", "b"])
    wrapped.embed_batch(["b", "c"])

    # "b" is already cached; only "c" is computed on the second batch.
    assert underlying.calls == ["a", "b", "c"]


def test_json_serialization_roundtrip() -> None:
    embedding = DeterministicEmbedder(dim=64).embed("round trip")
    restored = _embedding_from_json(_embedding_to_json(embedding))

    assert restored.dense == embedding.dense
    assert restored.sparse == embedding.sparse


def test_cache_key_scopes_by_dim_and_text() -> None:
    assert _embedding_cache_key("x", dim=64) != _embedding_cache_key("x", dim=128)
    assert _embedding_cache_key("x", dim=64) != _embedding_cache_key("y", dim=64)
