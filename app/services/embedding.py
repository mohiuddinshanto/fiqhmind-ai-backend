"""Embedding adapter (Phase 8 — Vector Database).

The vector DB layer depends on an embedding *interface*, never on the BGE-M3
model itself. `Embedder` is the port; `DeterministicEmbedder` is the default
adapter — a dependency-free, deterministic feature-hashing embedding that makes
the indexer and hybrid-search layers runnable and testable with no model
download. The production adapter (BGE-M3, ARCHITECTURE Phase 7) plugs into the
same interface without touching the indexer, the search layer, or the store.

Determinism matters here: `chunk_id` is a content hash, re-indexing is
delete-then-upsert, and tests assert exact embeddings. Python's built-in
`hash()` is randomized per process for strings, so token hashing uses
`hashlib.md5` digests instead.
"""

import hashlib
import re
from dataclasses import dataclass
from math import sqrt
from typing import Any, Protocol

import structlog

from app.core.config import Settings, get_settings
from app.services.cache import CacheService

logger = structlog.get_logger(__name__)

# BGE-M3 dense dimension (ARCHITECTURE §Phase 8 collection schema).
DEFAULT_DENSE_DIM = 1024
# Sparse hash space: indices live in [0, 2**20), far below Qdrant's max.
DEFAULT_SPARSE_DIM = 1 << 20

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SparseEmbedding:
    """Qdrant-compatible sparse vector: parallel sorted index/value lists."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: SparseEmbedding


class Embedder(Protocol):
    """Port implemented by every embedding adapter (deterministic, BGE-M3…)."""

    def embed(self, text: str) -> Embedding: ...
    def embed_batch(self, texts: list[str]) -> list[Embedding]: ...


def _hash_index(token: str, mod: int) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big") % mod


class DeterministicEmbedder:
    """Feature-hashing embedder: dense bag-of-hashes + sparse token counts.

    Dense is L2-normalized (Cosine distance). Sparse indices are sorted and
    deduplicated with summed term frequencies, as Qdrant requires. Two equal
    texts always embed identically; the adapter is intentionally *not* a
    semantic model — it only exercises the vector DB path.
    """

    def __init__(self, dim: int = DEFAULT_DENSE_DIM, sparse_dim: int = DEFAULT_SPARSE_DIM) -> None:
        self._dim = dim
        self._sparse_dim = sparse_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> Embedding:
        tokens = _TOKEN_RE.findall(text.lower())

        dense: list[float] = [0.0] * self._dim
        counts: dict[int, int] = {}
        for token in tokens:
            dense[_hash_index(token, self._dim)] += 1.0
            index = _hash_index(token, self._sparse_dim)
            counts[index] = counts.get(index, 0) + 1

        norm = sqrt(sum(value * value for value in dense))
        if norm:
            dense = [value / norm for value in dense]

        indices = sorted(counts)
        values = [float(counts[index]) for index in indices]
        return Embedding(dense=dense, sparse=SparseEmbedding(indices=indices, values=values))

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Embed each text, preserving input order (Phase 15 batched embedding)."""
        return [self.embed(text) for text in texts]


_EMBEDDING_CACHE_VERSION = 1


def _embedding_cache_key(text: str, *, dim: int) -> str:
    """Cache key = version + dense dimension + text hash.

    The dimension is part of the key so a change in `qdrant_vector_size` (or a
    future model swap bumping the version) can never serve a stale-shaped vector.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embedding:v{_EMBEDDING_CACHE_VERSION}:{dim}:{digest}"


def _embedding_to_json(embedding: Embedding) -> dict[str, Any]:
    return {
        "dense": embedding.dense,
        "sparse": {
            "indices": embedding.sparse.indices,
            "values": embedding.sparse.values,
        },
    }


def _embedding_from_json(payload: dict[str, Any]) -> Embedding:
    sparse = payload["sparse"]
    return Embedding(
        dense=[float(value) for value in payload["dense"]],
        sparse=SparseEmbedding(
            indices=[int(index) for index in sparse["indices"]],
            values=[float(value) for value in sparse["values"]],
        ),
    )


class CachingEmbedder:
    """Wraps an `Embedder` with a best-effort vector cache (Phase 15).

    Identical texts embed identically (the deterministic adapter is a pure
    function), so identical requests — across retrieval and indexing — reuse the
    cached vector without changing correctness. Cache failures degrade to the
    wrapped embedder; `embed_batch` preserves input order and only computes the
    cache misses in that same order.
    """

    def __init__(
        self,
        embedder: Embedder,
        cache: CacheService,
        *,
        ttl_seconds: int,
        key_dim: int,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._ttl_seconds = ttl_seconds
        self._key_dim = key_dim

    def embed(self, text: str) -> Embedding:
        key = _embedding_cache_key(text, dim=self._key_dim)
        cached = self._cache.get(key)
        if cached is not None:
            return _embedding_from_json(cached)
        embedding = self._embedder.embed(text)
        self._cache.set(key, _embedding_to_json(embedding), ttl_seconds=self._ttl_seconds)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        keys = [_embedding_cache_key(text, dim=self._key_dim) for text in texts]
        results: list[Embedding | None] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []
        for index, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = _embedding_from_json(cached)
            else:
                to_compute.append((index, texts[index]))

        if to_compute:
            computed = self._embedder.embed_batch([text for _, text in to_compute])
            for (index, text), embedding in zip(to_compute, computed):
                results[index] = embedding
                self._cache.set(
                    keys[index], _embedding_to_json(embedding), ttl_seconds=self._ttl_seconds
                )

        embeddings = [result for result in results if result is not None]
        if len(embeddings) != len(texts):  # pragma: no cover - defensive
            raise RuntimeError("cached embedder produced an incomplete batch")
        return embeddings


def build_cached_embedder(
    embedder: Embedder,
    cache: CacheService,
    settings: Settings | None = None,
) -> CachingEmbedder:
    """Wrap `embedder` in the Phase 15 vector cache using application settings."""
    resolved = settings or get_settings()
    key_dim = int(getattr(embedder, "dim", resolved.qdrant_vector_size))
    return CachingEmbedder(
        embedder,
        cache,
        ttl_seconds=resolved.cache_embedding_ttl_seconds,
        key_dim=key_dim,
    )


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Return the configured embedder. `deterministic` is the default (Phase 8).

    BGE-M3 (ARCHITECTURE Phase 7) is the future production adapter; wiring the
    model itself is deliberately out of scope for the vector database phase.
    """
    resolved = settings or get_settings()
    if resolved.embedding_provider == "deterministic":
        return DeterministicEmbedder(dim=resolved.qdrant_vector_size)
    if resolved.embedding_provider == "bge_m3":
        raise NotImplementedError(
            "the BGE-M3 embedder is the embedding phase (ARCHITECTURE Phase 7); "
            "use embedding_provider=deterministic until then"
        )
    raise ValueError(f"unknown embedding_provider: {resolved.embedding_provider}")
