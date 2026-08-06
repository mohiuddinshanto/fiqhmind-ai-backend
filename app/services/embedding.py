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
from typing import Protocol

import structlog

from app.core.config import Settings, get_settings

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
