"""Tests for the Phase 8 embedding adapter (deterministic feature-hashing)."""

import pytest

from app.core.config import Settings
from app.services.embedding import (
    DEFAULT_DENSE_DIM,
    DEFAULT_SPARSE_DIM,
    DeterministicEmbedder,
    Embedding,
    get_embedder,
)


def test_embed_returns_dense_and_sparse_vectors() -> None:
    result = DeterministicEmbedder().embed("zakat al-fitr is wajib")

    assert isinstance(result, Embedding)
    assert len(result.dense) == DEFAULT_DENSE_DIM
    assert len(result.sparse.indices) == len(result.sparse.values)
    assert result.sparse.indices == sorted(result.sparse.indices)
    assert all(index < DEFAULT_SPARSE_DIM for index in result.sparse.indices)
    assert all(value > 0 for value in result.sparse.values)


def test_embed_is_deterministic() -> None:
    embedder = DeterministicEmbedder()

    first = embedder.embed("washing before every prayer is sunnah")
    second = embedder.embed("washing before every prayer is sunnah")

    assert first.dense == second.dense
    assert first.sparse == second.sparse


def test_embed_distinguishes_different_texts() -> None:
    embedder = DeterministicEmbedder()

    a = embedder.embed("zakat on savings is due")
    b = embedder.embed("fasting the month of ramadan")

    assert a.dense != b.dense
    assert a.sparse.indices != b.sparse.indices


def test_embed_dense_is_l2_normalized() -> None:
    result = DeterministicEmbedder().embed("a line of fiqh text to normalize")

    norm = sum(value * value for value in result.dense) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_embed_deduplicates_sparse_tokens_with_counts() -> None:
    result = DeterministicEmbedder().embed("zakat zakat zakat")

    assert len(result.sparse.indices) == 1
    assert result.sparse.values == [3.0]


def test_embed_empty_text() -> None:
    result = DeterministicEmbedder().embed("")

    assert result.dense == [0.0] * DEFAULT_DENSE_DIM
    assert result.sparse.indices == []
    assert result.sparse.values == []


def test_embed_respects_custom_dimensions() -> None:
    result = DeterministicEmbedder(dim=64, sparse_dim=1024).embed("custom dims")

    assert len(result.dense) == 64
    assert all(index < 1024 for index in result.sparse.indices)


def test_get_embedder_defaults_to_deterministic() -> None:
    embedder = get_embedder()

    assert isinstance(embedder, DeterministicEmbedder)
    assert embedder.dim == DEFAULT_DENSE_DIM


def test_get_embedder_uses_settings_dim() -> None:
    settings = Settings(qdrant_vector_size=256)
    embedder = get_embedder(settings)

    assert isinstance(embedder, DeterministicEmbedder)
    assert embedder.dim == 256


def test_get_embedder_rejects_bge_m3_for_now() -> None:
    settings = Settings(embedding_provider="bge_m3")

    with pytest.raises(NotImplementedError):
        get_embedder(settings)


def test_get_embedder_rejects_unknown_provider() -> None:
    settings = Settings(embedding_provider="gpt")

    with pytest.raises(ValueError):
        get_embedder(settings)
