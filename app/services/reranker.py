"""Reranking (Phase 9 — Retrieval Pipeline).

ARCHITECTURE §Phase 9 step 7: score every candidate with the cross-encoder
BGE-reranker-v2-m3 — an *external model* whose weights are downloaded. The
pipeline therefore depends on a `Reranker` interface, and `DefaultReranker` is
the dependency-free adapter: a deterministic Jaccard token-overlap scorer in
[0, 1] that exercises the rerank path with no model download. The production
cross-encoder plugs into the same interface when the weights phase lands.
"""

import re
from typing import Protocol

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"\S+")


class Reranker(Protocol):
    """Port implemented by every reranking adapter (default, BGE…)."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class DefaultReranker:
    """Deterministic lexical-overlap scorer (dependency-free default).

    `score = |Q ∩ T| / |Q ∪ T|` (Jaccard) over lowercase whitespace tokens,
    so identical wording scores 1.0, disjoint wording 0.0. It is intentionally
    not a cross-encoder — it only exercises the two-query rerank path.
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        scores: list[float] = []
        for text in texts:
            if not query_tokens:
                scores.append(0.0)
                continue
            text_tokens = set(_TOKEN_RE.findall(text.lower()))
            if not text_tokens:
                scores.append(0.0)
                continue
            scores.append(len(query_tokens & text_tokens) / len(query_tokens | text_tokens))
        return scores


class BgeRerankerV2M3:
    """Production cross-encoder slot (ARCHITECTURE step 7).

    The BGE-reranker-v2-m3 weights are an external download, out of scope for
    this phase; constructing the adapter raises so misconfiguration is loud.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        raise NotImplementedError(
            "BGE-reranker-v2-m3 weights are an external download; "
            f"reranking on {model_name} is not wired yet (use reranker_provider=default)"
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError("BGE-reranker-v2-m3 is not wired in Phase 9")


def get_reranker(settings: Settings | None = None) -> Reranker:
    """Return the configured reranker.

    `default` is the dependency-free adapter (Phase 9); the BGE cross-encoder
    is the future production adapter, deliberately not wired in this phase.
    """
    resolved = settings or get_settings()
    if resolved.reranker_provider == "default":
        return DefaultReranker()
    if resolved.reranker_provider == "bge_reranker_v2_m3":
        return BgeRerankerV2M3()
    raise ValueError(f"unknown reranker_provider: {resolved.reranker_provider}")
