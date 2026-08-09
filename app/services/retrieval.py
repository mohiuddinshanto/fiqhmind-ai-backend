"""Retrieval pipeline (Phase 9 — Retrieval Pipeline).

Wires ARCHITECTURE §Phase 9 steps 1–8 into one deterministic runner:
  1. preprocess (QueryPreprocessor)            → two copies (display + canonical)
  2. language detection (LanguageDetector)
  3. Arabic translation (Translator)           → canonical Arabic query (store both)
  4. query expansion (ExpansionRunner)         → bounded candidate queries
  5. hybrid search per candidate (HybridSearchService, dense+sparse RRF fusion,
     payload filters applied at query time)
  6. merge + dedupe by chunk_id → Top-40
  7. rerank (Reranker): score every hit against the canonical Arabic query AND
     the original-language query; take the max (handles translation drift)
  8. context compression: drop scores below the evidence floor + dedupe
     near-identical passages by a (book, topic, text-prefix) signature

Evidence synthesis / LLM generation belong to Phase 10; this runner stops after
compression and reports `evidence_sufficient`.
"""

import hashlib
from dataclasses import dataclass, field

import structlog
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.qdrant import QdrantStore
from app.services.hybrid_search import HybridSearchService, PayloadFilter, SearchHit
from app.services.language import LanguageDetector, get_language_detector, require_language
from app.services.query_expansion import ExpandedQuery, ExpansionRunner
from app.services.query_preprocessing import QueryPreprocessor, normalize_arabic
from app.services.reranker import Reranker, get_reranker
from app.services.translation import (
    PassthroughTranslator,
    TranslationResult,
    Translator,
    get_translator,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """One evidence chunk after rerank + compression (Phase 9 steps 7–8)."""

    chunk_id: str
    text: str
    book_name: str | None = None
    volume: str | None = None
    printed_page_start: int | None = None
    printed_page_end: int | None = None
    kitab: str | None = None
    bab: str | None = None
    fasl: str | None = None
    topic: str | None = None
    region: str | None = None
    lang: str | None = None
    verified: bool = False
    rerank_score: float = 0.0
    arabic_score: float | None = None
    original_score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """The full Phase 9 output: context, provenance and evidence verdict."""

    query: str
    canonical_arabic_query: str
    language: str
    translated: bool
    candidates: list[str] = field(default_factory=list)
    evidence_sufficient: bool = False
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass(frozen=True)
class _RankedHit:
    max_score: float
    arabic_score: float | None
    original_score: float | None
    hit: SearchHit


class RetrievalRunner:
    """Deterministic Phase 9 pipeline: preprocess → retrieve → rerank → compress."""

    def __init__(
        self,
        session: Session | None,
        store: QdrantStore,
        *,
        settings: Settings | None = None,
        hybrid: HybridSearchService | None = None,
        preprocessor: QueryPreprocessor | None = None,
        detector: LanguageDetector | None = None,
        translator: Translator | None = None,
        expansion: ExpansionRunner | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._hybrid = hybrid or HybridSearchService(store)
        self._preprocessor = preprocessor or QueryPreprocessor(self._settings)
        self._detector = detector or get_language_detector(self._settings)
        self._translator = translator or get_translator(self._settings)
        self._expansion = expansion or ExpansionRunner(session, self._settings)
        self._reranker = reranker or get_reranker(self._settings)

    def search(
        self,
        query: str,
        *,
        filters: PayloadFilter | None = None,
        top_n: int | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline and return compressed, reranked evidence."""
        prepared = self._preprocessor.prepare(query)
        language = require_language(prepared.display, self._detector)

        translation = self._translate(prepared.display, language.code)
        canonical_arabic = normalize_arabic(translation.text)
        expanded = self._expansion.expand(
            original=prepared.display,
            canonical_arabic=canonical_arabic,
        )

        merged = self._hybrid_merge(expanded, filters)
        ranked = self._rerank(merged, canonical_arabic, prepared.canonical)
        chunks = self._compress(ranked, top_n)

        return RetrievalResult(
            query=prepared.display,
            canonical_arabic_query=canonical_arabic,
            language=language.code,
            translated=translation.translated,
            candidates=[candidate.text for candidate in expanded.candidates],
            evidence_sufficient=bool(chunks),
            chunks=chunks,
        )

    def _translate(self, text: str, source_lang: str) -> TranslationResult:
        if source_lang == "ar":
            return PassthroughTranslator().translate(text, source_lang="ar")
        return self._translator.translate(text, source_lang=source_lang)

    def _hybrid_merge(
        self,
        expanded: ExpandedQuery,
        filters: PayloadFilter | None,
    ) -> list[SearchHit]:
        """Step 5 + 6: hybrid-search every candidate, dedupe by chunk_id → Top-40."""
        limit = max(int(self._settings.retrieval_candidates), 1)
        best: dict[str, SearchHit] = {}
        for candidate in expanded.candidates:
            hits = self._hybrid.search(candidate.text, limit=limit, filters=filters)
            for hit in hits:
                existing = best.get(hit.chunk_id)
                if existing is None or hit.score > existing.score:
                    best[hit.chunk_id] = hit
        ordered = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return ordered[:limit]

    def _rerank(
        self,
        merged: list[SearchHit],
        canonical_arabic: str,
        original_query: str,
    ) -> list[_RankedHit]:
        """Step 7: score all hits against both queries; keep the max per hit."""
        if not merged:
            return []
        texts = [hit.payload.get("text", "") or "" for hit in merged]
        if not self._settings.retrieval_reranking_enabled:
            return [
                _RankedHit(max_score=hit.score, arabic_score=None, original_score=None, hit=hit)
                for hit in merged
            ]
        arabic_scores = self._reranker.score(canonical_arabic, texts)
        original_scores = self._reranker.score(original_query, texts)
        ranked = [
            _RankedHit(
                max_score=max(a_score, o_score),
                arabic_score=a_score,
                original_score=o_score,
                hit=hit,
            )
            for hit, a_score, o_score in zip(merged, arabic_scores, original_scores)
        ]
        return sorted(ranked, key=lambda item: item.max_score, reverse=True)

    def _compress(
        self,
        ranked: list[_RankedHit],
        top_n: int | None,
    ) -> list[RetrievedChunk]:
        """Step 8: evidence floor + near-identical dedupe, bounded to Top-6..8."""
        top_n = top_n or self._settings.retrieval_top_n
        floor = self._settings.retrieval_evidence_floor
        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        for item in ranked:
            if self._settings.retrieval_reranking_enabled and item.max_score < floor:
                break  # ranked descending — everything after is below the floor
            signature = self._signature(item.hit)
            if signature in seen:
                continue
            seen.add(signature)
            payload = item.hit.payload
            chunks.append(
                RetrievedChunk(
                    chunk_id=item.hit.chunk_id,
                    text=payload.get("text", "") or "",
                    book_name=payload.get("book_name"),
                    volume=payload.get("volume"),
                    printed_page_start=payload.get("printed_page_start"),
                    printed_page_end=payload.get("printed_page_end"),
                    kitab=payload.get("kitab"),
                    bab=payload.get("bab"),
                    fasl=payload.get("fasl"),
                    topic=payload.get("topic"),
                    region=payload.get("region"),
                    lang=payload.get("lang"),
                    verified=bool(payload.get("verified", False)),
                    rerank_score=item.max_score,
                    arabic_score=item.arabic_score,
                    original_score=item.original_score,
                )
            )
            if len(chunks) >= top_n:
                break
        return chunks

    @staticmethod
    def _signature(hit: SearchHit) -> str:
        """Near-duplicate signature: book + topic + normalized text prefix.

        ARCHITECTURE step 8 "dedupe near-identical passages (title/topic hash)" —
        the same (book, topic) with the same text prefix is treated as one
        passage; different pages on the same topic are kept.
        """
        book = hit.payload.get("book_id") or hit.payload.get("book_name") or ""
        topic = hit.payload.get("topic") or ""
        text = normalize_arabic(hit.payload.get("text", "") or "")[:200]
        return hashlib.sha256(f"{book}:{topic}:{text}".encode()).hexdigest()
