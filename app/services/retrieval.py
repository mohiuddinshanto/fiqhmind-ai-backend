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

Phase 15 §706-707: I/O is overlapped with asyncio — the cross-lingual dense
search on the original-language query runs concurrently with translation, and
the remaining candidate queries all run concurrently (dense + sparse within
each via `HybridSearchService.search_async`). The public `search()` stays
synchronous; it only adds the chunk cache and bridges to `_search_async`.
"""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field

import structlog
from sqlalchemy.orm import Session

from app.core.asyncio_utils import run_coroutine
from app.core.config import Settings, get_settings
from app.core.qdrant import QdrantStore
from app.services.cache import CacheService
from app.services.embedding import build_cached_embedder, get_embedder
from app.services.hybrid_search import HybridSearchService, PayloadFilter, SearchHit
from app.services.language import LanguageDetector, get_language_detector, require_language
from app.services.query_expansion import CandidateQuery, ExpansionRunner
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
        cache: CacheService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache
        if hybrid is not None:
            self._hybrid = hybrid
        else:
            embedder = get_embedder(self._settings)
            if cache is not None:
                embedder = build_cached_embedder(embedder, cache, self._settings)
            self._hybrid = HybridSearchService(
                store, embedder=embedder, k=self._settings.vector_rrf_k
            )
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
        """Run the full pipeline and return compressed, reranked evidence.

        Sync entry point (sync FastAPI endpoints, Celery tasks, tests); the
        Phase 15 §706-707 concurrency lives in `_search_async`. This method
        only adds the 15-minute chunk-level results cache (ARCHITECTURE
        §Phase 15), keyed by the full query scope so different filters/top-N
        never share an entry. Cache failures are swallowed and the pipeline
        recomputes.
        """
        cache = self._cache
        cache_key = _retrieval_cache_key(query, filters, top_n) if cache is not None else None
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                try:
                    return _retrieval_from_dict(cached)
                except Exception as exc:  # noqa: BLE001 - corrupt cache is recomputed
                    logger.warning("retrieval_cache_read_failed", key=cache_key, error=str(exc))

        result = run_coroutine(lambda: self._search_async(query, filters=filters, top_n=top_n))

        if cache is not None and cache_key is not None:
            cache.set(
                cache_key,
                _retrieval_to_dict(result),
                ttl_seconds=self._settings.cache_retrieval_ttl_seconds,
            )
        return result

    async def _search_async(
        self,
        query: str,
        *,
        filters: PayloadFilter | None,
        top_n: int | None,
    ) -> RetrievalResult:
        """Phase 15 §707: translation ∥ cross-lingual dense search, then all
        remaining candidate queries concurrently; rerank + compress."""
        prepared = self._preprocessor.prepare(query)
        language = require_language(prepared.display, self._detector)
        limit = max(int(self._settings.retrieval_candidates), 1)

        # The original-language query drives the cross-lingual dense search
        # (BGE-M3 embeds bn/en and Arabic in one space) — that and the
        # translation start together and neither blocks the other (§707).
        original_text = prepared.display
        translation_task = asyncio.create_task(
            self._translate_with_fallback(original_text, language.code)
        )
        original_task = asyncio.create_task(
            self._hybrid.search_async(original_text, limit=limit, filters=filters)
        )

        translation = await translation_task
        canonical_arabic = normalize_arabic(translation.text)
        expanded = self._expansion.expand(original=original_text, canonical_arabic=canonical_arabic)

        # The original query was already searched above; only the remaining
        # (Arabic) variants need their own searches — all at once (§706).
        remaining = self._remaining_candidates(expanded.candidates, original_text)
        candidate_tasks = [
            asyncio.create_task(
                self._hybrid.search_async(candidate.text, limit=limit, filters=filters)
            )
            for candidate in remaining
        ]

        original_hits, original_error = await self._await_search(original_task, "original")
        candidate_results = await self._await_candidates(candidate_tasks)
        if original_hits is None and not candidate_results:
            # Never swallow a total retrieval outage as empty evidence.
            if original_error is not None:
                raise original_error
            raise RuntimeError("all hybrid searches failed")

        merged = _merge_hits([original_hits, *candidate_results], limit)
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

    async def _translate_with_fallback(self, text: str, source_lang: str) -> TranslationResult:
        """Translate on a worker thread; on failure keep the original as canonical.

        A translation outage must not kill retrieval — the original-language
        dense search already ran concurrently — so the failure is logged and
        the passthrough (identity) result is used.
        """
        try:
            return await asyncio.to_thread(self._translate, text, source_lang)
        except Exception as exc:  # noqa: BLE001 - translation outage is tolerated
            logger.warning("retrieval_translation_failed", source_lang=source_lang, error=str(exc))
            return PassthroughTranslator().translate(text, source_lang=source_lang)

    async def _await_search(
        self,
        task: asyncio.Task[list[SearchHit]],
        stage: str,
    ) -> tuple[list[SearchHit] | None, Exception | None]:
        """Await one hybrid search; a failure is logged and tolerated."""
        try:
            return await task, None
        except Exception as exc:  # noqa: BLE001 - a single failed search is tolerated
            logger.warning("retrieval_search_failed", stage=stage, error=str(exc))
            return None, exc

    async def _await_candidates(
        self, tasks: list[asyncio.Task[list[SearchHit]]]
    ) -> list[list[SearchHit]]:
        """Await all candidate searches concurrently; drop and log failures."""
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes: list[list[SearchHit]] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("retrieval_candidate_search_failed", error=str(result))
            else:
                successes.append(result)
        return successes

    @staticmethod
    def _remaining_candidates(
        candidates: tuple[CandidateQuery, ...], original_text: str
    ) -> list[CandidateQuery]:
        """The candidate queries not already covered by the original search.

        Deduped by text so duplicate variants never cause duplicate embedding
        work (expansion already de-dupes internally; this is the final guard).
        """
        seen: set[str] = set()
        remaining: list[CandidateQuery] = []
        for candidate in candidates:
            if candidate.text == original_text or candidate.text in seen:
                continue
            seen.add(candidate.text)
            remaining.append(candidate)
        return remaining

    def _rerank(
        self,
        merged: list[SearchHit],
        canonical_arabic: str,
        original_query: str,
    ) -> list[_RankedHit]:
        """Step 7: score all hits against both queries; keep the max per hit.

        The max includes the raw dense cosine score (`hit.dense_score`): the
        RRF score is rank-based (bounded well below the evidence floor) and the
        deterministic Jaccard scorer misses semantically-related-but-lexically
        disjoint passages, so the dense similarity is the semantic-evidence
        signal that keeps the floor reachable until the cross-encoder lands.
        """
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
                max_score=max(a_score, o_score, hit.dense_score or 0.0),
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


def _merge_hits(hit_lists: list[list[SearchHit] | None], limit: int) -> list[SearchHit]:
    """Merge concurrent hybrid results: dedupe by chunk_id, keep the best score.

    Preserves the Phase 9 step 6 behavior (dedupe by `chunk_id` → Top-`limit`)
    for results that now arrive from several in-flight searches at once.
    """
    best: dict[str, SearchHit] = {}
    for hits in hit_lists:
        if not hits:
            continue
        for hit in hits:
            existing = best.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                best[hit.chunk_id] = hit
    ordered = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
    return ordered[:limit]


def _retrieval_cache_key(query: str, filters: PayloadFilter | None, top_n: int | None) -> str:
    """Content-scoped key for the chunk-level results cache.

    Every input that shapes the result — the query plus the full metadata filter
    scope and top-N — is part of the key, so two requests never share an entry
    unless every input matches (no cross-user/cross-scope contamination).
    """
    scope = filters or PayloadFilter()
    payload = json.dumps(
        [
            query,
            scope.book_id,
            scope.volume,
            scope.region,
            scope.verified,
            top_n,
        ],
        ensure_ascii=False,
    )
    return f"chunk:v1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _retrieval_to_dict(result: RetrievalResult) -> dict:
    """JSON-safe serialization of a `RetrievalResult` for the cache layer."""
    return {
        "query": result.query,
        "canonical_arabic_query": result.canonical_arabic_query,
        "language": result.language,
        "translated": result.translated,
        "candidates": result.candidates,
        "evidence_sufficient": result.evidence_sufficient,
        "chunks": [asdict(chunk) for chunk in result.chunks],
    }


def _retrieval_from_dict(payload: dict) -> RetrievalResult:
    """Rehydrate a `RetrievalResult` previously stored by `_retrieval_to_dict`."""
    return RetrievalResult(
        query=payload["query"],
        canonical_arabic_query=payload["canonical_arabic_query"],
        language=payload["language"],
        translated=payload["translated"],
        candidates=list(payload.get("candidates", [])),
        evidence_sufficient=bool(payload.get("evidence_sufficient", False)),
        chunks=[RetrievedChunk(**chunk) for chunk in payload.get("chunks", [])],
    )
