"""Phase 15 §706-707 concurrency tests for the retrieval pipeline.

The public services stay synchronous, so overlap is verified with threading
barriers: each blocking fake signals the moment it entered its method and then
waits for a release. If two operations are concurrent, BOTH signals are set
while neither has returned. No arbitrary sleeps are used.
"""

import asyncio
import threading

import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.base import Base
from app.services.cache import CacheService
from app.services.hybrid_search import PayloadFilter, SearchHit
from app.services.retrieval import RetrievalRunner
from app.services.translation import TranslationResult

# English query -> translation task actually runs (source_lang != "ar").
ORIGINAL = "washing with water"
# Translated Arabic that triggers two lexicon variants (ماء→طهور, وجوب→ثبوت),
# so there are multiple candidate searches that can overlap.
CANONICAL = "الماء والوجوب"


def _payload(chunk_id: str, **overrides) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "text": "الماء طهور لا ينجسه شيء",
        "book_name": "Al-Hidayah",
        "book_id": "book-1",
        "volume": "1",
        "printed_page_start": 5,
        "printed_page_end": 6,
        "kitab": "الطهارة",
        "bab": "باب المياه",
        "fasl": None,
        "topic": "طهارة",
        "region": "main",
        "lang": "ar",
        "verified": True,
    }
    payload.update(overrides)
    return payload


def _hit(chunk_id: str, *, score: float = 1.0, **payload_overrides) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, score=score, payload=_payload(chunk_id, **payload_overrides)
    )

class BlockingTranslator:
    """Signals when translation starts, then blocks until `release`."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def translate(self, text, *, source_lang, target_lang="ar") -> TranslationResult:
        self.started.set()
        self.release.wait(timeout=5)
        return TranslationResult(
            text=CANONICAL,
            source_lang=source_lang,
            target_lang=target_lang,
            translated=True,
            confidence=1.0,
        )


class FixedTranslator:
    """Returns the fixed Arabic canonical without blocking."""

    def translate(self, text, *, source_lang, target_lang="ar") -> TranslationResult:
        return TranslationResult(
            text=CANONICAL,
            source_lang=source_lang,
            target_lang=target_lang,
            translated=True,
            confidence=1.0,
        )


class BlockingHybrid:
    """Blocks every search; tracks original + candidate searches in flight."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.original_started = threading.Event()
        self.candidate_started = threading.Event()
        self.two_candidates_started = threading.Event()
        self.release = threading.Event()
        self._candidates_in_flight = 0
        self._lock = threading.Lock()

    async def search_async(self, query: str, *, limit: int, filters: PayloadFilter | None = None):
        with self._lock:
            if query == ORIGINAL:
                self.original_started.set()
            else:
                self._candidates_in_flight += 1
                self.candidate_started.set()
                if self._candidates_in_flight >= 2:
                    self.two_candidates_started.set()
        await asyncio.to_thread(self.release.wait, 5)
        return self.hits


class StubReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return list(self.scores)


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


@pytest.fixture()
def cache() -> CacheService:
    server = fakeredis.FakeServer()
    return CacheService(fakeredis.FakeStrictRedis(server=server))


def _runner(
    session: Session,
    hybrid,
    reranker,
    translator,
    *,
    cache: CacheService | None = None,
    **settings_overrides,
) -> RetrievalRunner:
    settings = Settings(retrieval_evidence_floor=0.0, **settings_overrides)
    return RetrievalRunner(
        session,
        object(),  # store is unused when hybrid is injected
        settings=settings,
        hybrid=hybrid,
        reranker=reranker,
        translator=translator,
        cache=cache,
    )


def _start_search(runner: RetrievalRunner, results: dict) -> threading.Thread:
    def _run() -> None:
        results["result"] = runner.search(ORIGINAL)

    thread = threading.Thread(target=_run)
    thread.start()
    return thread


def test_translation_and_initial_retrieval_overlap(session: Session) -> None:
    """§707: translation runs concurrently with the cross-lingual dense search."""
    hybrid = BlockingHybrid([_hit("c1")])
    translator = BlockingTranslator()
    runner = _runner(session, hybrid, StubReranker([0.9]), translator)
    results: dict = {}

    thread = _start_search(runner, results)

    assert translator.started.wait(timeout=5)  # translation in flight
    assert hybrid.original_started.wait(timeout=5)  # original dense search in flight
    # Both entered before either returned -> they overlapped; now unblock both.

    hybrid.release.set()
    translator.release.set()
    thread.join(timeout=5)

    result = results["result"]
    assert not thread.is_alive()
    assert result.translated is True
    assert result.language == "en"
    assert result.query == ORIGINAL
    assert result.canonical_arabic_query == CANONICAL
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1"]


def test_candidate_searches_run_concurrently(session: Session) -> None:
    """§706: multiple query variants are searched at the same time."""
    hybrid = BlockingHybrid([_hit("c1")])
    translator = BlockingTranslator()
    runner = _runner(session, hybrid, StubReranker([0.9]), translator)
    results: dict = {}

    thread = _start_search(runner, results)

    assert translator.started.wait(timeout=5)
    translator.release.set()  # expansion proceeds; candidates start and block
    assert hybrid.candidate_started.wait(timeout=5)
    assert hybrid.two_candidates_started.wait(timeout=5)  # 2+ in flight together

    hybrid.release.set()
    thread.join(timeout=5)

    result = results["result"]
    assert not thread.is_alive()
    assert len(result.candidates) >= 3  # original + canonical + 2 synonyms
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1"]


def test_concurrent_merge_keeps_best_score_per_chunk(session: Session) -> None:
    """Deterministic fusion: concurrent searches keep the highest score per id."""

    class ScoringHybrid:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def search_async(
            self, query: str, *, limit: int, filters: PayloadFilter | None = None
        ):
            self.calls.append(query)
            return [
                _hit("c1", score=0.5 if query == ORIGINAL else 0.9),
                _hit("c2", score=0.2, topic="صلاة", text="الصلاة نور"),
            ]

    hybrid = ScoringHybrid()
    runner = _runner(
        session,
        hybrid,
        StubReranker([0.0, 0.0]),
        FixedTranslator(),
        retrieval_reranking_enabled=False,
    )

    result = runner.search(ORIGINAL)

    assert hybrid.calls[0] == ORIGINAL  # original-language search launched first
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2"]
    assert result.chunks[0].rerank_score == pytest.approx(0.9)  # best score kept


def test_m1_cache_hits_under_async_orchestration(
    session: Session, cache: CacheService
) -> None:
    """The Phase 15 chunk cache still serves repeats through the async path."""

    class CountingHybrid:
        def __init__(self) -> None:
            self.calls = 0

        async def search_async(
            self, query: str, *, limit: int, filters: PayloadFilter | None = None
        ):
            self.calls += 1
            return [_hit("c1")]

    hybrid = CountingHybrid()
    runner = _runner(session, hybrid, StubReranker([0.9]), FixedTranslator(), cache=cache)

    first = runner.search(ORIGINAL)
    calls_after_first = hybrid.calls
    second = runner.search(ORIGINAL)

    assert first == second
    assert calls_after_first >= 1  # first request ran the pipeline
    assert hybrid.calls == calls_after_first  # the second never reached the pipeline


def test_one_candidate_failure_does_not_abort_retrieval(session: Session) -> None:
    """A single failed candidate is logged and skipped, not silently swallowed."""

    class FlakyHybrid:
        def __init__(self) -> None:
            self.failed_once = False

        async def search_async(
            self, query: str, *, limit: int, filters: PayloadFilter | None = None
        ):
            if query != ORIGINAL and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("candidate search failed")
            return [_hit("c1"), _hit("c2", topic="صلاة", text="الصلاة نور")]

    runner = _runner(session, FlakyHybrid(), StubReranker([0.9, 0.9]), FixedTranslator())

    result = runner.search(ORIGINAL)

    assert result.evidence_sufficient is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2"]
    assert result.canonical_arabic_query == CANONICAL


def test_all_searches_failed_raises(session: Session) -> None:
    """Total retrieval failure surfaces as an error instead of empty evidence."""

    class DeadHybrid:
        async def search_async(
            self, query: str, *, limit: int, filters: PayloadFilter | None = None
        ):
            raise RuntimeError("qdrant unavailable")

    runner = _runner(session, DeadHybrid(), StubReranker([]), FixedTranslator())

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        runner.search(ORIGINAL)


def test_translation_failure_falls_back_to_original_query(session: Session) -> None:
    """A translation outage must not kill retrieval: original is kept as canonical."""

    class BrokenTranslator:
        def translate(self, text, *, source_lang, target_lang="ar") -> TranslationResult:
            raise RuntimeError("translation API down")

    class SimpleHybrid:
        async def search_async(
            self, query: str, *, limit: int, filters: PayloadFilter | None = None
        ):
            return [_hit("c1")]

    runner = _runner(session, SimpleHybrid(), StubReranker([0.9]), BrokenTranslator())

    result = runner.search(ORIGINAL)

    assert result.translated is False
    assert result.canonical_arabic_query == ORIGINAL
    assert result.evidence_sufficient is True
