"""Tests for the Phase 15 chunk-level results cache (15 min TTL)."""

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


class FakeHybrid:
    """Returns a fixed hit list and counts every search call."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls = 0

    def search(self, query: str, *, limit: int, filters: PayloadFilter | None = None):
        self.calls += 1
        return self.hits

    async def search_async(self, query: str, *, limit: int, filters: PayloadFilter | None = None):
        self.calls += 1
        return self.hits


class StubReranker:
    """Returns a fixed score list and counts every scoring call."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls += 1
        return list(self.scores)


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


def _hit(chunk_id: str, **overrides) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, score=1.0, payload=_payload(chunk_id, **overrides))


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


@pytest.fixture()
def cache() -> CacheService:
    server = fakeredis.FakeServer()
    return CacheService(fakeredis.FakeStrictRedis(server=server))


def _runner(session: Session, hybrid: FakeHybrid, cache: CacheService | None) -> RetrievalRunner:
    return RetrievalRunner(
        session,
        object(),  # store is unused when hybrid is injected
        settings=Settings(retrieval_evidence_floor=0.0),
        hybrid=hybrid,
        reranker=StubReranker([0.9, 0.4]),
        cache=cache,
    )


def test_second_identical_search_hits_cache(session: Session, cache: CacheService) -> None:
    hybrid = FakeHybrid([_hit("c1"), _hit("c2")])
    runner = _runner(session, hybrid, cache)

    first = runner.search("ما حكم الوضوء")
    second = runner.search("ما حكم الوضوء")

    assert first == second
    assert hybrid.calls == 1  # second call is served from the cache


def test_different_scope_produces_distinct_cache_entries(
    session: Session, cache: CacheService
) -> None:
    hybrid = FakeHybrid([_hit("c1")])
    runner = _runner(session, hybrid, cache)

    runner.search("الوضوء")
    runner.search("الوضوء", filters=PayloadFilter(book_id="book-1"))
    runner.search("الوضوء", filters=PayloadFilter(book_id="book-1"), top_n=2)

    # Different book scope / top-N never reuse another entry.
    assert hybrid.calls == 3


def test_runner_without_cache_recomputes_every_time(session: Session) -> None:
    hybrid = FakeHybrid([_hit("c1")])
    runner = _runner(session, hybrid, None)

    runner.search("الوضوء")
    runner.search("الوضوء")

    assert hybrid.calls == 2
