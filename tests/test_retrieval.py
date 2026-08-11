"""Tests for the Phase 9 retrieval pipeline runner (steps 1-8)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.exceptions import QueryValidationError
from app.db.base import Base
from app.services.hybrid_search import PayloadFilter, SearchHit
from app.services.retrieval import RetrievalResult, RetrievalRunner


class FakeHybrid:
    """Records search calls and returns a fixed hit list."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, PayloadFilter | None]] = []

    def search(self, query: str, *, limit: int, filters: PayloadFilter | None = None):
        self.calls.append((query, limit, filters))
        return self.hits

    async def search_async(self, query: str, *, limit: int, filters: PayloadFilter | None = None):
        self.calls.append((query, limit, filters))
        return self.hits


class StubReranker:
    """Returns a fixed score list, recording the queries it saw."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return list(self.scores)


def _payload(
    chunk_id: str, *, text: str = "الماء طهور لا ينجسه شيء", topic: str = "طهارة", **overrides
) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "text": text,
        "book_name": "Al-Hidayah",
        "book_id": "book-1",
        "volume": "1",
        "printed_page_start": 5,
        "printed_page_end": 6,
        "kitab": "الطهارة",
        "bab": "باب المياه",
        "fasl": None,
        "topic": topic,
        "region": "main",
        "lang": "ar",
        "verified": True,
    }
    payload.update(overrides)
    return payload


def _hit(chunk_id: str, *, score: float, **payload_overrides) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        payload=_payload(chunk_id, **payload_overrides),
    )


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


def _runner(
    session: Session,
    hybrid: FakeHybrid,
    reranker: StubReranker,
    **settings_overrides,
) -> RetrievalRunner:
    settings = Settings(**settings_overrides)
    return RetrievalRunner(
        session,
        object(),  # store is unused when hybrid is injected
        settings=settings,
        hybrid=hybrid,
        reranker=reranker,
    )


def test_search_returns_compressed_evidence(session: Session) -> None:
    hybrid = FakeHybrid(
        [_hit("c1", score=0.02), _hit("c2", score=0.01, topic="وضوء", text="الوضوء من سنن الفطرة")]
    )
    runner = _runner(session, hybrid, StubReranker([0.9, 0.4]))

    result = runner.search("ما حكم الوضوء")

    assert isinstance(result, RetrievalResult)
    assert result.query == "ما حكم الوضوء"
    assert result.canonical_arabic_query == "ما حكم الوضوء"
    assert result.language == "ar"
    assert result.translated is False
    assert result.evidence_sufficient is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2"]
    assert result.chunks[0].rerank_score == pytest.approx(0.9)
    assert result.chunks[0].verified is True


def test_search_dedupes_candidates_by_chunk_id(session: Session) -> None:
    hits = [
        _hit("c1", score=0.03),
        _hit("c1", score=0.02),
        _hit("c2", score=0.01, topic="وضوء"),
    ]
    runner = _runner(session, FakeHybrid(hits), StubReranker([0.8, 0.6]))

    result = runner.search("الوضوء")

    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2"]


def test_rerank_scores_both_queries_and_takes_max(session: Session) -> None:
    hits = [_hit("c1", score=0.02)]
    reranker = StubReranker([0.7])
    runner = _runner(session, FakeHybrid(hits), reranker)

    result = runner.search("ما حكم الوضوء")

    queries = [query for query, _ in reranker.calls]
    assert len(queries) == 2
    assert queries[0] == "ما حكم الوضوء"  # canonical Arabic
    assert queries[1] == "ما حكم الوضوء"  # original-language query
    assert result.chunks[0].rerank_score == pytest.approx(0.7)


def test_evidence_floor_drops_below_threshold(session: Session) -> None:
    hits = [_hit("c1", score=0.02), _hit("c2", score=0.01)]
    runner = _runner(
        session, FakeHybrid(hits), StubReranker([0.9, 0.01]), retrieval_evidence_floor=0.05
    )

    result = runner.search("الوضوء")

    assert result.evidence_sufficient is True
    assert [chunk.chunk_id for chunk in result.chunks] == ["c1"]


def test_all_below_floor_is_insufficient_evidence(session: Session) -> None:
    hits = [_hit("c1", score=0.02), _hit("c2", score=0.01)]
    runner = _runner(
        session, FakeHybrid(hits), StubReranker([0.01, 0.01]), retrieval_evidence_floor=0.05
    )

    result = runner.search("الوضوء")

    assert result.evidence_sufficient is False
    assert result.chunks == []


def test_near_duplicate_passages_are_deduplicated(session: Session) -> None:
    hits = [
        _hit("c1", score=0.02, topic="طهارة"),
        _hit("c2", score=0.01, topic="طهارة", printed_page_end=7),
    ]
    runner = _runner(session, FakeHybrid(hits), StubReranker([0.9, 0.8]))

    result = runner.search("الوضوء")

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c1"


def test_different_topics_on_same_page_are_kept(session: Session) -> None:
    hits = [
        _hit("c1", score=0.02, topic="طهارة", text="الماء طهور"),
        _hit("c2", score=0.01, topic="صلاة", text="الصلاة نور"),
    ]
    runner = _runner(session, FakeHybrid(hits), StubReranker([0.9, 0.8]))

    result = runner.search("الوضوء")

    assert len(result.chunks) == 2


def test_metadata_filters_are_forwarded(session: Session) -> None:
    hybrid = FakeHybrid([_hit("c1", score=0.02)])
    runner = _runner(session, hybrid, StubReranker([0.9]))

    filters = PayloadFilter(book_id="book-1", verified=True)
    runner.search("الوضوء", filters=filters)

    for query, limit, forwarded in hybrid.calls:
        assert forwarded == filters
        assert limit == 40


def test_translated_query_reports_flag_and_keeps_original(session: Session) -> None:
    hybrid = FakeHybrid([_hit("c1", score=0.02)])
    runner = _runner(session, hybrid, StubReranker([0.9]))

    result = runner.search("ওযুর নিয়ম কি")

    assert result.language == "bn"
    assert result.translated is False  # passthrough default
    assert result.query == "ওযুর নিয়ম কি"


def test_reranking_disabled_uses_rrf_scores(session: Session) -> None:
    hits = [_hit("c1", score=0.02), _hit("c2", score=0.01, topic="وضوء")]
    runner = _runner(
        session,
        FakeHybrid(hits),
        StubReranker([0.0, 0.0]),
        retrieval_reranking_enabled=False,
    )

    result = runner.search("الوضوء")

    assert [chunk.chunk_id for chunk in result.chunks] == ["c1", "c2"]
    assert result.chunks[0].rerank_score == pytest.approx(0.02)


def test_top_n_bounds_the_output(session: Session) -> None:
    hits = [_hit(f"c{i}", score=0.01, topic=f"t{i}") for i in range(5)]
    runner = _runner(session, FakeHybrid(hits), StubReranker([0.9] * 5))

    result = runner.search("الوضوء", top_n=2)

    assert len(result.chunks) == 2


def test_rejected_query_raises_validation_error(session: Session) -> None:
    runner = _runner(session, FakeHybrid([]), StubReranker([]))

    with pytest.raises(QueryValidationError):
        runner.search("   ")
