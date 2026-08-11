"""Tests for the Phase 10 chat SSE endpoint (POST /api/v1/chat)."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import deps
from app.api.v1.endpoints import chat as chat_endpoints
from app.core.config import Settings
from app.db.base import Base
from app.db.models import ChatHistory
from app.main import app
from app.services.hybrid_search import PayloadFilter
from app.services.retrieval import RetrievalResult, RetrievedChunk


class FakeStore:
    def search_dense(self, vector, *, limit, filter_=None):
        return [
            SimpleNamespace(
                id="c1",
                score=1.0,
                payload={
                    "chunk_id": "c1",
                    "text": "الماء طهور لا ينجسه شيء",
                    "book_name": "Al-Hidayah",
                    "volume": "1",
                    "printed_page_start": 5,
                    "printed_page_end": 6,
                    "topic": "طهارة",
                    "region": "main",
                    "lang": "ar",
                    "verified": True,
                },
            ),
            SimpleNamespace(
                id="c2",
                score=1.0,
                payload={
                    "chunk_id": "c2",
                    "text": "الوضوء شرط الصلاة",
                    "book_name": "Al-Hidayah",
                    "volume": "1",
                    "printed_page_start": 9,
                    "printed_page_end": 10,
                    "topic": "طهارة",
                    "region": "main",
                    "lang": "ar",
                    "verified": True,
                },
            ),
        ]

    def search_sparse(self, vector, *, limit, filter_=None):
        return []


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
def client(session: Session, monkeypatch) -> TestClient:
    from app.core.config import get_settings
    get_settings.cache_clear()
    
    mock_settings = Settings(retrieval_evidence_floor=0.0, retrieval_reranking_enabled=False)
    monkeypatch.setattr("app.core.config.get_settings", lambda: mock_settings)
    monkeypatch.setattr("app.services.retrieval.get_settings", lambda: mock_settings)
    monkeypatch.setattr("app.services.generation.service.get_settings", lambda: mock_settings)
    monkeypatch.setattr("app.services.generation.synthesis.get_settings", lambda: mock_settings)

    def override_db():
        yield session

    def override_store():
        return FakeStore()

    app.dependency_overrides[deps.get_db] = override_db
    app.dependency_overrides[deps.get_store_dep] = override_store
    yield TestClient(app)
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event is not None:
            events.append((event, data))
    return events


def test_chat_streams_typed_events_in_order(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [event for event, _ in events]
    assert "error" not in names
    assert names[0] == "meta"
    assert names[1] == "sources"
    assert names[-2] == "confidence"
    assert names[-1] == "done"
    # delta → quote → citation happen strictly in that order before confidence
    assert set(names[2:-2]) == {"delta", "quote", "citation"}
    assert names.index("quote") > names.index("delta")
    assert names.index("citation") > names.index("quote")
    assert names.index("confidence") > names.index("citation")

    _, meta = events[0]
    assert meta["query"] == "ما حكم الوضوء"
    assert meta["answer_language"] == "bn"

    _, sources = events[1]
    assert sources["evidence_sufficient"] is True
    assert sources["count"] == 2
    assert sources["chunks"][0]["book_name"] == "Al-Hidayah"

    _, done = events[-1]
    assert done["refusal"] is None
    assert done["answer_language"] == "bn"


def test_chat_delta_events_reassemble_explanation(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    events = _parse_sse(response.text)

    deltas = [data["text"] for name, data in events if name == "delta"]
    assert deltas
    explanation = "".join(deltas).rstrip()
    assert "الماء طهور" in explanation

    quotes = [data for name, data in events if name == "quote"]
    assert [q["text"] for q in quotes] == ["الماء طهور لا ينجسه شيء", "الوضوء شرط الصلاة"]

    citations = [data for name, data in events if name == "citation"]
    assert citations[0]["page"] == "5"
    assert citations[0]["book"] == "Al-Hidayah"

    confidence = next(data for name, data in events if name == "confidence")
    assert confidence["level"] == "low"
    assert confidence["source_agreement"] == "consensus"


def test_chat_respects_answer_language(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "answer_language": "en"})
    events = _parse_sse(response.text)
    done = next(data for name, data in events if name == "done")
    meta = next(data for name, data in events if name == "meta")
    assert done["answer_language"] == "en"
    assert meta["answer_language"] == "en"


def test_chat_insufficient_evidence_emits_refusal(client: TestClient, monkeypatch) -> None:
    class EmptyRunner:
        def __init__(self, session, store) -> None:
            pass

        def search(self, query, *, filters=None, top_n=None) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                canonical_arabic_query="ما حكم الوضوء",
                language="bn",
                translated=True,
                evidence_sufficient=False,
                chunks=[],
            )

    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", EmptyRunner)
    response = client.post("/api/v1/chat", json={"query": "অজানা প্রশ্ন"})

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "error" not in names
    done = next(data for name, data in events if name == "done")
    assert done["refusal"]["reason"] == "insufficient_evidence"
    assert done["refusal"]["closest_evidence"] == []
    # refusal path still streams a meta and a confidence event
    assert "meta" in names
    assert "confidence" in names
    assert "citation" not in names


def test_chat_forwards_filters_and_top_n(client: TestClient, monkeypatch) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, session, store) -> None:
            pass

        def search(self, query, *, filters: PayloadFilter | None = None, top_n=None):
            captured["query"] = query
            captured["filters"] = filters
            captured["top_n"] = top_n
            return RetrievalResult(
                query=query,
                canonical_arabic_query="الوضوء",
                language="bn",
                translated=True,
                evidence_sufficient=True,
                chunks=[RetrievedChunk(chunk_id="c1", text="الوضوء", rerank_score=0.5)],
            )

    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", FakeRunner)
    response = client.post(
        "/api/v1/chat",
        json={
            "query": "الوضوء",
            "book_id": "book-1",
            "volume": "2",
            "region": "main",
            "verified": True,
            "top_n": 3,
        },
    )

    assert response.status_code == 200
    assert captured["query"] == "الوضوء"
    assert captured["filters"] == PayloadFilter(
        book_id="book-1", volume="2", region="main", verified=True
    )
    assert captured["top_n"] == 3


def test_chat_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "query_validation_error"


def test_chat_rejects_invalid_region(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "الوضوء", "region": "bogus", "top_n": 9})

    assert response.status_code == 422


def test_chat_persists_history_row(session: Session, client: TestClient) -> None:
    client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "answer_language": "bn"})

    rows = session.scalars(select(ChatHistory)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.question == "ما حكم الوضوء"
    assert row.normalized_query == "ما حكم الوضوء"
    assert row.answer_language == "bn"
    assert row.answer is not None
    assert row.answer["explanation"]["html"]
    assert row.confidence == "low"
    assert row.refusal is None
    assert len(row.sources) == 2
    assert row.sources[0]["chunk_id"] == "c1"


def test_chat_rate_limited_per_ip_with_retry_after(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import deps as deps_module
    from app.services.rate_limit import RateLimitStore

    # The conftest fixture overrides get_redis with a fakeredis instance.
    redis = app.dependency_overrides[deps_module.get_redis]()
    limiter = RateLimitStore(redis)
    scope = "chat"

    # Drive the shared redis limiter past the 20/min chat cap for this IP.
    client_ip = "testclient"
    for _ in range(20):
        allowed, _ = limiter.check(f"{scope}:ip:{client_ip}", limit=20, window_seconds=60)
        assert allowed
    allowed, retry_after = limiter.check(f"{scope}:ip:{client_ip}", limit=20, window_seconds=60)
    assert not allowed
    assert retry_after > 0

    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(response.headers["Retry-After"]) > 0
