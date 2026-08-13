"""Phase 15 M3 tests (C-L): POST /api/v1/chat with `stream: true`.

Covers the true-streaming protocol (`start -> token* -> done | error`),
cache-hit replay, cache-miss accumulate/validate/cache, shared cache keys
across modes, exactly-once history persistence, failure -> `error` (no partial
cache/history), refusal streaming, and the preserved rate limit.
"""

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
from app.schemas.chat import ChatRequest
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


def _counting_generator(calls: dict):
    real_generator = chat_endpoints.get_generator

    def factory():
        generator = real_generator()
        original_generate = generator.generate
        original_stream = generator.stream_answer

        def generate(retrieval, *, answer_language="bn"):
            calls["generate"] += 1
            return original_generate(retrieval, answer_language=answer_language)

        def stream_answer(retrieval, *, answer_language="bn"):
            calls["stream"] += 1
            yield from original_stream(retrieval, answer_language=answer_language)

        generator.generate = generate
        generator.stream_answer = stream_answer
        return generator

    return factory


# ------------------------------------------------------------------ C. protocol


def test_stream_true_emits_start_tokens_done(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "error" not in names
    assert names[0] == "start"
    assert "token" in names
    assert names[-1] == "done"
    assert "delta" not in names  # true streaming protocol, not the Phase 10 replay

    _, start = events[0]
    assert start["query"] == "ما حكم الوضوء"
    assert start["answer_language"] == "bn"
    assert start["latency_budget_ms"] > 0

    _, done = events[-1]
    assert done["refusal"] is None
    assert done["answer_language"] == "bn"


def test_stream_tokens_reassemble_into_validated_answer(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    events = _parse_sse(response.text)

    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens
    done = next(data for name, data in events if name == "done")

    # Live tokens reassemble into the final validated explanation.
    assert "".join(tokens).rstrip() == done["explanation"]["html"]
    assert "الماء طهور" in done["explanation"]["html"]
    # The `done` payload is the full structured answer for rendering.
    assert done["citations"][0]["book"] == "Al-Hidayah"
    assert done["citations"][0]["page"] == "5"
    assert [q["text"] for q in done["arabic_quotes"]] == [
        "الماء طهور لا ينجسه شيء",
        "الوضوء شرط الصلاة",
    ]
    assert done["confidence"]["level"] == "low"
    assert done["confidence"]["source_agreement"] == "consensus"
    assert done["caveats"]
    assert done["related"]


def test_stream_defaults_to_false_and_keeps_phase10_protocol(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    events = _parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "meta"
    assert "delta" in names
    assert names[-1] == "done"
    assert "token" not in names


# ----------------------------------------------- F. cache hit replay (streaming)


def test_stream_cache_hit_replays_without_retrieval_or_generation(
    client: TestClient, monkeypatch
) -> None:
    calls = {"generate": 0, "stream": 0, "search": 0}

    class CountingRunner:
        def __init__(self, session, store, **kwargs) -> None:
            pass

        def search(self, query, *, filters=None, top_n=None) -> RetrievalResult:
            calls["search"] += 1
            return RetrievalResult(
                query=query,
                canonical_arabic_query="ما حكم الوضوء",
                language="bn",
                translated=True,
                evidence_sufficient=True,
                chunks=[
                    RetrievedChunk(chunk_id="c1", text="الوضوء", rerank_score=0.5)
                ],
            )

    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", CountingRunner)
    monkeypatch.setattr(
        chat_endpoints, "get_generator", _counting_generator(calls)
    )

    first = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert first.status_code == 200
    assert calls["search"] == 1
    assert calls["stream"] == 1

    second = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert second.status_code == 200
    # The repeated question is replayed from the QA cache: no retrieval, no generation.
    assert calls["search"] == 1
    assert calls["stream"] == 1

    events = _parse_sse(second.text)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert "token" in names
    assert names[-1] == "done"
    assert "error" not in names
    assert _parse_sse(second.text) == _parse_sse(first.text)


# ------------------------------------------------- G. cache miss accumulate+cache


def test_stream_cache_miss_caches_validated_answer(client: TestClient, monkeypatch) -> None:
    calls = {"generate": 0, "stream": 0}
    monkeypatch.setattr(chat_endpoints, "get_generator", _counting_generator(calls))

    first = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert first.status_code == 200
    assert calls["stream"] == 1

    second = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert second.status_code == 200
    # Second request is a cache hit — stream_answer never runs again.
    assert calls["stream"] == 1
    assert _parse_sse(second.text) == _parse_sse(first.text)


# ------------------------------------------------ H. shared cache across modes


def test_stream_and_replay_modes_share_the_qa_cache(
    client: TestClient, monkeypatch
) -> None:
    calls = {"generate": 0, "stream": 0}
    monkeypatch.setattr(chat_endpoints, "get_generator", _counting_generator(calls))

    # Seed the cache with a non-streaming request...
    first = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    assert first.status_code == 200
    assert calls["generate"] == 1
    assert calls["stream"] == 0

    # ...then a streaming request must be served from the same cache entry.
    second = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert second.status_code == 200
    assert calls["generate"] == 1
    assert calls["stream"] == 0
    names = [name for name, _ in _parse_sse(second.text)]
    assert names[0] == "start"
    assert names[-1] == "done"

    # And the reverse: streaming first, replay second.
    third = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert calls["stream"] == 0  # still cached
    assert calls["generate"] == 1
    names = [name for name, _ in _parse_sse(third.text)]
    assert names[0] == "start"
    assert "token" in names
    assert names[-1] == "done"


def test_replay_mode_still_replays_cached_answer_after_stream_seed(
    client: TestClient, monkeypatch
) -> None:
    calls = {"generate": 0, "stream": 0}
    monkeypatch.setattr(chat_endpoints, "get_generator", _counting_generator(calls))

    client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert calls["stream"] == 1

    replay = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    assert replay.status_code == 200
    names = [name for name, _ in _parse_sse(replay.text)]
    assert names[0] == "meta"
    assert "delta" in names
    assert names[-1] == "done"
    assert calls["stream"] == 1


# ----------------------------------------------------- I. history persistence


def test_streaming_persists_exactly_one_history_row(
    session: Session, client: TestClient
) -> None:
    client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})

    rows = session.scalars(select(ChatHistory)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.question == "ما حكم الوضوء"
    assert row.answer_language == "bn"
    assert row.answer is not None
    assert row.answer["explanation"]["html"]
    assert row.confidence == "low"
    assert row.refusal is None
    assert len(row.sources) == 2
    assert row.sources[0]["chunk_id"] == "c1"


# ------------------------------------------------- J. failure -> error event


def test_stream_generation_failure_emits_error_without_cache_or_history(
    session: Session, client: TestClient, monkeypatch
) -> None:
    class _FailingGenerator:
        def stream_answer(self, retrieval, *, answer_language="bn"):
            raise RuntimeError("vendor exploded")
            yield  # pragma: no cover - keeps this a generator

    monkeypatch.setattr(chat_endpoints, "get_generator", lambda: _FailingGenerator())

    response = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert "done" not in names
    assert names[-1] == "error"
    error = events[-1][1]
    assert error["code"] == "generation_error"
    assert "vendor exploded" in error["message"]

    # A failed generation writes neither a cache entry nor a history row.
    from app.api.v1 import deps as deps_module

    redis = app.dependency_overrides[deps_module.get_redis]()
    cache_key = chat_endpoints._qa_cache_key(ChatRequest(query="ما حكم الوضوء", stream=True))
    assert redis.get(cache_key) is None
    assert session.scalars(select(ChatHistory)).all() == []


# ------------------------------------------------- K. refusal streaming


def test_stream_insufficient_evidence_emits_done_with_refusal(
    client: TestClient, monkeypatch
) -> None:
    class EmptyRunner:
        def __init__(self, session, store, **kwargs) -> None:
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
    response = client.post("/api/v1/chat", json={"query": "অজানা প্রশ্ন", "stream": True})

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "error" not in names
    assert "token" not in names
    assert names[0] == "start"
    assert names[-1] == "done"
    done = events[-1][1]
    assert done["refusal"]["reason"] == "insufficient_evidence"
    assert done["refusal"]["closest_evidence"] == []


# ------------------------------------------------------ L. rate limit preserved


def test_stream_rate_limited_per_ip_with_retry_after(
    client: TestClient, monkeypatch
) -> None:
    from app.api.v1 import deps as deps_module
    from app.services.rate_limit import RateLimitStore

    redis = app.dependency_overrides[deps_module.get_redis]()
    limiter = RateLimitStore(redis)
    scope = "chat"

    client_ip = "testclient"
    for _ in range(20):
        allowed, _ = limiter.check(f"{scope}:ip:{client_ip}", limit=20, window_seconds=60)
        assert allowed
    allowed, retry_after = limiter.check(f"{scope}:ip:{client_ip}", limit=20, window_seconds=60)
    assert not allowed
    assert retry_after > 0

    response = client.post(
        "/api/v1/chat", json={"query": "ما حكم الوضوء", "stream": True}
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(response.headers["Retry-After"]) > 0


# ------------------------------------------------- filter/key parity sanity


def test_stream_qa_cache_key_excludes_the_stream_flag() -> None:
    assert chat_endpoints._qa_cache_key(
        ChatRequest(query="ما حكم الوضوء", stream=True)
    ) == chat_endpoints._qa_cache_key(ChatRequest(query="ما حكم الوضوء", stream=False))
