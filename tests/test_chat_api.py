"""Tests for the Phase 10 chat SSE endpoint (POST /api/v1/chat)."""

import json
import re
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


def test_chat_refusal_answer_is_not_cached(client: TestClient, monkeypatch) -> None:
    calls = {"runner": 0}

    class CountingRunner:
        def __init__(self, session, store, **kwargs) -> None:
            pass

        def search(self, query, *, filters=None, top_n=None) -> RetrievalResult:
            calls["runner"] += 1
            return RetrievalResult(
                query=query,
                canonical_arabic_query="ما حكم الوضوء",
                language="bn",
                translated=True,
                evidence_sufficient=False,
                chunks=[],
            )

    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", CountingRunner)
    client.post("/api/v1/chat", json={"query": "অজানা প্রশ্ন"})
    client.post("/api/v1/chat", json={"query": "অজানা প্রশ্ন"})
    # Each call runs retrieval — the refusal is never served from cache.
    assert calls["runner"] == 2


def test_chat_forwards_filters_and_top_n(client: TestClient, monkeypatch) -> None:
    captured = {}

    class FakeRunner:
        def __init__(self, session, store, **kwargs) -> None:
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


def test_chat_serves_repeated_question_from_qa_cache(
    client: TestClient, monkeypatch
) -> None:
    calls = {"generator": 0}

    real_generator = chat_endpoints.get_generator

    def counting_generator():
        generator = real_generator()
        original_generate = generator.generate

        def generate(retrieval, *, answer_language="bn"):
            calls["generator"] += 1
            return original_generate(retrieval, answer_language=answer_language)

        generator.generate = generate
        return generator

    monkeypatch.setattr(chat_endpoints, "get_generator", counting_generator)

    first = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    assert first.status_code == 200
    assert calls["generator"] == 1

    second = client.post("/api/v1/chat", json={"query": "ما حكم الوضوء"})
    assert second.status_code == 200
    # The repeated question is served from the QA cache — no re-generation.
    assert calls["generator"] == 1
    assert _parse_sse(second.text) == _parse_sse(first.text)


def test_chat_qa_cache_is_scoped_by_language(client: TestClient, monkeypatch) -> None:
    calls = {"generator": 0}

    real_generator = chat_endpoints.get_generator

    def counting_generator():
        generator = real_generator()
        original_generate = generator.generate

        def generate(retrieval, *, answer_language="bn"):
            calls["generator"] += 1
            return original_generate(retrieval, answer_language=answer_language)

        generator.generate = generate
        return generator

    monkeypatch.setattr(chat_endpoints, "get_generator", counting_generator)

    client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "answer_language": "bn"})
    client.post("/api/v1/chat", json={"query": "ما حكم الوضوء", "answer_language": "en"})

    # Different answer language => a distinct QA cache entry.
    assert calls["generator"] == 2


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


# ------------------------------------------------------------------
# DeterministicSynthesizer concise-answer regression tests
# ------------------------------------------------------------------

_AUTHOR_CHUNK_TEXT = (
    "Kitab: بین یدی الكتاب\n"
    "Topic: ১. ভূমিকা\n\n"
    "শায়খ মুহাম্মাদ আওয়ামাহ রচিত 'আছারুল হাদীসিশ শরীফ ফী "
    "ইখতিলাফিল আইম্মাতিল ফুকাহা' নামক একটি গুরুত্বপূর্ণ গ্রন্থ।"
)
_TITLE_CHUNK_TEXT = (
    "Kitab: بین یدی الكتاب\n"
    "Topic: ১. ভূমিকা\n\n"
    "আছারুল হাদীস প্রশ্নোত্তর — এটি একটি গুরুত্বপূর্ণ ফিকহী গ্রন্থ।"
)
_GENERAL_CHUNK_TEXT = (
    "Kitab: بین یدی الكتاب\n"
    "Topic: মতভেদ\n\n"
    "ইমামগণ হাদিসকে সুন্নাহর মূল উৎস বলে মান্য করেছেন।"
)


class BengaliFakeStore:
    """Store returning Bengali author / title / general evidence chunks."""

    def __init__(self, chunks=None):
        self._chunks = chunks or [
            SimpleNamespace(
                id="author-c1",
                score=0.9,
                payload={
                    "chunk_id": "author-c1",
                    "text": _AUTHOR_CHUNK_TEXT,
                    "book_name": "আছারুল হাদীস প্রশ্নোত্তর",
                    "volume": None,
                    "printed_page_start": None,
                    "printed_page_end": None,
                    "topic": "ভূমিকা",
                    "region": "main",
                    "lang": "bn",
                    "verified": True,
                },
            ),
            SimpleNamespace(
                id="general-c1",
                score=0.5,
                payload={
                    "chunk_id": "general-c1",
                    "text": _GENERAL_CHUNK_TEXT,
                    "book_name": "আছারুল হাদীস প্রশ্নোত্তর",
                    "volume": None,
                    "printed_page_start": None,
                    "printed_page_end": None,
                    "topic": "মতভেদ",
                    "region": "main",
                    "lang": "bn",
                    "verified": True,
                },
            ),
        ]

    def search_dense(self, vector, *, limit, filter_=None):
        return self._chunks

    def search_sparse(self, vector, *, limit, filter_=None):
        return []


def _patch_bengali_store(monkeypatch, session):
    """Replace deps so the chat endpoint uses BengaliFakeStore."""
    from app.api.v1 import deps

    app.dependency_overrides[deps.get_db] = lambda: session
    app.dependency_overrides[deps.get_store_dep] = lambda: BengaliFakeStore()


# ---- Test A: Bengali author question → concise answer with author name ----


def test_bengali_author_question_concise_answer(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Author question returns a concise answer containing the author name,
    NOT multiple [EVIDENCE_x] blocks dumped into the explanation HTML."""
    _patch_bengali_store(monkeypatch, session)

    response = client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [e for e, _ in events]
    assert "error" not in names

    done = next(data for name, data in events if name == "done")
    assert done["refusal"] is None

    # Reassemble explanation HTML from delta events.
    deltas = [data["text"] for name, data in events if name == "delta"]
    html = "".join(deltas)

    # The author name must appear in the answer.
    assert "শায়খ মুহাম্মাদ আওয়ামাহ" in html

    # No [EVIDENCE_x] blocks in the user-facing explanation.
    assert "[EVIDENCE_" not in html

    # The answer should be concise — well under 500 chars for a factual answer.
    # Strip HTML tags for a length check.
    plain = re.sub(r"<[^>]+>", "", html).strip()
    assert len(plain) < 500, f"Answer too long ({len(plain)} chars): {plain[:200]}..."


# ---- Test B: Bengali book title question → concise title answer ----


def test_bengali_title_question_concise_answer(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Book title question returns the title, NOT [EVIDENCE_x] blocks."""
    title_chunks = [
        SimpleNamespace(
            id="title-c1",
            score=0.9,
            payload={
                "chunk_id": "title-c1",
                "text": _TITLE_CHUNK_TEXT,
                "book_name": "আছারুল হাদীস প্রশ্নোত্তর",
                "volume": None,
                "printed_page_start": None,
                "printed_page_end": None,
                "topic": "ভূমিকা",
                "region": "main",
                "lang": "bn",
                "verified": True,
            },
        ),
    ]
    from app.api.v1 import deps

    app.dependency_overrides[deps.get_db] = lambda: session
    app.dependency_overrides[deps.get_store_dep] = lambda: BengaliFakeStore(title_chunks)

    response = client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের নাম কি", "answer_language": "bn"},
    )
    assert response.status_code == 200

    events = _parse_sse(response.text)
    done = next(data for name, data in events if name == "done")
    assert done["refusal"] is None

    deltas = [data["text"] for name, data in events if name == "delta"]
    html = "".join(deltas)

    # The book title must appear.
    assert "আছারুল হাদীস প্রশ্নোত্তর" in html

    # No [EVIDENCE_x] blocks.
    assert "[EVIDENCE_" not in html


# ---- Test C: Absent evidence → refusal, no hallucinated author ----


def test_absent_evidence_refuses_without_hallucinating_author(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """When no author evidence is in the chunks, the system refuses
    rather than hallucinating an author name."""
    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", type(
        "Runner",
        (),
        {
            "__init__": lambda self, *a, **kw: None,
            "search": lambda self, query, **kw: RetrievalResult(
                query=query,
                canonical_arabic_query=query,
                language="bn",
                translated=False,
                evidence_sufficient=False,
                chunks=[],
            ),
        },
    ))

    response = client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    events = _parse_sse(response.text)
    done = next(data for name, data in events if name == "done")
    assert done["refusal"] is not None
    assert done["refusal"]["reason"] == "insufficient_evidence"


# ---- Test D: Successful answer is still cached ----


def test_bengali_author_answer_is_cached(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """A successful (non-refusal) answer is served from cache on repeat."""
    _patch_bengali_store(monkeypatch, session)

    calls = {"gen": 0}
    real_gen = chat_endpoints.get_generator

    def counting_gen():
        g = real_gen()
        orig = g.generate
        def counting(retrieval, *, answer_language="bn"):
            calls["gen"] += 1
            return orig(retrieval, answer_language=answer_language)
        g.generate = counting
        return g

    monkeypatch.setattr(chat_endpoints, "get_generator", counting_gen)

    r1 = client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    assert r1.status_code == 200
    assert calls["gen"] == 1

    r2 = client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    assert r2.status_code == 200
    # Cache hit — no re-generation.
    assert calls["gen"] == 1


# ---- Test E: Refusal is NOT cached (regression guard) ----


def test_bengali_refusal_answer_is_not_cached(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Refusal answers must never be cached, preserving the cache regression fix."""
    calls = {"runner": 0}

    class CountingRunner:
        def __init__(self, session, store, **kwargs):
            pass

        def search(self, query, *, filters=None, top_n=None):
            calls["runner"] += 1
            return RetrievalResult(
                query=query,
                canonical_arabic_query=query,
                language="bn",
                translated=False,
                evidence_sufficient=False,
                chunks=[],
            )

    monkeypatch.setattr(chat_endpoints, "RetrievalRunner", CountingRunner)
    client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    client.post(
        "/api/v1/chat",
        json={"query": "এই কিতাবের লেখক কে", "answer_language": "bn"},
    )
    # Each call runs retrieval — the refusal is never served from cache.
    assert calls["runner"] == 2
