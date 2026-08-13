"""Tests for the Phase 15 maintenance tasks (cache eviction, health checks)."""

from collections.abc import Generator

import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.services.generation.providers as providers_module
import app.tasks.maintenance as maintenance_module
from app.core.config import Settings
from app.db.base import Base
from app.db.models import Chunk
from app.services.cache import CacheService


def test_evict_caches_clears_qa_and_chunk_namespaces(monkeypatch) -> None:
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeStrictRedis(server=server)
    cache = CacheService(redis)
    cache.set("qa:book1", "a", ttl_seconds=60)
    cache.set("qa:book2", "b", ttl_seconds=60)
    cache.set("chunk:hot", "c", ttl_seconds=60)
    cache.set("embedding:v1:1024:x", "d", ttl_seconds=60)
    cache.set("rate:limit:ip", "e", ttl_seconds=60)

    monkeypatch.setattr(maintenance_module, "get_redis", lambda: redis)
    deleted = maintenance_module.evict_caches()

    assert deleted == {"qa:*": 2, "chunk:*": 1}
    assert cache.get("qa:book1") is None
    assert cache.get("chunk:hot") is None
    # Embedding cache is content-addressed + versioned, and rate-limit keys are
    # not caches — the daily eviction deliberately leaves both alone.
    assert cache.get("embedding:v1:1024:x") == "d"
    assert cache.get("rate:limit:ip") == "e"


def test_provider_health_check_returns_status_dict(monkeypatch) -> None:
    """With no API keys the deterministic adapter needs no providers."""
    no_key_settings = Settings(gemini_api_key=None, groq_api_key=None, openrouter_api_key=None)
    monkeypatch.setattr(providers_module, "get_settings", lambda: no_key_settings)

    status = maintenance_module.provider_health_check()

    assert status == {}


@pytest.fixture()
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


class _FakeQdrantStore:
    def __init__(self, point_ids: list[str]) -> None:
        self._point_ids = point_ids

    def list_point_ids(self) -> list[str]:
        return list(self._point_ids)


def test_index_health_check_reports_orphans_never_indexed_duplicates(
    session: Session, monkeypatch
) -> None:
    """Weekly reconciliation: Qdrant orphans, un-indexed chunks, duplicate texts."""
    session.add_all(
        [
            # Both sides healthy.
            Chunk(chunk_id="healthy1", raw_text="h1", normalized_text="h1"),
            # In Postgres but never indexed → never_indexed.
            Chunk(chunk_id="unindexed1", raw_text="u1", normalized_text="u1"),
            Chunk(chunk_id="unindexed2", raw_text="u2", normalized_text="u2"),
            # Duplicate normalized text → duplicate detection.
            Chunk(chunk_id="dup1", raw_text="same text", normalized_text="same text"),
            Chunk(chunk_id="dup2", raw_text="same text again", normalized_text="same text"),
        ]
    )
    session.commit()

    # Qdrant has healthy1 + an orphan point that has no Postgres row.
    qdrant_ids = ["healthy1", "orphan1"]
    monkeypatch.setattr(
        maintenance_module,
        "get_qdrant_store",
        lambda: _FakeQdrantStore(qdrant_ids),
    )

    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(maintenance_module, "get_session_factory", lambda: factory)

    report = maintenance_module.index_health_check()

    assert report["qdrant_point_count"] == 2
    assert report["pg_chunk_count"] == 5
    assert report["orphan_points"] == ["orphan1"]
    assert report["never_indexed"] == ["dup1", "dup2", "unindexed1", "unindexed2"]
    assert report["duplicate_count"] == 1
    assert report["duplicates"] == [("same text", 2)]

