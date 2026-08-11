"""Tests for the Phase 15 maintenance task (daily cache eviction)."""

import fakeredis

import app.tasks.maintenance as maintenance_module
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
