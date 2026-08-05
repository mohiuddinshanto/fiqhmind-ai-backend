import fakeredis
import pytest

from app.services.cache import CacheService
from app.services.rate_limit import RateLimitStore


@pytest.fixture()
def redis():
    server = fakeredis.FakeServer()
    return fakeredis.FakeStrictRedis(server=server)


def test_cache_service_roundtrip(redis) -> None:
    cache = CacheService(redis)
    cache.set("qa:hello", {"answer": "نص"}, ttl_seconds=60)
    assert cache.get("qa:hello") == {"answer": "نص"}
    assert cache.exists("qa:hello")
    cache.delete("qa:hello")
    assert cache.get("qa:hello") is None


def test_cache_service_delete_pattern(redis) -> None:
    cache = CacheService(redis)
    cache.set("qa:book:1:q1", "a", ttl_seconds=60)
    cache.set("qa:book:1:q2", "b", ttl_seconds=60)
    cache.set("qa:book:2:q1", "c", ttl_seconds=60)
    deleted = cache.delete_pattern("qa:book:1:*")
    assert deleted == 2
    assert cache.get("qa:book:2:q1") == "c"


def test_cache_service_get_or_set(redis) -> None:
    cache = CacheService(redis)
    calls = []

    def loader():
        calls.append(1)
        return {"value": 42}

    assert cache.get_or_set("qa:hot", ttl_seconds=60, loader=loader) == {"value": 42}
    assert cache.get_or_set("qa:hot", ttl_seconds=60, loader=loader) == {"value": 42}
    assert len(calls) == 1


def test_rate_limit_allows_within_window(redis) -> None:
    limiter = RateLimitStore(redis)
    for _ in range(5):
        allowed, retry_after = limiter.check("user:1", limit=5, window_seconds=60)
        assert allowed
        assert retry_after == 0


def test_rate_limit_blocks_beyond_limit(redis) -> None:
    limiter = RateLimitStore(redis)
    for _ in range(5):
        limiter.check("user:1", limit=5, window_seconds=60)
    allowed, retry_after = limiter.check("user:1", limit=5, window_seconds=60)
    assert not allowed
    assert retry_after > 0


def test_rate_limit_reset(redis) -> None:
    limiter = RateLimitStore(redis)
    for _ in range(6):
        limiter.check("user:1", limit=5, window_seconds=60)
    assert not limiter.check("user:1", limit=5, window_seconds=60)[0]
    limiter.reset("user:1")
    assert limiter.check("user:1", limit=5, window_seconds=60)[0]


def test_rate_limit_remaining(redis) -> None:
    limiter = RateLimitStore(redis)
    assert limiter.remaining("user:1", limit=10, window_seconds=60) == 10
    limiter.check("user:1", limit=10, window_seconds=60)
    limiter.check("user:1", limit=10, window_seconds=60)
    assert limiter.remaining("user:1", limit=10, window_seconds=60) == 8
