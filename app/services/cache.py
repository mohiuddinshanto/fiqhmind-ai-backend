import json
from typing import Any, cast

import structlog
from redis import Redis

logger = structlog.get_logger(__name__)


class CacheService:
    """JSON cache over Redis for answers, embeddings and query results (Phase 4)."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def get(self, key: str) -> Any | None:
        raw = cast("str | None", self._redis.get(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("cache_get_invalid_json", key=key)
            return None

    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)

    def set_raw(self, key: str, raw: str, *, ttl_seconds: int) -> None:
        self._redis.set(key, raw, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self._redis.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._redis.exists(key))

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern (used for re-index invalidation)."""
        deleted = 0
        for key in self._redis.scan_iter(match=pattern):
            deleted += cast(int, self._redis.delete(key))
        return deleted

    def get_or_set(self, key: str, *, ttl_seconds: int, loader: Any) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = loader()
        if value is not None:
            self.set(key, value, ttl_seconds=ttl_seconds)
        return value
