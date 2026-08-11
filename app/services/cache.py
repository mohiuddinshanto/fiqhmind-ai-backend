import json
from typing import Any, cast

import structlog
from redis import Redis

logger = structlog.get_logger(__name__)


class CacheService:
    """Best-effort JSON cache over Redis for answers, embeddings and query results.

    Every operation fails open (Phase 15): when Redis is unreachable the cache
    degrades to a no-op so the primary request path (retrieval, generation,
    indexing) never depends on cache availability.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def get(self, key: str) -> Any | None:
        try:
            raw = cast("str | None", self._redis.get(key))
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_get_failed", key=key, error=str(exc))
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("cache_get_invalid_json", key=key)
            return None

    def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        try:
            self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_set_failed", key=key, error=str(exc))

    def set_raw(self, key: str, raw: str, *, ttl_seconds: int) -> None:
        try:
            self._redis.set(key, raw, ex=ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_set_raw_failed", key=key, error=str(exc))

    def delete(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_delete_failed", key=key, error=str(exc))

    def exists(self, key: str) -> bool:
        try:
            return bool(self._redis.exists(key))
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_exists_failed", key=key, error=str(exc))
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern (used for re-index invalidation)."""
        deleted = 0
        try:
            for key in self._redis.scan_iter(match=pattern):
                deleted += cast(int, self._redis.delete(key))
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            logger.warning("cache_delete_pattern_failed", pattern=pattern, error=str(exc))
        return deleted

    def get_or_set(self, key: str, *, ttl_seconds: int, loader: Any) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = loader()
        if value is not None:
            self.set(key, value, ttl_seconds=ttl_seconds)
        return value
