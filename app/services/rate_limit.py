import time
import uuid
from typing import cast

import structlog
from redis import Redis

logger = structlog.get_logger(__name__)


class RateLimitStore:
    """Redis-backed sliding-window rate limiter (Phase 14).

    Best-effort by design (like CacheService): when Redis is unreachable every
    operation fails *open* — it logs a warning and allows the request — so a
    Redis outage degrades the API to unthrottled instead of taking it down
    (Phase 15 "best-effort, degrade" principle).
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _key(self, scope: str) -> str:
        return f"rate:{scope}"

    def check(self, scope: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). A check records one attempt."""
        try:
            return self._check(scope, limit=limit, window_seconds=window_seconds)
        except Exception as exc:  # noqa: BLE001 - fail open when Redis is unavailable
            logger.warning("rate_limit_check_failed", scope=scope, error=str(exc))
            return True, 0

    def _check(self, scope: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        window_start = now - window_seconds
        key = self._key(scope)

        pipe = self._redis.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds)
        _removed, _added, count, _expire = pipe.execute()

        if count <= limit:
            return True, 0

        oldest = cast("list[tuple[str, float]]", self._redis.zrange(key, 0, 0, withscores=True))
        retry_after = 0
        if oldest:
            retry_after = max(1, int(window_seconds - (now - oldest[0][1])) + 1)
        return False, retry_after

    def remaining(self, scope: str, *, limit: int, window_seconds: int) -> int:
        try:
            now = time.time()
            window_start = now - window_seconds
            key = self._key(scope)

            self._redis.zremrangebyscore(key, 0, window_start)
            count = cast(int, self._redis.zcard(key))
            self._redis.expire(key, window_seconds)
            return max(0, limit - count)
        except Exception as exc:  # noqa: BLE001 - fail open when Redis is unavailable
            logger.warning("rate_limit_remaining_failed", scope=scope, error=str(exc))
            return limit

    def reset(self, scope: str) -> None:
        try:
            self._redis.delete(self._key(scope))
        except Exception as exc:  # noqa: BLE001 - fail open when Redis is unavailable
            logger.warning("rate_limit_reset_failed", scope=scope, error=str(exc))
