from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import RateLimitError
from app.core.postgres import get_db
from app.core.qdrant import QdrantStore, get_qdrant_client, get_qdrant_store
from app.core.redis import get_redis
from app.core.storage import StorageProvider, get_storage_provider
from app.services.cache import CacheService
from app.services.rate_limit import RateLimitStore

DbSession = Annotated[Session, Depends(get_db)]


def get_cache_service(redis: Annotated[Redis, Depends(get_redis)]) -> CacheService:
    return CacheService(redis)


def get_rate_limiter(redis: Annotated[Redis, Depends(get_redis)]) -> RateLimitStore:
    return RateLimitStore(redis)


def _rate_limit_bounds(settings: Settings, scope: str) -> tuple[int, int]:
    """Return (limit, window_seconds) for a named Phase 14 scope."""
    if scope == "chat":
        return settings.rate_limit_chat_per_min, 60
    if scope == "search":
        return settings.rate_limit_search_per_min, 60
    if scope == "auth":
        return settings.rate_limit_auth_per_min, 60
    if scope == "ingest":
        return settings.rate_limit_ingest_per_min, 60
    if scope == "eval":
        return settings.rate_limit_eval_per_hour, 3600
    raise ValueError(f"unknown rate-limit scope: {scope}")


def require_rate_limit(scope: str) -> Callable:
    """Build a per-IP sliding-window rate-limit dependency for a Phase 14 scope.

    Anonymous (unauthenticated) callers are scoped by client IP; the dependency
    raises `RateLimitError` (429 with `Retry-After`) when the window is exceeded.
    Authenticated per-user scoping can layer on top once auth exists.
    """

    def dependency(
        request: Request,
        limiter: Annotated[RateLimitStore, Depends(get_rate_limiter)],
    ) -> None:
        limit, window = _rate_limit_bounds(get_settings(), scope)
        client_ip = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(
            f"{scope}:ip:{client_ip}", limit=limit, window_seconds=window
        )
        if not allowed:
            raise RateLimitError(
                "rate limit exceeded, please retry later", retry_after_seconds=retry_after
            )

    return dependency


def get_qdrant_dep() -> QdrantClient:
    return get_qdrant_client()


def get_store_dep() -> QdrantStore:
    return get_qdrant_store()


def get_storage_dep() -> StorageProvider:
    return get_storage_provider()


__all__ = [
    "DbSession",
    "get_cache_service",
    "get_db",
    "get_qdrant_dep",
    "get_qdrant_store",
    "get_rate_limiter",
    "get_redis",
    "get_storage_dep",
    "get_storage_provider",
    "get_store_dep",
    "require_rate_limit",
]
