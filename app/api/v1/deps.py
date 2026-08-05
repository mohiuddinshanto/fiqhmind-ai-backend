from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy.orm import Session

from app.core.postgres import get_db
from app.core.qdrant import QdrantStore, get_qdrant_client, get_qdrant_store
from app.core.redis import get_redis
from app.services.cache import CacheService
from app.services.rate_limit import RateLimitStore

DbSession = Annotated[Session, Depends(get_db)]


def get_cache_service(redis: Annotated[Redis, Depends(get_redis)]) -> CacheService:
    return CacheService(redis)


def get_rate_limiter(redis: Annotated[Redis, Depends(get_redis)]) -> RateLimitStore:
    return RateLimitStore(redis)


def get_qdrant_dep() -> QdrantClient:
    return get_qdrant_client()


def get_store_dep() -> QdrantStore:
    return get_qdrant_store()


__all__ = [
    "DbSession",
    "get_cache_service",
    "get_db",
    "get_qdrant_dep",
    "get_qdrant_store",
    "get_rate_limiter",
    "get_redis",
    "get_store_dep",
]
