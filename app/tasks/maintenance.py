"""Phase 15 background maintenance tasks.

`evict_caches` runs daily via Celery beat. The QA answer cache and the
15-minute chunk-level results cache are content-scoped — they can serve stale
entries after a book is re-indexed — so they are cleared on a schedule
(ARCHITECTURE §Phase 15 "Daily: QA cache eviction for re-indexed books").

The embedding cache is intentionally left alone: its vectors are content-
addressed (text hash) and versioned (dense dimension), so re-indexing can never
reuse a stale-shaped vector and its TTL is the correct invalidation policy.
"""

import structlog

from app.core.redis import get_redis
from app.services.cache import CacheService
from app.worker.celery_app import celery_app

logger = structlog.get_logger(__name__)

_CACHE_NAMESPACES = ("qa:*", "chunk:*")


@celery_app.task(name="app.tasks.maintenance.evict_caches")
def evict_caches() -> dict[str, int]:
    """Evict content-scoped Phase 15 caches (QA answers, retrieval results)."""
    cache = CacheService(get_redis())
    deleted: dict[str, int] = {}
    for pattern in _CACHE_NAMESPACES:
        deleted[pattern] = cache.delete_pattern(pattern)
    logger.info("cache_eviction_completed", deleted=deleted)
    return deleted
