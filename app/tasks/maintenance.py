"""Phase 15 background maintenance tasks.

`evict_caches` runs daily via Celery beat. The QA answer cache and the
15-minute chunk-level results cache are content-scoped — they can serve stale
entries after a book is re-indexed — so they are cleared on a schedule
(ARCHITECTURE §Phase 15 "Daily: QA cache eviction for re-indexed books").

`provider_health_check` also runs daily (ARCHITECTURE §Phase 15 "Daily:
provider-quota health check"): it re-pulls each configured LLM provider's
free-tier limits so a silently-changed quota is caught before it breaks chat.

`index_health_check` runs weekly (ARCHITECTURE §Phase 15 "Weekly: index
health"): it reconciles the Qdrant point set against the Postgres `chunks`
table (orphan points, chunks never indexed) and flags duplicate normalized
texts, plus reports the QA-cache/embedding-cache hit potential by namespace.

The embedding cache is intentionally left alone: its vectors are content-
addressed (text hash) and versioned (dense dimension), so re-indexing can never
reuse a stale-shaped vector and its TTL is the correct invalidation policy.
"""

import structlog

from app.core.asyncio_utils import run_coroutine
from app.core.postgres import get_session_factory
from app.core.qdrant import get_qdrant_store
from app.core.redis import get_redis
from app.db.repositories import ChunkRepository
from app.services.cache import CacheService
from app.services.generation.providers import ProviderHealthService
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


@celery_app.task(name="app.tasks.maintenance.provider_health_check")
def provider_health_check() -> dict:
    """Re-check configured LLM provider quotas/circuit breakers (daily).

    Runs one pass of `ProviderHealthService` checks off the request path. With
    no API keys configured this returns an empty status — the deterministic
    adapter needs no external provider. Returns the resulting health status.
    """
    service = ProviderHealthService()
    status = run_coroutine(service.run_health_checks_once)
    logger.info(
        "provider_health_check_completed",
        providers=sorted(status),
    )
    return status


@celery_app.task(name="app.tasks.maintenance.index_health_check")
def index_health_check() -> dict:
    """Reconcile the Qdrant index against the Postgres chunks table (weekly).

    Detects orphan points (vectors whose chunk_id has no Postgres row), chunks
    that were never indexed (Postgres chunk_id with no Qdrant point), and
    duplicate normalized texts. Returns a report; issues are logged as warnings
    so the task result doubles as an alert surface.
    """
    session = get_session_factory()()
    try:
        store = get_qdrant_store()
        chunk_repo = ChunkRepository(session)

        try:
            qdrant_ids = set(store.list_point_ids())
        except Exception as exc:  # noqa: BLE001 - report, do not crash the beat
            logger.warning("index_health_qdrant_unavailable", error=str(exc))
            qdrant_ids = set()

        pg_ids = set(chunk_repo.list_all_ids())
        orphans = sorted(qdrant_ids - pg_ids)
        never_indexed = sorted(pg_ids - qdrant_ids)
        duplicates = chunk_repo.list_duplicate_normalized_texts()

        report = {
            "qdrant_point_count": len(qdrant_ids),
            "pg_chunk_count": len(pg_ids),
            "orphan_point_count": len(orphans),
            "orphan_points": orphans[:100],
            "never_indexed_count": len(never_indexed),
            "never_indexed": never_indexed[:100],
            "duplicate_count": len(duplicates),
            "duplicates": duplicates[:100],
        }
        if orphans:
            logger.warning(
                "index_health_orphan_points",
                count=len(orphans),
                sample=orphans[:10],
            )
        if never_indexed:
            logger.warning(
                "index_health_chunks_never_indexed",
                count=len(never_indexed),
                sample=never_indexed[:10],
            )
        if duplicates:
            logger.warning(
                "index_health_duplicate_texts",
                count=len(duplicates),
                sample=duplicates[:10],
            )
        else:
            logger.info("index_health_clean", report=report)
        return report
    finally:
        session.close()
