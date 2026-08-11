from celery import Celery
from celery.schedules import crontab

from app.core.config import Settings, get_settings


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Build the Celery application (broker/backend Redis) with per-pipeline queues."""
    resolved = settings or get_settings()
    app = Celery(
        "fiqhmind",
        broker=resolved.celery_broker_url,
        backend=resolved.celery_result_backend,
    )
    app.conf.update(
        timezone="UTC",
        enable_utc=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_default_queue="ingest",
        task_routes={
            "app.tasks.ingestion.extract_*": {"queue": "extract"},
            "app.tasks.ingestion.embed_*": {"queue": "embed"},
            "app.tasks.ingestion.index_*": {"queue": "index"},
        },
        beat_schedule={
            "daily-cache-eviction": {
                "task": "app.tasks.maintenance.evict_caches",
                "schedule": crontab(hour=3, minute=0),
            },
        },
        broker_connection_retry_on_startup=True,
    )
    return app


celery_app = create_celery_app()
