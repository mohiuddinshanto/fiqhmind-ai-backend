from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.core.postgres import check_postgres_health
from app.core.qdrant import check_qdrant_health
from app.core.redis import check_redis_health
from app.core.storage import check_storage_health
from app.schemas.health import HealthResponse, ServiceHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    components: dict[str, ServiceHealth] = {
        "postgres": ServiceHealth(status="ok" if check_postgres_health() else "degraded"),
        "redis": ServiceHealth(status="ok" if check_redis_health() else "degraded"),
        "qdrant": ServiceHealth(status="ok" if check_qdrant_health() else "degraded"),
        "storage": ServiceHealth(status="ok" if check_storage_health() else "degraded"),
    }
    all_ok = all(component.status == "ok" for component in components.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        service=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        components=components,
    )
