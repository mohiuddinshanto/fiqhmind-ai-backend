import structlog
from redis import Redis

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_redis: Redis | None = None


def create_redis_client(settings: Settings) -> Redis:
    """Create a Redis client from application settings."""
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def get_redis() -> Redis:
    """Lazily create and cache the process-wide Redis client."""
    global _redis
    if _redis is None:
        _redis = create_redis_client(get_settings())
    return _redis


def check_redis_health() -> bool:
    """Return True when Redis responds to PING."""
    try:
        return get_redis().ping() is True
    except Exception:
        logger.exception("redis_health_check_failed")
        return False


def _coerce_int(value: object) -> int:
    return int(value) if isinstance(value, (str, int, bytes)) else 0


def _parse_info(info: dict[bytes | str, object]) -> tuple[int, int]:
    """Extract used_memory and maxmemory from an INFO dict (keys may be bytes or str)."""
    used = 0
    maximum = 0
    for key, value in info.items():
        name = key.decode() if isinstance(key, bytes) else key
        if name == "used_memory":
            used = _coerce_int(value)
        elif name == "maxmemory":
            maximum = _coerce_int(value)
    return used, maximum


def get_redis_used_memory() -> tuple[int, int]:
    """Return (used_memory, maxmemory) bytes from Redis INFO, or (0, 0) on failure."""
    try:
        info = get_redis().info()
        if isinstance(info, dict):
            return _parse_info(info)
    except Exception:
        logger.exception("redis_info_failed")
    return 0, 0
