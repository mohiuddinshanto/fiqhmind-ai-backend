from collections.abc import Generator

import structlog
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_engine_for(settings: Settings) -> Engine:
    """Create a synchronous SQLAlchemy engine from application settings."""
    connect_args: dict[str, object] = {}
    if not settings.database_url.startswith("sqlite"):
        connect_args = {"connect_timeout": 3}
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def get_engine() -> Engine:
    """Lazily create and cache the process-wide SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine_for(get_settings())
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Lazily create and cache the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and always closes it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def check_postgres_health() -> bool:
    """Return True when a `SELECT 1` against the database succeeds."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("postgres_health_check_failed")
        return False
