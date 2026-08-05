from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FiqhMind AI Backend"
    version: str = "0.1.0"
    environment: str = Field(
        default="development", pattern="^(development|staging|production|testing)$"
    )
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    frontend_origin: str = "http://localhost:3000"

    # PostgreSQL — sync SQLAlchemy engine (ADR-006)
    database_url: str = "postgresql+psycopg://fiqhmind:fiqhmind@localhost:5432/fiqhmind"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_recycle: int = 3600

    # Redis — cache, rate limits, Celery broker/backend
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Qdrant — vector store for chunks
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "fiqh_chunks"
    qdrant_vector_size: int = 1024  # BGE-M3 dense dimension
    qdrant_vector_distance: str = "Cosine"

    # Rate limiting defaults (per Phase 14)
    rate_limit_chat_per_min: int = 20
    rate_limit_search_per_min: int = 60
    rate_limit_auth_per_min: int = 10
    rate_limit_ingest_per_min: int = 5
    rate_limit_eval_per_hour: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
