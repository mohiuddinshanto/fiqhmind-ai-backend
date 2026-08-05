from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FiqhMind AI Backend"
    version: str = "0.1.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    frontend_origin: str = "http://localhost:3000"

    database_url: str = "postgresql+psycopg://fiqhmind:fiqhmind@localhost:5432/fiqhmind"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
