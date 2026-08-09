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

    # Embedding + hybrid search (Phase 8). The vector DB layer depends on an
    # embedding interface, not the BGE-M3 model itself (ARCHITECTURE Phase 7);
    # `deterministic` is the dependency-free adapter used for tests/local runs.
    embedding_provider: str = "deterministic"  # deterministic | bge_m3 (later phase)
    # Reciprocal Rank Fusion constant (ARCHITECTURE §Phase 8: k ≈ 60).
    vector_rrf_k: int = 60

    # Retrieval pipeline (Phase 9). The fastText language head, free-tier
    # translation APIs, and the BGE-reranker-v2-m3 weights are external
    # dependencies; the pipeline ships dependency-free default adapters behind
    # the same interfaces (`heuristic`, `passthrough`, `default`) until those
    # land. `retrieval_llm_expansion_enabled` gates the external LLM paraphrase
    # step (ARCHITECTURE step 4) so it is off until a provider is configured.
    language_detector_provider: str = "heuristic"  # heuristic | fasttext (later)
    translator_provider: str = "passthrough"  # passthrough | google_free | gemini (later)
    reranker_provider: str = "default"  # default | bge_reranker_v2_m3 (later)
    retrieval_top_n: int = 8
    retrieval_candidates: int = 40
    retrieval_max_variants: int = 5
    retrieval_evidence_floor: float = 0.05
    retrieval_llm_expansion_enabled: bool = False
    retrieval_reranking_enabled: bool = True
    query_max_length: int = 500

    # LLM answer generation (Phase 10). External models sit behind a common
    # provider port (ARCHITECTURE §Phase 10 "Provider ports"); `deterministic`
    # is the dependency-free adapter that synthesizes a grounded answer from the
    # retrieved evidence with no external call, so the chat pipeline keeps
    # working when no provider key is configured.
    generator_provider: str = "deterministic"  # deterministic | gemini | groq | openrouter
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    # OpenRouter is the rotating third fallback rung only, never a primary
    # dependency (ARCHITECTURE §Phase 11). When set, a free-preview model is
    # selected and rotated automatically.
    openrouter_api_key: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    # Fallback chain order. `generator_provider` is tried first; remaining
    # providers are tried in this order as they become unhealthy/unavailable.
    provider_fallback_order: str = "gemini,groq,openrouter"
    # Context cap for evidence injected into prompts (ARCHITECTURE: 5-7 chunks).
    generation_max_chunks: int = 7
    # Number of regenerate attempts after a validation failure (validate → fail → retry once).
    generation_retries: int = 1
    # Outbound provider request timeout in seconds.
    generation_request_timeout_seconds: float = 60.0
    # Target first-delta latency budget reported in the SSE `meta` event.
    generation_latency_budget_ms: int = 3000

    # Rate limiting defaults (per Phase 14)
    rate_limit_chat_per_min: int = 20
    rate_limit_search_per_min: int = 60
    rate_limit_auth_per_min: int = 10
    rate_limit_ingest_per_min: int = 5
    rate_limit_eval_per_hour: int = 2

    # Uploads (per Phase 5.1 / Phase 3 foundation)
    upload_storage_path: str = "storage/uploads"
    upload_storage_provider: str = "local"  # local | r2 (r2 lands in a later phase)
    upload_max_size_bytes: int = 200 * 1024 * 1024  # ARCHITECTURE.md: 200 MB cap
    upload_allowed_mime: str = "application/pdf"
    upload_chunk_size: int = 64 * 1024  # streaming read chunk, never buffer whole files
    upload_max_files_per_request: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
