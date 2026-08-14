from functools import lru_cache
from urllib.parse import quote

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_database_url(url: str) -> str:
    """Force the psycopg (v3) driver onto a bare postgres/postgresql URL.

    The project depends on `psycopg[binary]`, not psycopg2, so a supplied
    `postgres://...` / `postgresql://...` URL must be rewritten to
    `postgresql+psycopg://...` or SQLAlchemy will try the missing psycopg2
    dialect at engine creation time.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def _qdrant_url_from_host(host: str) -> str:
    """Build a `QdrantClient(url=...)` value from QDRANT_HOST.

    The client expects a full URL (scheme + host + port). Accept either a full
    URL (`https://cluster.qdrant.io:6333`) or a bare hostname (`cluster.qdrant.io`),
    normalising the latter to `https://<host>:6333` without ever doubling a scheme.
    """
    host = host.strip()
    if "://" in host:
        return host
    return f"https://{host}:6333"


def _upstash_host(rest_url: str) -> str:
    """Extract the RES/RESP hostname from an Upstash REST URL.

    `https://us1-abc.upstash.io` (optionally with a path) -> `us1-abc.upstash.io`.
    """
    host = rest_url.strip()
    for prefix in ("https://", "http://"):
        if host.lower().startswith(prefix):
            host = host[len(prefix) :]
    host = host.split("/", 1)[0].split("?", 1)[0]
    host = host.rsplit("@", 1)[-1]
    head, _, tail = host.rpartition(":")
    if head and tail.isdigit():
        host = head
    return host


def _upstash_redis_url(rest_url: str, token: str) -> str:
    """Build a redis-py compatible URL from Upstash REST credentials.

    Upstash speaks the standard Redis protocol over TLS, so the existing
    redis-py architecture is reused unchanged. Username is `default`, the REST
    token is the password, and Upstash exposes a single logical database, so
    every consumer (cache/rate-limit/broker/result) shares db 0. The explicit
    `ssl_cert_reqs=CERT_REQUIRED` query param is required by Celery's redis
    result backend, which refuses a `rediss://` URL without it.
    """
    host = _upstash_host(rest_url)
    return (
        f"rediss://default:{quote(token, safe='')}@{host}:6379/0"
        "?ssl_cert_reqs=CERT_REQUIRED"
    )


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

    # Optional Upstash Redis. When set (and REDIS_URL is NOT explicitly set),
    # the REST URL + token are converted to a single redis-py `rediss://` URL
    # that backs cache, rate limits, and the Celery broker/result store alike
    # (Upstash exposes one logical database, so every consumer shares db 0).
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None

    # Qdrant — vector store for chunks. QDRANT_HOST accepts either a full URL
    # (`https://cluster.qdrant.io:6333`) or a bare hostname (normalised to
    # `https://<host>:6333`). A QDRANT_URL set explicitly always wins.
    qdrant_host: str | None = None
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
    # Sanitize LLM-generated `explanation.html` through the allowlist filter
    # before it is stored/streamed. The frontend renders this field via
    # `dangerouslySetInnerHTML`, so it must never carry scripts/event handlers
    # (app.services.html_sanitizer). Deterministic synthesis is unaffected.
    generation_sanitize_html: bool = True
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

    # Phase 15 caching (Redis-backed). The cache layer is best-effort: when Redis
    # is unavailable the application degrades to the uncached path, so these TTLs
    # only bound how long cached entries live (ARCHITECTURE §Phase 15 "Caching").
    cache_qa_ttl_seconds: int = 86400  # QA answer cache — 24h, evicted daily by beat
    cache_embedding_ttl_seconds: int = 604800  # embedding cache — 7d, content-addressed
    cache_retrieval_ttl_seconds: int = 900  # chunk-level results cache — 15 min hot queries
    # Batched embedding per ARCHITECTURE §Phase 15 "Batch processing (ingestion)".
    embedding_batch_size: int = 128

    # Uploads (per Phase 5.1 / Phase 3 foundation)
    upload_storage_path: str = "storage/uploads"
    upload_storage_provider: str = "local"  # local | r2 (r2 lands in a later phase)
    upload_max_size_bytes: int = 200 * 1024 * 1024  # ARCHITECTURE.md: 200 MB cap
    upload_max_pages: int = 2000  # decompression-bomb guard: reject oversized PDFs
    # Page-level parallel extraction (Phase 15 §713): worker threads each open
    # their own PyMuPDF document and parse a contiguous slice of pages.
    extraction_workers: int = 4
    # Checkpoint granularity for crash-resumable extraction (Phase 15 §Batch
    # processing): page rows are committed to Postgres every N pages, so a
    # failed run resumes mid-book from the last durable checkpoint.
    extraction_checkpoint_pages: int = 25
    upload_allowed_mime: str = "application/pdf"
    upload_chunk_size: int = 64 * 1024  # streaming read chunk, never buffer whole files
    upload_max_files_per_request: int = 20

    @model_validator(mode="after")
    def _derive_external_urls(self) -> "Settings":
        """Normalise external-service URLs after env/.env loading.

        Every field keeps its default (localhost) when no external value is
        supplied, so local development and the existing test suite behave
        exactly as before. Explicit values (already in `model_fields_set`)
        always take precedence over derived ones.
        """
        self.database_url = _normalize_database_url(self.database_url)

        if self.qdrant_host and "qdrant_url" not in self.model_fields_set:
            self.qdrant_url = _qdrant_url_from_host(self.qdrant_host)

        if (
            self.upstash_redis_rest_url
            and self.upstash_redis_rest_token
            and "redis_url" not in self.model_fields_set
        ):
            derived = _upstash_redis_url(
                self.upstash_redis_rest_url, self.upstash_redis_rest_token
            )
            self.redis_url = derived
            self.celery_broker_url = derived
            self.celery_result_backend = derived

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
