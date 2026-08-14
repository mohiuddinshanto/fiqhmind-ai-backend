"""Settings loading and external-service URL derivation.

The `Settings` model auto-loads `backend/.env` (which is gitignored), so these
tests instantiate the model with a nonexistent `_env_file` to isolate them from
whatever a developer has in their local `.env`. Explicit values (passed as
kwargs) always take precedence over derived ones.
"""

from app.core.config import (
    Settings,
    _normalize_database_url,
    _qdrant_url_from_host,
    _upstash_host,
    _upstash_redis_url,
)


def _settings(**kwargs) -> Settings:
    return Settings(_env_file="tests/test_config_no_such_env_file.env", **kwargs)


def test_defaults_are_local():
    settings = Settings(_env_file="tests/test_config_no_such_env_file.env")
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.celery_broker_url == "redis://localhost:6379/1"
    assert settings.celery_result_backend == "redis://localhost:6379/2"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.upstash_redis_rest_url is None


def test_qdrant_host_bare_hostname_is_normalised():
    settings = _settings(qdrant_host="cluster.qdrant.io")
    assert settings.qdrant_url == "https://cluster.qdrant.io:6333"


def test_qdrant_host_full_url_is_not_doubled():
    settings = _settings(qdrant_host="https://cluster.qdrant.io:6333")
    assert settings.qdrant_url == "https://cluster.qdrant.io:6333"


def test_explicit_qdrant_url_wins_over_qdrant_host():
    settings = _settings(qdrant_host="cluster.qdrant.io", qdrant_url="http://localhost:6333")
    assert settings.qdrant_url == "http://localhost:6333"


def test_upstash_credentials_derive_single_redis_url():
    settings = _settings(
        upstash_redis_rest_url="https://us1-demo.upstash.io",
        upstash_redis_rest_token="top-secret",
    )
    expected = "rediss://default:top-secret@us1-demo.upstash.io:6379/0?ssl_cert_reqs=CERT_REQUIRED"
    assert settings.redis_url == expected
    assert settings.celery_broker_url == expected
    assert settings.celery_result_backend == expected


def test_explicit_redis_url_wins_over_upstash_derivation():
    settings = _settings(
        upstash_redis_rest_url="https://us1-demo.upstash.io",
        upstash_redis_rest_token="top-secret",
        redis_url="redis://localhost:6379/9",
    )
    assert settings.redis_url == "redis://localhost:6379/9"
    assert settings.celery_broker_url == "redis://localhost:6379/1"
    assert settings.celery_result_backend == "redis://localhost:6379/2"


def test_upstash_token_is_url_encoded():
    settings = _settings(
        upstash_redis_rest_url="https://us1-demo.upstash.io",
        upstash_redis_rest_token="a/b@c:d",
    )
    assert settings.redis_url == "rediss://default:a%2Fb%40c%3Ad@us1-demo.upstash.io:6379/0?ssl_cert_reqs=CERT_REQUIRED"


def test_database_url_bare_postgres_is_normalised():
    settings = _settings(database_url="postgres://user:pass@db.example.com/fiqhmind")
    assert settings.database_url == "postgresql+psycopg://user:pass@db.example.com/fiqhmind"
    settings = _settings(database_url="postgresql://user:pass@db.example.com/fiqhmind")
    assert settings.database_url == "postgresql+psycopg://user:pass@db.example.com/fiqhmind"


def test_database_url_psycopg_dialect_is_unchanged():
    url = "postgresql+psycopg://user:pass@localhost:5432/fiqhmind"
    assert _settings(database_url=url).database_url == url


def test_upstash_host_parsing():
    assert _upstash_host("https://us1-demo.upstash.io") == "us1-demo.upstash.io"
    assert _upstash_host("https://us1-demo.upstash.io?foo=bar") == "us1-demo.upstash.io"
    assert _upstash_host("http://us1-demo.upstash.io/path") == "us1-demo.upstash.io"


def test_upstash_redis_url_helper():
    url = _upstash_redis_url("https://us1-demo.upstash.io", "tok")
    assert url == "rediss://default:tok@us1-demo.upstash.io:6379/0?ssl_cert_reqs=CERT_REQUIRED"


def test_qdrant_url_from_host_helper():
    assert _qdrant_url_from_host("cluster.qdrant.io") == "https://cluster.qdrant.io:6333"
    assert _qdrant_url_from_host("https://cluster.qdrant.io:6333") == "https://cluster.qdrant.io:6333"


def test_normalize_database_url_helper():
    assert _normalize_database_url("postgres://h/db") == "postgresql+psycopg://h/db"
    assert _normalize_database_url("postgresql://h/db") == "postgresql+psycopg://h/db"
    assert _normalize_database_url("postgresql+psycopg://h/db") == "postgresql+psycopg://h/db"
