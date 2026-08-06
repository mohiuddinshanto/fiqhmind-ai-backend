from fastapi.testclient import TestClient

import app.api.v1.endpoints.health as health_module
from app.main import app

client = TestClient(app)


def test_health_returns_ok_when_all_components_up(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "check_postgres_health", lambda: True)
    monkeypatch.setattr(health_module, "check_redis_health", lambda: True)
    monkeypatch.setattr(health_module, "check_qdrant_health", lambda: True)
    monkeypatch.setattr(health_module, "check_storage_health", lambda: True)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "FiqhMind AI Backend"
    assert "version" in payload
    assert payload["components"]["postgres"]["status"] == "ok"
    assert payload["components"]["redis"]["status"] == "ok"
    assert payload["components"]["qdrant"]["status"] == "ok"
    assert payload["components"]["storage"]["status"] == "ok"


def test_health_returns_503_when_component_degraded(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "check_postgres_health", lambda: True)
    monkeypatch.setattr(health_module, "check_redis_health", lambda: True)
    monkeypatch.setattr(health_module, "check_qdrant_health", lambda: True)
    monkeypatch.setattr(health_module, "check_storage_health", lambda: False)

    response = client.get("/api/v1/health")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["components"]["postgres"]["status"] == "ok"
    assert payload["components"]["storage"]["status"] == "degraded"


def test_health_echoes_request_id(monkeypatch) -> None:
    monkeypatch.setattr(health_module, "check_postgres_health", lambda: True)
    monkeypatch.setattr(health_module, "check_redis_health", lambda: True)
    monkeypatch.setattr(health_module, "check_qdrant_health", lambda: True)
    monkeypatch.setattr(health_module, "check_storage_health", lambda: True)

    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-request-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-request-123"
