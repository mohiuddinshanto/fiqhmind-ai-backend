from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "FiqhMind AI Backend"
    assert "version" in payload


def test_health_echoes_request_id() -> None:
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-request-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-request-123"
