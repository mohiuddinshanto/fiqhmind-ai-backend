from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import models

from app.core import qdrant as qdrant_module
from app.core.qdrant import QdrantStore, check_qdrant_health


def _mock_client(exists: bool = False, payload_schema: dict | None = None) -> Mock:
    client = Mock()
    client.collection_exists.return_value = exists
    info = Mock()
    info.payload_schema = payload_schema or {}
    client.get_collection.return_value = info
    return client


def test_ensure_collection_creates_when_missing() -> None:
    client = _mock_client(exists=False)
    store = QdrantStore(client, vector_size=1024)

    store.ensure_collection()

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "fiqh_chunks"
    vectors = kwargs["vectors_config"]
    assert vectors.size == 1024
    assert vectors.distance == models.Distance.COSINE
    assert "text" in kwargs["sparse_vectors_config"]
    assert client.create_payload_index.call_count == len(qdrant_module.PAYLOAD_INDEX_FIELDS)


def test_ensure_collection_skips_existing_and_indexes_present() -> None:
    existing = {field: Mock() for field in qdrant_module.PAYLOAD_INDEX_FIELDS}
    client = _mock_client(exists=True, payload_schema=existing)
    store = QdrantStore(client)

    store.ensure_collection()

    client.create_collection.assert_not_called()
    client.create_payload_index.assert_not_called()


def test_ensure_collection_adds_missing_indexes() -> None:
    client = _mock_client(exists=True, payload_schema={"book_id": Mock()})
    store = QdrantStore(client)

    store.ensure_collection()

    assert client.create_payload_index.call_count == len(qdrant_module.PAYLOAD_INDEX_FIELDS) - 1
    created = [call.kwargs["field_name"] for call in client.create_payload_index.call_args_list]
    assert "book_id" not in created


def test_collection_uses_documented_payload_index_fields() -> None:
    assert set(qdrant_module.PAYLOAD_INDEX_FIELDS) == {
        "book_id",
        "volume",
        "region",
        "verified",
        "upload_id",
    }


def test_check_qdrant_health(monkeypatch) -> None:
    client = Mock()
    response = Mock()
    response.collections = [SimpleNamespace(name="fiqh_chunks"), SimpleNamespace(name="other")]
    client.get_collections.return_value = response
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

    assert check_qdrant_health() is True


def test_check_qdrant_health_missing_collection(monkeypatch) -> None:
    client = Mock()
    response = Mock()
    response.collections = [SimpleNamespace(name="other")]
    client.get_collections.return_value = response
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

    assert check_qdrant_health() is False


def test_check_qdrant_health_on_error(monkeypatch) -> None:
    client = Mock()
    client.get_collections.side_effect = RuntimeError("connection refused")
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: client)

    assert check_qdrant_health() is False


@pytest.mark.parametrize("distance", [models.Distance.COSINE])
def test_store_exposes_client_and_collection(distance) -> None:
    store = QdrantStore(_mock_client(), collection="fiqh_chunks", distance=distance)
    assert store.collection == "fiqh_chunks"
    assert store.client is not None
