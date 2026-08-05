import structlog
from qdrant_client import QdrantClient, models

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "fiqh_chunks"
DENSE_VECTOR_SIZE = 1024  # BGE-M3
DENSE_DISTANCE = models.Distance.COSINE
SPARSE_VECTOR_NAME = "text"
PAYLOAD_INDEX_FIELDS = ("book_id", "volume", "region", "verified")

DENSE_HNSW_CONFIG = models.HnswConfigDiff(m=32, ef_construct=128)


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Create a Qdrant client from application settings."""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)


class QdrantStore:
    """Owns the `fiqh_chunks` collection: idempotent creation + payload indexes."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str = COLLECTION_NAME,
        vector_size: int = DENSE_VECTOR_SIZE,
        distance: models.Distance = DENSE_DISTANCE,
    ) -> None:
        self._client = client
        self._collection = collection
        self._vector_size = vector_size
        self._distance = distance

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Create the collection (dense + sparse vectors) and payload indexes if missing."""
        if not self._client.collection_exists(self._collection):
            logger.info("qdrant_collection_creating", collection=self._collection)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    hnsw_config=DENSE_HNSW_CONFIG,
                ),
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                },
            )
        else:
            logger.info("qdrant_collection_exists", collection=self._collection)

        existing = set(self._client.get_collection(self._collection).payload_schema or {})
        for field in PAYLOAD_INDEX_FIELDS:
            if field not in existing:
                logger.info(
                    "qdrant_payload_index_creating",
                    collection=self._collection,
                    field=field,
                )
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )


_client: QdrantClient | None = None
_store: QdrantStore | None = None


def get_qdrant_client() -> QdrantClient:
    """Lazily create and cache the process-wide Qdrant client."""
    global _client
    if _client is None:
        _client = create_qdrant_client(get_settings())
    return _client


def get_qdrant_store() -> QdrantStore:
    """Lazily create and cache the process-wide collection manager."""
    global _store
    if _store is None:
        _store = QdrantStore(get_qdrant_client())
    return _store


def check_qdrant_health() -> bool:
    """Return True when Qdrant responds to a collections listing."""
    try:
        response = get_qdrant_client().get_collections()
        if response is None:
            return False
        names = {c.name for c in response.collections} if response.collections else set()
        return COLLECTION_NAME in names
    except Exception:
        logger.exception("qdrant_health_check_failed")
        return False
