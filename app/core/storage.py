"""Storage abstraction for uploaded files (Phase 5.1 / Phase 3 foundation).

`StorageProvider` is the interface. `LocalStorageProvider` is the default
implementation; a Cloudflare R2 provider (S3-compatible) can implement the
same interface later without touching business logic. Providers are pure
I/O — validation, hashing and dedupe live in the upload service.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class StorageProvider(ABC):
    """Interface every storage backend must implement (local, R2, ...)."""

    name: str = "base"

    @abstractmethod
    def writer(self, key: str) -> BinaryIO:
        """Open a binary stream for writing `key`. Creates the object."""

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Open a binary stream for reading `key`."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete `key`. Missing keys are a no-op."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True when `key` exists."""

    @abstractmethod
    def size(self, key: str) -> int:
        """Return the byte size of `key`."""

    @abstractmethod
    def resolve(self, key: str) -> str:
        """Return a native filesystem path to `key` for parser libraries.

        PyMuPDF (Phase 4) needs a real path, not a stream. Non-local
        providers should download to a temp file or raise NotImplementedError.
        """

    @abstractmethod
    def health(self) -> bool:
        """Return True when the backend is reachable and writable."""


class LocalStorageProvider(StorageProvider):
    """Filesystem-backed provider storing objects under a root directory."""

    name = "local"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    def _key_to_path(self, key: str) -> Path:
        candidate = Path(key)
        if candidate.is_absolute():
            raise ValueError("storage key must be relative")
        parts = candidate.parts
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("storage key escapes root")
        path = self._root.joinpath(*parts)
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError("storage key escapes root")
        return resolved

    def writer(self, key: str) -> BinaryIO:
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("wb")

    def open(self, key: str) -> BinaryIO:
        return self._key_to_path(key).open("rb")

    def delete(self, key: str) -> None:
        self._key_to_path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    def size(self, key: str) -> int:
        return self._key_to_path(key).stat().st_size

    def resolve(self, key: str) -> str:
        return str(self._key_to_path(key))

    def health(self) -> bool:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            probe = self._root / ".health_probe"
            probe.write_bytes(b"ok")
            healthy = probe.read_bytes() == b"ok"
            probe.unlink(missing_ok=True)
            return healthy
        except Exception:
            logger.exception("local_storage_health_check_failed")
            return False


_provider: StorageProvider | None = None


def get_storage_provider() -> StorageProvider:
    """Lazily create and cache the process-wide storage provider."""
    global _provider
    if _provider is None:
        settings: Settings = get_settings()
        if settings.upload_storage_provider != "local":
            # R2 lands in a later phase; the interface is ready (ADR-015).
            raise ValueError(
                f"unsupported upload_storage_provider: {settings.upload_storage_provider}"
            )
        _provider = LocalStorageProvider(settings.upload_storage_path)
    return _provider


def check_storage_health() -> bool:
    """Return True when the upload directory exists and is writable."""
    try:
        return get_storage_provider().health()
    except Exception:
        logger.exception("storage_health_check_failed")
        return False
