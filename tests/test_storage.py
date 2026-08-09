import pytest

from app.core.storage import LocalStorageProvider
from tests.support import make_pdf_bytes


@pytest.fixture()
def storage(tmp_path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "uploads")


def test_write_read_roundtrip(storage: LocalStorageProvider) -> None:
    key = "abc123.pdf"
    data = make_pdf_bytes()
    with storage.writer(key) as writer:
        writer.write(data)
    assert storage.exists(key)
    assert storage.size(key) == len(data)
    with storage.open(key) as handle:
        assert handle.read() == data


def test_streaming_multiple_writes(storage: LocalStorageProvider) -> None:
    key = "stream.pdf"
    with storage.writer(key) as writer:
        writer.write(b"%PDF-1.4\n")
        writer.write(b"some body content\n")
        writer.write(b"%%EOF\n")
    assert storage.size(key) == len(b"%PDF-1.4\nsome body content\n%%EOF\n")


def test_delete_and_missing_noop(storage: LocalStorageProvider) -> None:
    key = "gone.pdf"
    with storage.writer(key) as writer:
        writer.write(make_pdf_bytes())
    storage.delete(key)
    assert not storage.exists(key)
    storage.delete(key)  # deleting a missing key must not raise


def test_health_creates_root(tmp_path) -> None:
    provider = LocalStorageProvider(tmp_path / "nested" / "uploads")
    assert provider.health() is True
    assert (tmp_path / "nested" / "uploads").is_dir()


def test_path_traversal_rejected(storage: LocalStorageProvider) -> None:
    with pytest.raises(ValueError):
        storage.writer("../escape.pdf")
    with pytest.raises(ValueError):
        storage.writer("a/../../escape.pdf")
    with pytest.raises(ValueError):
        storage.writer("C:/Windows/escape.pdf")


def test_keys_cannot_escape_root(tmp_path) -> None:
    provider = LocalStorageProvider(tmp_path / "uploads")
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError):
        provider.open("..\\elsewhere\\evil.pdf")
