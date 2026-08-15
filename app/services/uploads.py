"""Upload pipeline service (Phase 5.1 / Phase 3 foundation).

Streams files chunk-by-chunk (never buffers whole files), validates magic bytes,
size, MIME and the EOF trailer, computes SHA-256 while streaming for duplicate
detection, writes through the storage provider, and creates the ingestion job.
OCR / extraction / chunking are explicitly NOT performed here — later phases.
"""

import hashlib
import uuid
from typing import BinaryIO

import structlog
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import DuplicateUploadError, UploadTooLargeError, UploadValidationError
from app.core.storage import StorageProvider
from app.db.models import Upload
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.tasks.book import start_ingestion_pipeline_task
from app.tasks.ingestion import mark_queued

logger = structlog.get_logger(__name__)

PDF_MAGIC = b"%PDF"
PDF_EOF = b"%%EOF"
EOF_SCAN_BYTES = 1024
PROGRESS_COMMIT_INTERVAL_BYTES = 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Strip path components and control characters from a client filename."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ord(ch) >= 32)
    return (name.strip() or "document").strip()[:255]


class UploadService:
    """HTTP-agnostic upload pipeline. Accepts a stream plus metadata."""

    def __init__(self, session: Session, storage: StorageProvider, settings: Settings) -> None:
        self._session = session
        self._storage = storage
        self._settings = settings

    def receive(
        self,
        stream: BinaryIO,
        *,
        original_filename: str,
        content_type: str | None,
    ) -> Upload:
        """Stream one file into storage, validate it, and create the ingestion job.

        Raises `UploadValidationError`, `UploadTooLargeError` or
        `DuplicateUploadError` on rejection (the partial row + file are cleaned up).
        """
        original_filename = sanitize_filename(original_filename)
        self._validate_mime(content_type)

        upload_id = uuid.uuid4().hex
        key = f"{upload_id}.pdf"
        repo = UploadRepository(self._session)
        upload = repo.create(
            Upload(
                id=upload_id,
                original_filename=original_filename,
                filename=key,
                storage_path=key,
                mime=self._settings.upload_allowed_mime,
                status="uploading",
            )
        )
        repo.add_log(upload, "uploading", message="receiving file")

        hasher = hashlib.sha256()
        total = 0
        saw_data = False
        chunks_since_progress = 0
        try:
            with self._storage.writer(key) as writer:
                while True:
                    chunk = stream.read(self._settings.upload_chunk_size)
                    if not chunk:
                        break
                    if not saw_data:
                        saw_data = True
                        if not chunk.startswith(PDF_MAGIC):
                            raise UploadValidationError(
                                "invalid pdf: file does not start with %PDF"
                            )
                    total += len(chunk)
                    if total > self._settings.upload_max_size_bytes:
                        raise UploadTooLargeError(
                            f"file exceeds the {self._settings.upload_max_size_bytes} byte limit"
                        )
                    hasher.update(chunk)
                    writer.write(chunk)
                    chunks_since_progress += 1
                    if chunks_since_progress * self._settings.upload_chunk_size >= (
                        PROGRESS_COMMIT_INTERVAL_BYTES
                    ):
                        repo.record_progress(upload, total)
                        chunks_since_progress = 0

            if not saw_data:
                raise UploadValidationError("empty file")
            if not self._has_eof_trailer(key):
                raise UploadValidationError("corrupted pdf: missing EOF marker")

            sha256 = hasher.hexdigest()
            existing = repo.get_by_sha256(sha256)
            if existing is not None and existing.id != upload_id:
                raise DuplicateUploadError(
                    "duplicate file already uploaded",
                    details={"existing_upload_id": existing.id},
                )

            upload.sha256 = sha256
            upload.size = total
            upload.received_bytes = total
            upload.status = "queued"
            job = IngestionJobRepository(self._session).create_for_upload(upload_id)
            repo.add_log(upload, "queued", message="accepted for ingestion")
            repo.update(upload)
        except BaseException:
            self._cleanup_upload(upload, key)
            raise

        self._dispatch_job(job.id, upload_id)
        return upload

    def delete(self, upload: Upload) -> None:
        """Remove the storage object and the upload row (with job + logs)."""
        if upload.storage_path:
            self._storage.delete(upload.storage_path)
        UploadRepository(self._session).delete_with_artifacts(upload)

    def _validate_mime(self, content_type: str | None) -> None:
        if not content_type:
            return  # magic bytes are the authoritative check
        declared = content_type.split(";", 1)[0].strip().lower()
        if declared and declared != self._settings.upload_allowed_mime:
            raise UploadValidationError(
                f"wrong mime type '{declared}': only PDF files are accepted"
            )

    def _has_eof_trailer(self, key: str) -> bool:
        try:
            size = self._storage.size(key)
            with self._storage.open(key) as handle:
                handle.seek(max(0, size - EOF_SCAN_BYTES))
                return PDF_EOF in handle.read()
        except Exception:
            logger.warning("upload_eof_check_failed", key=key)
            return False

    def _dispatch_job(self, job_id: str, upload_id: str) -> None:
        try:
            mark_queued.delay(job_id, upload_id)
        except Exception:
            # The job row exists; a later reconciliation can requeue it.
            logger.warning("ingestion_job_dispatch_failed", job_id=job_id, upload_id=upload_id)
        try:
            # Auto-ingestion: queue the full pipeline (extract → metadata →
            # chunking → embed/index) so uploads no longer need a manual stage
            # endpoint. Dispatch failures are logged; the stage-job rows created
            # by the worker task keep the upload reconcilable.
            start_ingestion_pipeline_task.delay(upload_id)
        except Exception:
            logger.warning("ingestion_pipeline_dispatch_failed", upload_id=upload_id)

    def _cleanup_upload(self, upload: Upload, key: str) -> None:
        try:
            self._storage.delete(key)
        except Exception:
            logger.warning("upload_cleanup_storage_failed", key=key)
        try:
            self._session.delete(upload)
            self._session.commit()
        except Exception:
            self._session.rollback()
            logger.warning("upload_cleanup_db_failed", upload_id=upload.id)
