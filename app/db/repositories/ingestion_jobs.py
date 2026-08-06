from datetime import datetime

from sqlalchemy import select

from app.db.models import IngestionError, IngestionJob
from app.db.repositories.base import RepositoryBase

VALID_STATUSES = {
    "uploaded",
    "queued",
    "processing",
    "sanitizing",
    "extracting",
    "ocr",
    "ocr_correcting",
    "structuring",
    "chunking",
    "embedding",
    "indexed",
    "completed",
    "failed",
}


class IngestionJobRepository(RepositoryBase[IngestionJob]):
    model = IngestionJob

    def create_for_book(self, book_id: str, kind: str = "initial") -> IngestionJob:
        return self.create(IngestionJob(book_id=book_id, kind=kind, status="uploaded"))

    def create_for_upload(self, upload_id: str, kind: str = "initial") -> IngestionJob:
        return self.create(IngestionJob(upload_id=upload_id, kind=kind, status="uploaded"))

    def find_for_upload(self, upload_id: str, kind: str) -> IngestionJob | None:
        """Most recent job of `kind` for an upload, if any."""
        return self._session.scalar(
            select(IngestionJob)
            .where(IngestionJob.upload_id == upload_id, IngestionJob.kind == kind)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )

    def find_pipeline_job(self, upload_id: str) -> IngestionJob | None:
        """Most recent pipeline (non-analysis) job for an upload, if any.

        Excludes the `layout` and `metadata` analysis kinds so callers (layout
        and metadata endpoints) always find the underlying extraction job.
        """
        return self._session.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.upload_id == upload_id,
                IngestionJob.kind.notin_(("layout", "metadata")),
            )
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )

    def add_error(
        self,
        job: IngestionJob,
        step: str,
        message: str,
        *,
        details: dict | None = None,
    ) -> IngestionError:
        error = IngestionError(job_id=job.id, step=step, message=message, details=details)
        self._session.add(error)
        self._session.commit()
        return error

    def update_status(
        self,
        job: IngestionJob,
        status: str,
        *,
        progress_percent: int | None = None,
        current_step: str | None = None,
        error_message: str | None = None,
    ) -> IngestionJob:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid ingestion status: {status}")
        job.status = status
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if current_step is not None:
            job.current_step = current_step
        if error_message is not None:
            job.error_message = error_message
        if status in ("indexed", "completed", "failed"):
            job.finished_at = datetime.utcnow()
        return self.update(job)

    def list_active(self, *, limit: int = 50) -> list[IngestionJob]:
        return list(
            self._session.scalars(
                select(IngestionJob)
                .where(IngestionJob.status.not_in(("indexed", "failed")))
                .order_by(IngestionJob.created_at.desc())
                .limit(limit)
            )
        )
