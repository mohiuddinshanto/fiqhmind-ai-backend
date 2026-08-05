from datetime import datetime

from sqlalchemy import select

from app.db.models import IngestionJob
from app.db.repositories.base import RepositoryBase

VALID_STATUSES = {
    "uploaded",
    "sanitizing",
    "extracting",
    "ocr",
    "ocr_correcting",
    "structuring",
    "chunking",
    "embedding",
    "indexed",
    "failed",
}


class IngestionJobRepository(RepositoryBase[IngestionJob]):
    model = IngestionJob

    def create_for_book(self, book_id: str, kind: str = "initial") -> IngestionJob:
        return self.create(IngestionJob(book_id=book_id, kind=kind, status="uploaded"))

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
        if status in ("indexed", "failed"):
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
