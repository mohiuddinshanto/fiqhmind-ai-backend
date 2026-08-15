from sqlalchemy import func, select

from app.db.models import Upload, UploadLog
from app.db.repositories.base import RepositoryBase


class UploadRepository(RepositoryBase[Upload]):
    model = Upload

    def get_by_sha256(self, sha256: str) -> Upload | None:
        return self._session.scalar(select(Upload).where(Upload.sha256 == sha256))

    def list(self, *, skip: int = 0, limit: int = 100) -> list[Upload]:
        return list(
            self._session.scalars(
                select(Upload)
                .order_by(Upload.created_at.desc(), Upload.id.desc())
                .offset(skip)
                .limit(limit)
            )
        )

    def count(self) -> int:
        return int(self._session.scalar(select(func.count()).select_from(Upload)) or 0)

    def record_progress(self, upload: Upload, received_bytes: int) -> None:
        upload.received_bytes = received_bytes
        self._session.commit()

    def add_log(
        self,
        upload: Upload,
        event: str,
        *,
        message: str | None = None,
        details: dict | None = None,
    ) -> UploadLog:
        log = UploadLog(upload_id=upload.id, event=event, message=message, details=details)
        self._session.add(log)
        return log

    def delete_with_artifacts(self, upload: Upload) -> None:
        """Delete the upload row plus its ingestion jobs (and job artifacts) and logs."""
        for job in list(upload.ingestion_jobs):
            self._session.delete(job)
        self._session.delete(upload)
        self._session.commit()
