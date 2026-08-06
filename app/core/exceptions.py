class BaseAppError(Exception):
    """Base class for all application errors, mapped to typed HTTP responses."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(BaseAppError):
    status_code = 404
    code = "not_found"


class ValidationError(BaseAppError):
    status_code = 422
    code = "validation_error"


class AuthError(BaseAppError):
    status_code = 401
    code = "authentication_error"


class RateLimitError(BaseAppError):
    status_code = 429
    code = "rate_limit_exceeded"


class ProviderError(BaseAppError):
    status_code = 502
    code = "upstream_provider_error"


class IngestionError(BaseAppError):
    status_code = 500
    code = "ingestion_error"


class EvidenceInsufficientError(BaseAppError):
    status_code = 422
    code = "evidence_insufficient"


class UploadValidationError(BaseAppError):
    status_code = 422
    code = "upload_validation_error"


class UploadTooLargeError(BaseAppError):
    status_code = 413
    code = "upload_too_large"


class DuplicateUploadError(BaseAppError):
    status_code = 409
    code = "duplicate_upload"


class ExtractionConflictError(BaseAppError):
    status_code = 409
    code = "extraction_conflict"


class LayoutConflictError(BaseAppError):
    status_code = 409
    code = "layout_conflict"


class MalformedPdfError(BaseAppError):
    status_code = 422
    code = "malformed_pdf"


class EncryptedPdfError(BaseAppError):
    status_code = 422
    code = "encrypted_pdf"


class TransientExtractionError(BaseAppError):
    status_code = 500
    code = "extraction_retry"
