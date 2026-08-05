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
