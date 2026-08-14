import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import BaseAppError
from app.core.logging import configure_logging
from app.middleware.request_context import RequestContextMiddleware
from app.services.generation.providers import ProviderHealthService

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        health = ProviderHealthService(settings)
        task = await health.start_health_checks()
        logger.info("provider_health_checks_started")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    application = FastAPI(
        title=settings.app_name,
        version=settings.version,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestContextMiddleware)

    application.include_router(api_router, prefix=settings.api_v1_prefix)

    @application.exception_handler(BaseAppError)
    async def app_error_handler(request: Request, exc: BaseAppError) -> JSONResponse:
        structlog.get_logger("error").warning(
            "application_error", code=exc.code, message=exc.message, details=exc.details
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger("error").exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    return application


app = create_app()
