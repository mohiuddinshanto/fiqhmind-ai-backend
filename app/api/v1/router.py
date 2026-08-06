from fastapi import APIRouter

from app.api.v1.endpoints import chunking, extraction, health, layout, metadata, uploads

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(uploads.router)
api_router.include_router(extraction.router)
api_router.include_router(layout.router)
api_router.include_router(metadata.router)
api_router.include_router(chunking.router)
