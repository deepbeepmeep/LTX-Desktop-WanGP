"""Route handlers for /api/generate, /api/generate/cancel, /api/generation/progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    CancelResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerationProgressResponse,
    LtxInsufficientFundsErrorResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["generation"])


@router.post(
    "/generate",
    response_model=GenerateVideoResponse,
    responses={
        402: {
            "model": LtxInsufficientFundsErrorResponse,
            "description": "LTX API credits are insufficient for the requested generation",
        },
    },
)
def route_generate(
    req: GenerateVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoResponse:
    """POST /api/generate — video generation from JSON body."""
    return handler.video_generation.generate(req)


@router.get("/generate/models-specs", response_model=GenerateVideoModelsSpecsResponse)
def route_generate_model_specs(
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoModelsSpecsResponse:
    """GET /api/generate/models-specs."""
    return handler.video_generation.get_model_specs()


@router.post("/generate/cancel", response_model=CancelResponse)
def route_generate_cancel(handler: AppHandler = Depends(get_state_service)) -> CancelResponse:
    """POST /api/generate/cancel."""
    return handler.generation.cancel_generation()


@router.get("/generation/progress", response_model=GenerationProgressResponse)
def route_generation_progress(handler: AppHandler = Depends(get_state_service)) -> GenerationProgressResponse:
    """GET /api/generation/progress."""
    return handler.generation.get_generation_progress()
