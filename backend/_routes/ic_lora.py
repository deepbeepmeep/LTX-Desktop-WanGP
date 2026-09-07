"""Route handlers for /api/ic-lora/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/ic-lora", tags=["ic-lora"])


@router.post("/extract-conditioning", response_model=IcLoraExtractResponse)
def route_ic_lora_extract(
    req: IcLoraExtractRequest,
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraExtractResponse:
    return handler.ic_lora.extract_conditioning(req)


@router.post("/generate", response_model=IcLoraGenerateResponse)
def route_ic_lora_generate(
    req: IcLoraGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraGenerateResponse:
    return handler.ic_lora.generate(req)
