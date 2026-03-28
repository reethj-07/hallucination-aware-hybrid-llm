from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from api.config import Settings, get_settings


def get_rag_pipeline(settings: Settings = Depends(get_settings)):
    from rag.pipeline import get_pipeline

    return get_pipeline(settings)


async def verify_api_key(
    x_api_key: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
