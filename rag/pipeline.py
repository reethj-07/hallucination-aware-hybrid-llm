from __future__ import annotations

from api.config import Settings


def get_pipeline(settings: Settings):
    if settings.rag_lightweight:
        from rag.rag_inference_lightweight import LightweightPipeline

        return LightweightPipeline()

    from rag.rag_inference import FullPipeline

    return FullPipeline()
