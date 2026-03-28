from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rag_lightweight: bool = True
    rag_rerank: bool = False
    rag_skip_chunking: bool = False
    rag_max_context_chars: int = 4000
    rag_top_k: int = 5
    rag_index_path: Path = Path("rag/faiss_index/index.faiss")
    rag_docs_path: Path = Path("rag/faiss_index/docs.pkl")

    faithfulness_threshold: float = 0.4
    use_nli_guard: bool = True

    request_timeout_s: int = 20
    api_key: str = ""
    rate_limit: str = "60/minute"
    cors_allow_origins: str = "*"
    log_level: str = "INFO"
    port: int = 8000

    rag_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    lora_path: Path = Path("models/phi3_lora_final")

    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "rag-evals"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
