"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # One provider ("gemini"), an ordered failover list ("gemini,openrouter"), or "auto": every provider
    # that has a key, in the order gemini, openrouter, openai_compat, anthropic (mock when none has a key).
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"))
    anthropic_refusal_fallback: bool = field(default_factory=lambda: _bool("ANTHROPIC_REFUSAL_FALLBACK", True))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    gemini_fallback_models: List[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest").split(",") if m.strip()])
    # OpenRouter (hosted, has free ":free" models; 50 requests/day without purchased credits).
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_models: List[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv(
            "OPENROUTER_MODELS",
            # Tried in order; openrouter/free is OpenRouter's router over whatever free model is available.
            # Checked live with "Test every model": Nemotron 3 Ultra (too slow), Inkling (agent-only) and
            # Nemotron 3 Super (provider 404) were dropped.
            "google/gemma-4-31b-it:free,nvidia/nemotron-3.5-super-vl:free,dots-studio/dots-3-note-preview:free,"
            "google/gemma-4-26b-a4b-it:free,openrouter/free"
        ).split(",") if m.strip()])
    # Any other OpenAI-compatible chat API (Groq, Together, a vLLM server, ...).
    openai_compat_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_BASE_URL", ""))
    openai_compat_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_API_KEY", ""))
    openai_compat_models: List[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv("OPENAI_COMPAT_MODELS", "").split(",") if m.strip()])
    # Per-request timeout for OpenAI-compatible providers: short, so a stuck free model is skipped quickly.
    openai_compat_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("OPENAI_COMPAT_TIMEOUT_SECONDS", "60")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    llm_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "180")))
    # Per-agent effort: cheap extraction, deeper writing, balanced review.
    context_effort: str = field(default_factory=lambda: os.getenv("CONTEXT_EFFORT", "low"))
    writer_effort: str = field(default_factory=lambda: os.getenv("WRITER_EFFORT", "high"))
    reviewer_effort: str = field(default_factory=lambda: os.getenv("REVIEWER_EFFORT", "medium"))

    vector_backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "memory"))
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    milvus_token: str = field(default_factory=lambda: os.getenv("MILVUS_TOKEN", ""))
    milvus_collection: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "resume_knowledge"))
    # If Milvus can't be reached at start-up, use the in-memory store instead of failing. Off by default
    # (Docker/Kubernetes should fail loudly); the single-container Streamlit deployment turns it on.
    vector_fallback_to_memory: bool = field(default_factory=lambda: _bool("VECTOR_FALLBACK_TO_MEMORY", False))
    # "auto" uses Gemini's hosted embedding API when GEMINI_API_KEY is set, else the local hashing embedder.
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    gemini_embedding_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    gemini_embedding_dim: int = field(default_factory=lambda: int(os.getenv("GEMINI_EMBEDDING_DIM", "768")))
    # Seed the knowledge base at startup when empty. Kubernetes disables this and runs a one-off Job instead,
    # so concurrently starting replicas don't race to insert duplicate seed rows.
    seed_on_startup: bool = field(default_factory=lambda: _bool("SEED_ON_STARTUP", True))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "6")))

    checkpoint_backend: str = field(default_factory=lambda: os.getenv("CHECKPOINT_BACKEND", "memory"))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "./data/checkpoints.sqlite"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

    quality_threshold: float = field(default_factory=lambda: float(os.getenv("QUALITY_THRESHOLD", "8.0")))
    max_auto_revisions: int = field(default_factory=lambda: int(os.getenv("MAX_AUTO_REVISIONS", "2")))

    api_auth_token: str = field(default_factory=lambda: os.getenv("API_AUTH_TOKEN", ""))
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
