"""
src/config.py
=============
Centralised configuration loader.

Reads config/config.yaml for structural settings and overlays environment
variables (from .env) for secrets.  All other modules import `get_settings()`
and never read env-vars or yaml directly — single source of truth.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Resolve project root (works from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_YAML  = PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml() -> dict:
    """Load config.yaml once and return as plain dict."""
    if not CONFIG_YAML.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_YAML}")
    with CONFIG_YAML.open("r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Pydantic Settings — secrets from env, defaults from yaml
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    All configuration in one place.
    Env-vars override defaults (pydantic-settings behaviour).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # --- LLM Provider secrets ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str   = Field(default="", alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # --- Qdrant ---
    qdrant_url: str     = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    # --- Langfuse ---
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str       = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    # --- LangSmith (optional) ---
    langchain_api_key: str      = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool  = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str      = Field(default="fin-rag-analyst", alias="LANGCHAIN_PROJECT")

    # --- Postgres ---
    postgres_url: str = Field(
        default="postgresql://langfuse:langfuse_secret@localhost:5432/langfuse",
        alias="POSTGRES_URL",
    )

    # --- Derived from yaml (populated in __init__) ---
    _yaml: dict = {}

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        self._yaml = _load_yaml()

    # -----------------------------------------------------------------------
    # Convenience accessors (read from yaml section)
    # -----------------------------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return self._yaml["llm"]["active"]

    @property
    def llm_config(self) -> dict:
        return self._yaml["llm"][self.llm_provider]

    @property
    def embedding_dense(self) -> dict:
        return self._yaml["embeddings"]["dense"]

    @property
    def embedding_sparse(self) -> dict:
        return self._yaml["embeddings"]["sparse"]

    @property
    def qdrant_config(self) -> dict:
        cfg = self._yaml["qdrant"].copy()
        # allow env-var override of URL
        cfg["url"] = self.qdrant_url
        if self.qdrant_api_key:
            cfg["api_key"] = self.qdrant_api_key
        return cfg

    @property
    def ingestion_config(self) -> dict:
        return self._yaml["ingestion"]

    @property
    def observability_config(self) -> dict:
        return self._yaml["observability"]

    @property
    def guardrails_config(self) -> dict:
        return self._yaml["guardrails"]

    @property
    def graph_config(self) -> dict:
        return self._yaml["graph"]

    @property
    def eval_config(self) -> dict:
        return self._yaml["evaluation"]

    @property
    def ui_config(self) -> dict:
        return self._yaml["ui"]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


# ---------------------------------------------------------------------------
# Cached singleton — import and call get_settings() anywhere
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    logger.info(
        f"Configuration loaded | env={settings.environment} "
        f"| llm={settings.llm_provider} "
        f"| qdrant={settings.qdrant_url}"
    )
    return settings