"""
src/llm_factory.py
==================
LLM Factory — returns a LangChain-compatible ChatModel based on the active
provider in config.yaml.  Supports Gemini, Groq, and Ollama with zero
code changes between providers.

Usage:
    from src.llm_factory import get_llm
    llm = get_llm()
    response = llm.invoke("Summarise Apple Q4 2024 earnings.")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from loguru import logger

from src.config import get_settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def _build_gemini(cfg: dict, api_key: str) -> "BaseChatModel":
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=cfg["model"],
        google_api_key=api_key,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_tokens"],
        convert_system_message_to_human=True,
    )


def _build_groq(cfg: dict, api_key: str) -> "BaseChatModel":
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=cfg["model"],
        groq_api_key=api_key,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


def _build_ollama(cfg: dict) -> "BaseChatModel":
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=cfg["model"],
        base_url=cfg["base_url"],
        temperature=cfg["temperature"],
        num_predict=cfg["max_tokens"],
    )


def get_llm(with_langfuse: bool = True) -> "BaseChatModel":
    """
    Return the active ChatModel with optional Langfuse tracing callback.

    Args:
        with_langfuse: Attach LangfuseCallbackHandler when True and
                       Langfuse is enabled in config.

    Returns:
        Configured BaseChatModel instance.
    """
    settings = get_settings()
    provider  = settings.llm_provider
    cfg       = settings.llm_config

    logger.info(f"Building LLM | provider={provider} | model={cfg['model']}")

    match provider:
        case "gemini":
            if not settings.gemini_api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. "
                    "Export it or add it to .env"
                )
            llm = _build_gemini(cfg, settings.gemini_api_key)

        case "groq":
            if not settings.groq_api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. "
                    "Export it or add it to .env"
                )
            llm = _build_groq(cfg, settings.groq_api_key)

        case "ollama":
            llm = _build_ollama(cfg)

        case _:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Choose from: gemini | groq | ollama"
            )

    # -----------------------------------------------------------------------
    # Attach Langfuse callback for automatic token + latency tracking
    # -----------------------------------------------------------------------
    obs_cfg = settings.observability_config
    if with_langfuse and obs_cfg["provider"] == "langfuse" and obs_cfg["langfuse"]["enabled"]:
        try:
            langfuse_handler = LangfuseCallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            # LangChain models accept callbacks via .bind()
            llm = llm.bind(callbacks=[langfuse_handler])
            logger.info("Langfuse callback attached to LLM")
        except Exception as exc:
            logger.warning(f"Langfuse callback failed to attach: {exc}. Continuing without tracing.")

    return llm