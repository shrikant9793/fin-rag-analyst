"""
src/observability.py
====================
Langfuse Observability — tracing helpers for the multi-agent graph.

Provides:
  - LangfuseTracer: context manager that wraps any agent step in a Langfuse span
  - get_langfuse_client(): lazy singleton Langfuse client
  - trace_node(): decorator for LangGraph node functions
  - flush(): flush all pending traces (call at end of a run)

Usage in a node:
    from src.observability import trace_node

    @trace_node("synthesizer")
    def synthesizer_node(state: AgentState) -> dict:
        ...

Or manually:
    from src.observability import LangfuseTracer

    with LangfuseTracer("retriever", session_id=state["session_id"]) as span:
        results = pipeline.retrieve(query)
        span.update(output={"chunks": len(results)})
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from loguru import logger

from src.config import get_settings

# ---------------------------------------------------------------------------
# Lazy Langfuse client singleton
# ---------------------------------------------------------------------------

_langfuse_client = None


def get_langfuse_client():
    """
    Return a cached Langfuse client.
    Returns None if Langfuse is disabled or keys are missing.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    settings = get_settings()
    obs_cfg  = settings.observability_config

    if obs_cfg["provider"] != "langfuse" or not obs_cfg["langfuse"]["enabled"]:
        logger.debug("[observability] Langfuse disabled — no-op tracing")
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "[observability] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
            "tracing disabled. Add keys to .env to enable."
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            flush_at=obs_cfg["langfuse"]["flush_at"],
            flush_interval=obs_cfg["langfuse"]["flush_interval"],
        )
        logger.info(f"[observability] Langfuse client initialised | host={settings.langfuse_host}")
        return _langfuse_client

    except ImportError:
        logger.warning("[observability] langfuse package not installed — tracing disabled")
        return None
    except Exception as exc:
        logger.warning(f"[observability] Langfuse init failed ({exc}) — tracing disabled")
        return None


# ---------------------------------------------------------------------------
# LangfuseTracer context manager
# ---------------------------------------------------------------------------

class LangfuseTracer:
    """
    Context manager that wraps a block of code in a Langfuse trace + span.

    Creates a top-level trace on first use for a session, then adds a span
    for each agent node.

    Example:
        with LangfuseTracer("retriever", session_id="abc123", input={"query": q}) as span:
            results = pipeline.retrieve(q)
            span.update(output={"chunks_retrieved": len(results)})
    """

    def __init__(
        self,
        node_name: str,
        session_id: str = "",
        input_data: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.node_name  = node_name
        self.session_id = session_id
        self.input_data = input_data or {}
        self.metadata   = metadata or {}
        self._trace     = None
        self._span      = None
        self._t0        = 0.0
        self._client    = get_langfuse_client()

    def __enter__(self) -> "LangfuseTracer":
        self._t0 = time.perf_counter()

        if self._client is None:
            return self

        try:
            # One trace per session — Langfuse deduplicates by trace_id
            self._trace = self._client.trace(
                id=self.session_id or None,
                name=f"fin-rag-{self.session_id}",
                session_id=self.session_id,
                metadata=self.metadata,
            )

            self._span = self._trace.span(
                name=self.node_name,
                input=self.input_data,
                metadata={**self.metadata, "node": self.node_name},
            )
            logger.debug(f"[observability] Span opened: {self.node_name} | session={self.session_id}")

        except Exception as exc:
            logger.warning(f"[observability] Failed to open span '{self.node_name}': {exc}")

        return self

    def update(self, output: dict | None = None, metadata: dict | None = None) -> None:
        """Update the active span with output data and/or extra metadata."""
        if self._span is None:
            return
        try:
            kwargs: dict[str, Any] = {}
            if output:
                kwargs["output"] = output
            if metadata:
                kwargs["metadata"] = {**(self.metadata or {}), **metadata}
            self._span.update(**kwargs)
        except Exception as exc:
            logger.warning(f"[observability] span.update failed: {exc}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.perf_counter() - self._t0

        if self._span is None:
            return

        try:
            end_kwargs: dict[str, Any] = {}
            if exc_type is not None:
                end_kwargs["level"]       = "ERROR"
                end_kwargs["status_message"] = str(exc_val)
            self._span.end(**end_kwargs)
            logger.debug(
                f"[observability] Span closed: {self.node_name} | {elapsed:.2f}s"
                + (f" | ERROR: {exc_val}" if exc_type else "")
            )
        except Exception as exc:
            logger.warning(f"[observability] span.end failed: {exc}")

        return False  # don't suppress exceptions


# ---------------------------------------------------------------------------
# trace_node decorator — wrap a LangGraph node function automatically
# ---------------------------------------------------------------------------

def trace_node(node_name: str):
    """
    Decorator that wraps a LangGraph node function with Langfuse tracing.

    The decorated function must accept AgentState as its first argument
    and return a dict (partial state update).

    Example:
        @trace_node("synthesizer")
        def synthesizer_node(state: AgentState) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            session_id = state.get("session_id", "unknown")
            user_query = state.get("user_query", "")

            input_data = {
                "session_id": session_id,
                "user_query": user_query[:200],
                "status":     str(state.get("status", "")),
            }
            metadata = {
                "ticker":   state.get("ticker", []),
                "env":      get_settings().environment,
                "provider": get_settings().llm_provider,
            }

            with LangfuseTracer(
                node_name=node_name,
                session_id=session_id,
                input_data=input_data,
                metadata=metadata,
            ) as tracer:
                result = fn(state, *args, **kwargs)

                # Auto-update span with key output fields
                output_summary = {
                    "status":    str(result.get("status", "")),
                    "timing_s":  result.get("node_timings", {}).get(node_name, 0),
                }
                # Add node-specific output fields
                if "chunks_retrieved" in result.get("retrieval_metadata", {}):
                    output_summary["chunks_retrieved"] = result["retrieval_metadata"]["chunks_retrieved"]
                if "compliance_result" in result:
                    output_summary["compliance_result"] = str(result["compliance_result"])
                if "draft_report" in result:
                    output_summary["draft_len"] = len(result["draft_report"])

                tracer.update(output=output_summary)

            return result

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Score logging — for eval metrics (Day 5)
# ---------------------------------------------------------------------------

def log_eval_scores(
    session_id: str,
    scores: dict[str, float],
) -> None:
    """
    Log Ragas / DeepEval metric scores as Langfuse scores on a trace.

    Args:
        session_id: The session this evaluation belongs to.
        scores:     Dict of metric_name → float e.g. {"faithfulness": 0.87}
    """
    client = get_langfuse_client()
    if client is None:
        return

    try:
        for metric_name, value in scores.items():
            client.score(
                trace_id=session_id,
                name=metric_name,
                value=value,
                comment=f"Automated eval | {metric_name}={value:.3f}",
            )
        logger.info(f"[observability] Eval scores logged | session={session_id} | {scores}")
    except Exception as exc:
        logger.warning(f"[observability] Failed to log eval scores: {exc}")


# ---------------------------------------------------------------------------
# Flush helper — call at end of a request / run
# ---------------------------------------------------------------------------

def flush() -> None:
    """Flush all pending Langfuse events. Call at end of each graph run."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
            logger.debug("[observability] Langfuse flushed")
        except Exception as exc:
            logger.warning(f"[observability] Langfuse flush failed: {exc}")