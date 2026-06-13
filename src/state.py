"""
src/state.py
============
LangGraph AgentState — the single shared data structure that flows through
every node in the multi-agent graph.

Design principles:
  - All fields are Optional with defaults so any node can be the entry point.
  - Immutable updates: LangGraph merges returned dicts into the state;
    nodes never mutate the state object directly.
  - HITL fields (`hitl_approved`, `override_notes`) are set exclusively by
    the Human-in-the-Loop node and the Streamlit UI resume handler.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WorkflowStatus(str, Enum):
    """Tracks where the graph run currently sits."""
    INITIALIZED          = "initialized"
    FETCHING             = "fetching"
    RETRIEVING           = "retrieving"
    COMPLIANCE_CHECK     = "compliance_check"
    SYNTHESIZING         = "synthesizing"
    AWAITING_HITL        = "awaiting_hitl"       # graph is PAUSED here
    HITL_APPROVED        = "hitl_approved"
    HITL_REJECTED        = "hitl_rejected"
    COMPILING            = "compiling"
    COMPLETE             = "complete"
    ERROR                = "error"


class ComplianceResult(str, Enum):
    PASSED   = "passed"
    FLAGGED  = "flagged"      # output modified with disclaimer
    BLOCKED  = "blocked"      # cannot proceed, re-route to synthesizer


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    Shared state object passed between every LangGraph node.

    Fields are grouped by the agent that primarily writes them:

    ── Input ──────────────────────────────────────────────────────────────
    user_query          Raw question from the analyst / UI
    session_id          Unique run identifier (used for Langfuse tracing)
    thread_id           LangGraph checkpointer thread key (= session_id)

    ── Financial Data Fetcher ─────────────────────────────────────────────
    ticker              Extracted stock ticker(s) e.g. ["AAPL", "MSFT"]
    date_range          {"start": "2024-01-01", "end": "2024-10-31"}
    doc_type_filter     e.g. "sec_filing" | "earnings_transcript" | None
    query_intent        Short description of what the analyst is asking

    ── RAG Retriever ──────────────────────────────────────────────────────
    retrieved_docs      List of LangChain Documents with scores in metadata
    retrieval_metadata  {"chunks_retrieved": 5, "top_score": 0.87, ...}

    ── Compliance Agent ───────────────────────────────────────────────────
    compliance_result   ComplianceResult enum value
    compliance_notes    List of rail violations or warnings
    compliance_retries  Counter — blocks infinite retry loops

    ── Report Synthesizer ─────────────────────────────────────────────────
    draft_report        Structured markdown report produced by LLM
    report_sections     Parsed sections {"summary": ..., "financials": ...}
    synthesis_retries   Counter for guardrail-triggered re-synthesis

    ── HITL Node (Human-in-the-Loop) ──────────────────────────────────────
    hitl_approved       True = analyst approved, False = rejected/override
    override_notes      Analyst's free-text corrections / override reason
    hitl_timestamp      ISO timestamp when analyst acted

    ── Final Compiler ─────────────────────────────────────────────────────
    final_report        Production-ready report after HITL sign-off
    report_version      Incremented on each HITL rejection + re-synthesis

    ── Observability ──────────────────────────────────────────────────────
    trace_id            Langfuse trace ID for the current run
    node_timings        {"node_name": elapsed_seconds, ...}
    status              WorkflowStatus enum value

    ── Error handling ─────────────────────────────────────────────────────
    error_message       Set if any node raises; triggers error edge
    messages            LangChain message history (add_messages reducer)
    """

    # Input
    user_query:         str
    session_id:         str
    thread_id:          str

    # Financial Data Fetcher outputs
    ticker:             list[str]
    date_range:         dict[str, str]
    doc_type_filter:    str | None
    query_intent:       str

    # RAG Retriever outputs
    retrieved_docs:     list[Document]
    retrieval_metadata: dict[str, Any]

    # Compliance Agent outputs
    compliance_result:  ComplianceResult
    compliance_notes:   list[str]
    compliance_retries: int

    # Report Synthesizer outputs
    draft_report:       str
    report_sections:    dict[str, str]
    synthesis_retries:  int

    # HITL outputs (set by UI / resume handler)
    hitl_approved:      bool
    override_notes:     str
    hitl_timestamp:     str

    # Final Compiler outputs
    final_report:       str
    report_version:     int

    # Observability
    trace_id:           str
    node_timings:       dict[str, float]
    status:             WorkflowStatus

    # Error
    error_message:      str

    # LangChain message history (uses add_messages reducer for append semantics)
    messages:           Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# State factory — create a clean initial state for a new run
# ---------------------------------------------------------------------------

def create_initial_state(
    user_query: str,
    session_id: str,
) -> AgentState:
    """
    Build a fresh AgentState for a new graph invocation.

    Args:
        user_query:  The analyst's financial research question.
        session_id:  Unique run ID (also used as LangGraph thread_id).

    Returns:
        Populated AgentState dict ready to pass to graph.invoke().
    """
    return AgentState(
        user_query         = user_query,
        session_id         = session_id,
        thread_id          = session_id,

        # Fetcher defaults
        ticker             = [],
        date_range         = {},
        doc_type_filter    = None,
        query_intent       = "",

        # Retriever defaults
        retrieved_docs     = [],
        retrieval_metadata = {},

        # Compliance defaults
        compliance_result  = ComplianceResult.PASSED,
        compliance_notes   = [],
        compliance_retries = 0,

        # Synthesizer defaults
        draft_report       = "",
        report_sections    = {},
        synthesis_retries  = 0,

        # HITL defaults (paused, awaiting analyst)
        hitl_approved      = False,
        override_notes     = "",
        hitl_timestamp     = "",

        # Final report defaults
        final_report       = "",
        report_version     = 1,

        # Observability
        trace_id           = "",
        node_timings       = {},
        status             = WorkflowStatus.INITIALIZED,

        # Error
        error_message      = "",

        # Messages
        messages           = [],
    )