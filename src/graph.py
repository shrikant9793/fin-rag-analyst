"""
src/graph.py
============
LangGraph Multi-Agent Graph — Financial Market Research Analyst
---------------------------------------------------------------

Graph topology:
                        ┌──────────────────────┐
                        │   fetcher_node        │  Parse query → ticker/intent
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   retriever_node      │  Hybrid RAG search (Qdrant)
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   compliance_node     │  NeMo Guardrails check
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
               PASSED/FLAGGED                  BLOCKED (retry)
                    │                              │
                    ▼                              ▼
          ┌─────────────────┐          ┌──────────────────┐
          │ synthesizer_node│◄─────────│ synthesizer_node │
          └────────┬────────┘          └──────────────────┘
                   │
          ┌────────▼────────┐
          │   hitl_node      │  ★ INTERRUPT — graph PAUSES here ★
          └────────┬────────┘    Analyst reviews draft in Streamlit UI
                   │
          ┌────────┴────────┐
       APPROVED           REJECTED
          │                  │
          ▼                  ▼
  ┌──────────────┐   ┌──────────────┐
  │ compiler_node│   │synthesizer_  │  (re-synthesise with override notes)
  │ (final rpt)  │   │node (retry)  │
  └──────────────┘   └──────────────┘

HITL Pause / Resume Mechanism:
    1. hitl_node calls LangGraph's interrupt() — graph execution halts and
       the current state is persisted to the checkpointer (SqliteSaver).
    2. The Streamlit UI reads the paused state, renders the draft_report,
       and lets the analyst approve or reject with override_notes.
    3. On approval:  graph.update_state(config, {"hitl_approved": True})
                     graph.invoke(None, config)   ← resumes from checkpoint
    4. On rejection: graph.update_state(config, {"hitl_approved": False,
                                                  "override_notes": "..."})
                     graph.invoke(None, config)   ← re-runs synthesizer_node

Public API:
    from src.graph import build_graph, run_graph, resume_graph
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from loguru import logger

from src.agents.compiler import compiler_node
from src.agents.fetcher import fetcher_node
from src.agents.retriever import retriever_node
from src.agents.synthesizer import synthesizer_node
from src.config import get_settings
from src.state import AgentState, ComplianceResult, WorkflowStatus, create_initial_state

# ---------------------------------------------------------------------------
# Lazy import compliance node (Day 3 — stub used until Day 3 is built)
# ---------------------------------------------------------------------------

def _get_compliance_node():
    """
    Import compliance_node lazily.
    Falls back to a passthrough stub if the Day-3 module isn't built yet.
    """
    try:
        from src.agents.compliance import compliance_node
        return compliance_node
    except ImportError:
        logger.warning(
            "[graph] compliance_node not found — using passthrough stub. "
            "This will be replaced in Day 3."
        )
        def _stub(state: AgentState) -> dict:
            return {
                "compliance_result": ComplianceResult.PASSED,
                "compliance_notes":  [],
                "status":            WorkflowStatus.COMPLIANCE_CHECK,
            }
        return _stub


# ---------------------------------------------------------------------------
# HITL Node
# ---------------------------------------------------------------------------

def hitl_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop node.

    Calls LangGraph's `interrupt()` to PAUSE graph execution and hand
    control back to the caller (Streamlit UI or CLI).

    The interrupt payload is the data the UI needs to render the approval
    panel — it is NOT a return value; it is surfaced via the graph's
    stream events and the thread's saved state.

    After the analyst acts:
      - Approved  → graph.update_state sets hitl_approved=True, resumes → compiler_node
      - Rejected  → graph.update_state sets hitl_approved=False + override_notes, resumes
                    → conditional edge routes back to synthesizer_node
    """
    logger.info(
        f"[hitl] PAUSING graph | session={state.get('session_id')} — "
        "awaiting analyst approval"
    )

    # interrupt() raises a special LangGraph exception that:
    #   1. Saves the entire current state to the checkpointer
    #   2. Surfaces the payload in the graph's streamed events
    #   3. Halts execution until graph.invoke(None, config) is called again
    interrupt({
        "message":      "Draft report ready for analyst review",
        "draft_report": state.get("draft_report", ""),
        "session_id":   state.get("session_id", ""),
        "tickers":      state.get("ticker", []),
    })

    # ── This line only executes AFTER the graph is resumed ──
    logger.info(
        f"[hitl] RESUMED | approved={state.get('hitl_approved')} | "
        f"override='{state.get('override_notes', '')[:60]}'"
    )

    approved = state.get("hitl_approved", False)
    return {
        "status": WorkflowStatus.HITL_APPROVED if approved else WorkflowStatus.HITL_REJECTED,
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_compliance(state: AgentState) -> str:
    """
    After compliance check:
      BLOCKED + retries remaining → back to synthesizer
      BLOCKED + retries exhausted → error
      PASSED / FLAGGED            → synthesizer (proceed)
    """
    result  = state.get("compliance_result", ComplianceResult.PASSED)
    retries = state.get("compliance_retries", 0)
    max_r   = get_settings().guardrails_config.get("max_retries", 2)

    if result == ComplianceResult.BLOCKED:
        if retries < max_r:
            logger.warning(f"[graph] Compliance BLOCKED — re-routing to synthesizer (retry {retries})")
            return "synthesizer"
        else:
            logger.error("[graph] Compliance BLOCKED — max retries exhausted → error")
            return "error_handler"

    return "synthesizer"


def route_after_hitl(state: AgentState) -> str:
    """
    After HITL node resumes:
      Approved  → compiler
      Rejected  → synthesizer (re-draft with override notes)
    """
    if state.get("hitl_approved", False):
        logger.info("[graph] HITL APPROVED → compiler")
        return "compiler"
    else:
        logger.info("[graph] HITL REJECTED → synthesizer (re-draft)")
        return "synthesizer"


def route_after_synthesizer(state: AgentState) -> str:
    """
    After synthesizer:
      Always go to hitl (analyst must review every draft).
    """
    return "hitl"


# ---------------------------------------------------------------------------
# Error handler node
# ---------------------------------------------------------------------------

def error_handler_node(state: AgentState) -> dict:
    """Catch-all error node — logs and marks workflow as errored."""
    msg = state.get("error_message", "Unknown error in agent graph")
    logger.error(f"[graph] ERROR node reached: {msg}")
    return {
        "status":        WorkflowStatus.ERROR,
        "final_report":  f"**Report generation failed.**\n\nError: {msg}",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> tuple[Any, SqliteSaver]:
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        (compiled_graph, checkpointer)

    The checkpointer is returned so the caller can use it for
    `graph.update_state()` and `graph.get_state()` during HITL.
    """
    settings = get_settings()
    graph_cfg = settings.graph_config

    # --- Checkpointer (state persistence for HITL pause/resume) ---
    db_path     = Path(graph_cfg.get("sqlite_db_path", "./data/checkpoints.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(str(db_path))
    logger.info(f"[graph] Checkpointer: SqliteSaver @ {db_path}")

    # --- Build graph ---
    graph = StateGraph(AgentState)

    compliance_fn = _get_compliance_node()

    # Add nodes
    graph.add_node("fetcher",       fetcher_node)
    graph.add_node("retriever",     retriever_node)
    graph.add_node("compliance",    compliance_fn)
    graph.add_node("synthesizer",   synthesizer_node)
    graph.add_node("hitl",          hitl_node)
    graph.add_node("compiler",      compiler_node)
    graph.add_node("error_handler", error_handler_node)

    # Add edges
    graph.add_edge(START,        "fetcher")
    graph.add_edge("fetcher",    "retriever")
    graph.add_edge("retriever",  "compliance")

    # Compliance → synthesizer (conditional on compliance result)
    graph.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "synthesizer":  "synthesizer",
            "error_handler": "error_handler",
        },
    )

    # Synthesizer → hitl (always — every draft needs analyst sign-off)
    graph.add_conditional_edges(
        "synthesizer",
        route_after_synthesizer,
        {"hitl": "hitl"},
    )

    # HITL → compiler (approved) or synthesizer (rejected)
    graph.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "compiler":    "compiler",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("compiler",      END)
    graph.add_edge("error_handler", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],   # pause BEFORE entering hitl_node
    )

    logger.success("[graph] Graph compiled successfully")
    return compiled, checkpointer


# ---------------------------------------------------------------------------
# Public run helpers
# ---------------------------------------------------------------------------

def run_graph(
    user_query: str,
    session_id: str,
) -> tuple[AgentState, RunnableConfig]:
    """
    Start a new graph run.

    Runs until the graph PAUSES at the HITL interrupt_before node.
    Returns the paused state and the config needed to resume.

    Args:
        user_query:  Analyst's research question.
        session_id:  Unique run identifier (used as LangGraph thread_id).

    Returns:
        (paused_state, config)
    """
    graph, _ = build_graph()

    initial_state = create_initial_state(user_query=user_query, session_id=session_id)
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    logger.info(f"[graph] Starting run | session={session_id}")

    # Stream events until interrupt
    paused_state: AgentState = {}
    for event in graph.stream(initial_state, config, stream_mode="values"):
        paused_state = event
        status = paused_state.get("status", "")
        logger.debug(f"[graph] Event | status={status}")

    logger.info(
        f"[graph] Graph paused at HITL | session={session_id} | "
        f"draft_len={len(paused_state.get('draft_report', ''))}"
    )
    return paused_state, config


def resume_graph(
    session_id: str,
    hitl_approved: bool,
    override_notes: str = "",
) -> AgentState:
    """
    Resume a paused graph run after analyst HITL decision.

    Args:
        session_id:     The session_id used in run_graph().
        hitl_approved:  True = analyst approved draft, False = rejected.
        override_notes: Analyst's correction text (used on rejection).

    Returns:
        Final AgentState after graph completes.
    """
    graph, checkpointer = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    # Inject analyst decision into the saved state
    graph.update_state(
        config,
        {
            "hitl_approved":  hitl_approved,
            "override_notes": override_notes,
            "hitl_timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        },
    )

    logger.info(
        f"[graph] Resuming | session={session_id} | "
        f"approved={hitl_approved} | notes='{override_notes[:60]}'"
    )

    # Resume — passes None as input (state is read from checkpointer)
    final_state: AgentState = {}
    for event in graph.stream(None, config, stream_mode="values"):
        final_state = event
        status = final_state.get("status", "")
        logger.debug(f"[graph] Resume event | status={status}")

    logger.success(
        f"[graph] Run complete | session={session_id} | "
        f"status={final_state.get('status')}"
    )
    return final_state


def get_graph_state(session_id: str) -> AgentState:
    """
    Retrieve the current persisted state for a given session.
    Useful for the Streamlit UI to poll the paused draft_report.
    """
    graph, _ = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)
    return snapshot.values if snapshot else {}