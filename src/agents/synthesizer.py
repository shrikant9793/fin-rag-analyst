"""
src/agents/synthesizer.py
=========================
Report Synthesizer Agent
------------------------
Third node in the LangGraph graph (after Compliance). Responsible for:
  1. Building an LLM prompt from retrieved_docs + user_query
  2. Generating a structured markdown financial research report
  3. Parsing the report into sections for the HITL review panel
  4. Handling re-synthesis requests when the analyst rejects a draft (HITL loop)

Node signature (LangGraph):
    def synthesizer_node(state: AgentState) -> dict
"""

from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.llm_factory import get_llm
from src.observability import trace_node
from src.state import AgentState, WorkflowStatus

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYNTHESIZER_SYSTEM_PROMPT = """You are a Senior Financial Research Analyst at a top-tier investment bank.
Your task is to produce a professional, structured financial research report based ONLY on the
provided source documents. Do NOT invent figures or make forward-looking investment recommendations.

Report format (use exactly these markdown headers):

## Executive Summary
2-3 sentence overview of key findings.

## Financial Highlights
Bullet points covering: revenue, net income, EPS, gross margin, YoY growth.
Always cite the source document name in brackets e.g. [AAPL_10K_2024.pdf].

## Segment Analysis
Break down by business unit, geography, or product line as available in the sources.

## Risk Factors
Key risks mentioned in the source documents. Do not speculate.

## Analyst Notes
Your synthesis of the data — patterns, anomalies, or noteworthy trends observed across documents.

## Data Sources
List every source document referenced, with ticker and document type.

---
IMPORTANT COMPLIANCE RULES:
- Never use language like "buy", "sell", "invest", "recommend", "outperform", "target price".
- Every material fact must be traceable to a source document.
- If information is unavailable in the sources, state: "Data not available in provided documents."
"""


def _build_context(docs) -> str:
    """Format retrieved documents into a numbered context block for the LLM."""
    if not docs:
        return "No documents retrieved."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        ticker  = meta.get("ticker", "N/A")
        dtype   = meta.get("doc_type", "document")
        date    = meta.get("filing_date", "unknown date")
        source  = meta.get("source_file", f"doc_{i}")
        score   = meta.get("score", 0.0)

        parts.append(
            f"[SOURCE {i}] {source} | {ticker} | {dtype} | {date} | relevance={score:.3f}\n"
            f"{doc.page_content}\n"
            f"{'─' * 60}"
        )

    return "\n\n".join(parts)


def _parse_report_sections(report_text: str) -> dict[str, str]:
    """
    Extract named sections from the markdown report.
    Returns dict: {"executive_summary": "...", "financial_highlights": "...", ...}
    """
    section_map = {
        "executive_summary":    r"##\s+Executive Summary",
        "financial_highlights": r"##\s+Financial Highlights",
        "segment_analysis":     r"##\s+Segment Analysis",
        "risk_factors":         r"##\s+Risk Factors",
        "analyst_notes":        r"##\s+Analyst Notes",
        "data_sources":         r"##\s+Data Sources",
    }

    sections: dict[str, str] = {}
    for key, pattern in section_map.items():
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            start = match.end()
            # Find next section header or end of string
            next_header = re.search(r"\n##\s+", report_text[start:])
            end = start + next_header.start() if next_header else len(report_text)
            sections[key] = report_text[start:end].strip()
        else:
            sections[key] = ""

    return sections


# ---------------------------------------------------------------------------
# Synthesizer Node
# ---------------------------------------------------------------------------

@trace_node("synthesizer")
def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph node: generate structured draft financial report from RAG context.

    Reads from state:
        user_query, query_intent, retrieved_docs, override_notes, synthesis_retries

    Writes to state:
        draft_report, report_sections, synthesis_retries, status, node_timings
    """
    t0         = time.perf_counter()
    session_id = state.get("session_id", "unknown")
    retries    = state.get("synthesis_retries", 0)

    logger.info(f"[synthesizer] START | session={session_id} | retry={retries}")

    # Build context from retrieved documents
    docs    = state.get("retrieved_docs", [])
    context = _build_context(docs)

    # User prompt — include analyst override notes if this is a re-synthesis
    user_prompt_parts = [
        f"Research Question: {state['user_query']}",
        f"Query Intent: {state.get('query_intent', state['user_query'])}",
        f"Tickers: {', '.join(state.get('ticker', []) or ['Not specified'])}",
        "",
        "SOURCE DOCUMENTS:",
        context,
    ]

    if override_notes := state.get("override_notes", "").strip():
        user_prompt_parts.insert(0, f"ANALYST OVERRIDE NOTES (must incorporate):\n{override_notes}\n")
        logger.info(f"[synthesizer] Incorporating analyst override notes: '{override_notes[:80]}'")

    user_prompt = "\n".join(user_prompt_parts)

    llm = get_llm(with_langfuse=True)
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response     = llm.invoke(messages)
        draft_report = response.content.strip()
        sections     = _parse_report_sections(draft_report)

        logger.success(
            f"[synthesizer] DONE | report_len={len(draft_report)} chars | "
            f"sections={list(sections.keys())}"
        )

    except Exception as exc:
        logger.error(f"[synthesizer] LLM call failed: {exc}")
        draft_report = f"Report generation failed: {exc}"
        sections     = {}

    elapsed = time.perf_counter() - t0

    return {
        "draft_report":     draft_report,
        "report_sections":  sections,
        "synthesis_retries": retries + 1,
        "status":           WorkflowStatus.SYNTHESIZING,
        "node_timings":     {**state.get("node_timings", {}), "synthesizer": round(elapsed, 3)},
    }