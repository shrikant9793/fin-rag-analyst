"""
src/agents/fetcher.py
=====================
Financial Data Fetcher Agent
-----------------------------
First node in the LangGraph graph. Responsible for:
  1. Parsing the analyst's natural-language query
  2. Extracting structured metadata: ticker(s), date range, doc_type, intent
  3. Populating the relevant AgentState fields for downstream nodes

The agent uses the LLM to do structured extraction via a JSON-mode prompt.
No external API calls to financial data providers at this stage — that is
an optional extension (e.g. yfinance) added in later sprints.

Node signature (LangGraph):
    def fetcher_node(state: AgentState) -> dict   ← partial state update
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.llm_factory import get_llm
from src.observability import trace_node
from src.state import AgentState, WorkflowStatus

# ---------------------------------------------------------------------------
# System prompt for structured extraction
# ---------------------------------------------------------------------------

FETCHER_SYSTEM_PROMPT = """You are a financial data extraction assistant.
Your job is to parse a financial analyst's research query and extract structured metadata.

Respond ONLY with a valid JSON object — no markdown, no explanation, no code fences.

JSON schema:
{
  "tickers":        ["AAPL"],          // list of stock ticker symbols (uppercase). Empty list if none.
  "date_range": {
    "start": "YYYY-MM-DD",             // earliest relevant date. Use "2020-01-01" if unspecified.
    "end":   "YYYY-MM-DD"              // latest relevant date. Use today's date if unspecified.
  },
  "doc_type_filter": "sec_filing",     // one of: sec_filing | earnings_transcript | research_note | null
  "query_intent": "..."                // one sentence: what the analyst wants to know
}

Rules:
- Tickers must be standard exchange symbols (e.g. AAPL, MSFT, GOOGL, NVDA, TSLA).
- If multiple companies are mentioned, include all their tickers.
- doc_type_filter: use null if the query doesn't specify a document type.
- query_intent: be specific and financial — e.g. "Analyse Apple gross margin trend Q3-Q4 FY2024"
"""

# ---------------------------------------------------------------------------
# Fetcher Node
# ---------------------------------------------------------------------------

@trace_node("fetcher")
def fetcher_node(state: AgentState) -> dict:
    """
    LangGraph node: parse user_query → extract ticker, date_range, intent.

    Args:
        state: Current AgentState (reads: user_query, session_id)

    Returns:
        Partial state update dict with fetcher outputs.
    """
    t0          = time.perf_counter()
    user_query  = state["user_query"]
    session_id  = state.get("session_id", "unknown")

    logger.info(f"[fetcher] START | session={session_id} | query='{user_query[:80]}'")

    llm = get_llm(with_langfuse=True)

    messages = [
        SystemMessage(content=FETCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"Parse this financial research query:\n\n{user_query}"),
    ]

    try:
        response    = llm.invoke(messages)
        raw_content = response.content.strip()

        # Strip accidental markdown fences
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
            raw_content = raw_content.strip()

        extracted: dict = json.loads(raw_content)

        tickers         = [t.upper() for t in extracted.get("tickers", [])]
        date_range      = extracted.get("date_range", {
            "start": "2020-01-01",
            "end":   datetime.now().strftime("%Y-%m-%d"),
        })
        doc_type_filter = extracted.get("doc_type_filter")
        query_intent    = extracted.get("query_intent", user_query)

        logger.success(
            f"[fetcher] DONE | tickers={tickers} | intent='{query_intent[:60]}'"
        )

    except (json.JSONDecodeError, KeyError, Exception) as exc:
        # Graceful fallback — don't crash the graph
        logger.warning(f"[fetcher] LLM extraction failed ({exc}), using fallback defaults")
        tickers         = []
        date_range      = {"start": "2020-01-01", "end": datetime.now().strftime("%Y-%m-%d")}
        doc_type_filter = None
        query_intent    = user_query

    elapsed = time.perf_counter() - t0

    return {
        "ticker":          tickers,
        "date_range":      date_range,
        "doc_type_filter": doc_type_filter,
        "query_intent":    query_intent,
        "status":          WorkflowStatus.FETCHING,
        "node_timings":    {**state.get("node_timings", {}), "fetcher": round(elapsed, 3)},
    }