"""
scripts/smoke_test.py
=====================
End-to-End Smoke Test — validates the full project is wired correctly
WITHOUT requiring any external API key, running Qdrant, or LLM calls.

Tests:
  1. Config loads cleanly and all providers are switchable
  2. AgentState schema validates with all required fields
  3. All graph routing functions return correct next-node strings
  4. Compliance regex rules fire on known bad text
  5. RAGPipeline helper functions (RRF, chunking) work correctly
  6. LangfuseTracer is a no-op when client is None
  7. All UI component constants are structurally valid
  8. Golden dataset loads and is well-formed
  9. All Python modules import without error

Run:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()
RESULTS: list[tuple[str, bool, str]] = []


# ---------------------------------------------------------------------------
# Test runner harness
# ---------------------------------------------------------------------------

def test(name: str):
    """Decorator that registers a smoke test function."""
    def decorator(fn: Callable) -> Callable:
        RESULTS.append((name, False, "not run"))

        def wrapper(*args, **kwargs):
            idx = next(i for i, (n, _, _) in enumerate(RESULTS) if n == name)
            try:
                fn(*args, **kwargs)
                RESULTS[idx] = (name, True, "")
            except Exception as exc:
                RESULTS[idx] = (name, False, str(exc))
        wrapper._name = name
        TESTS.append(wrapper)
        return wrapper
    return decorator


TESTS: list[Callable] = []


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@test("Config — loads settings singleton")
def _():
    from src.config import get_settings
    # Clear lru_cache to ensure fresh load
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider in ("gemini", "groq", "ollama")
    assert settings.qdrant_config["collection_name"]
    assert settings.ingestion_config["chunk_size"] > 0


@test("Config — provider switching in-memory")
def _():
    from src.config import get_settings
    settings = get_settings()
    original = settings.llm_provider
    for provider in ("gemini", "groq", "ollama"):
        settings._yaml["llm"]["active"] = provider
        assert settings.llm_provider == provider
    settings._yaml["llm"]["active"] = original


@test("State — create_initial_state returns clean AgentState")
def _():
    from src.state import (
        AgentState, ComplianceResult, WorkflowStatus, create_initial_state,
    )
    state = create_initial_state("Apple gross margin Q4?", "smoke-001")
    assert state["user_query"]        == "Apple gross margin Q4?"
    assert state["session_id"]        == "smoke-001"
    assert state["thread_id"]         == "smoke-001"
    assert state["ticker"]            == []
    assert state["retrieved_docs"]    == []
    assert state["draft_report"]      == ""
    assert state["hitl_approved"]     is False
    assert state["status"]            == WorkflowStatus.INITIALIZED
    assert state["compliance_result"] == ComplianceResult.PASSED


@test("State — WorkflowStatus enum completeness")
def _():
    from src.state import WorkflowStatus
    required = {
        "INITIALIZED", "FETCHING", "RETRIEVING", "COMPLIANCE_CHECK",
        "SYNTHESIZING", "AWAITING_HITL", "HITL_APPROVED", "HITL_REJECTED",
        "COMPILING", "COMPLETE", "ERROR",
    }
    actual = {s.name for s in WorkflowStatus}
    missing = required - actual
    assert not missing, f"WorkflowStatus missing: {missing}"


@test("Graph — route_after_compliance routing logic")
def _():
    from unittest.mock import patch
    from src.graph import route_after_compliance
    from src.state import ComplianceResult, create_initial_state

    state_passed  = {**create_initial_state("q", "s"), "compliance_result": ComplianceResult.PASSED,  "compliance_retries": 0}
    state_flagged = {**create_initial_state("q", "s"), "compliance_result": ComplianceResult.FLAGGED, "compliance_retries": 0}
    state_blocked = {**create_initial_state("q", "s"), "compliance_result": ComplianceResult.BLOCKED, "compliance_retries": 1}
    state_exhaust = {**create_initial_state("q", "s"), "compliance_result": ComplianceResult.BLOCKED, "compliance_retries": 5}

    mock_cfg = type("S", (), {"guardrails_config": {"max_retries": 2}})()
    with patch("src.graph.get_settings", return_value=mock_cfg):
        assert route_after_compliance(state_passed)  == "synthesizer"
        assert route_after_compliance(state_flagged) == "synthesizer"
        assert route_after_compliance(state_blocked) == "synthesizer"
        assert route_after_compliance(state_exhaust) == "error_handler"


@test("Graph — route_after_hitl routing logic")
def _():
    from src.graph import route_after_hitl
    from src.state import create_initial_state

    approved = {**create_initial_state("q", "s"), "hitl_approved": True}
    rejected = {**create_initial_state("q", "s"), "hitl_approved": False}
    assert route_after_hitl(approved) == "compiler"
    assert route_after_hitl(rejected) == "synthesizer"


@test("Graph — route_after_synthesizer always returns hitl")
def _():
    from src.graph import route_after_synthesizer
    from src.state import create_initial_state
    assert route_after_synthesizer(create_initial_state("q", "s")) == "hitl"


@test("Compliance — regex BLOCKED patterns fire correctly")
def _():
    from src.agents.compliance import _run_regex_check
    from src.state import ComplianceResult

    blocked_texts = [
        "This is a strong buy opportunity.",
        "I recommend buying AAPL at current levels.",
        "You should sell your position now.",
        "Price target of $250 within 12 months.",
    ]
    for text in blocked_texts:
        result, violations, _ = _run_regex_check(text)
        assert result == ComplianceResult.BLOCKED, \
            f"Expected BLOCKED for: '{text[:50]}' but got {result}"


@test("Compliance — regex FLAGGED patterns fire correctly")
def _():
    from src.agents.compliance import _run_regex_check
    from src.state import ComplianceResult

    flagged_texts = [
        "Apple is expected to outperform the market.",
        "This presents a buy opportunity for long-term investors.",
    ]
    for text in flagged_texts:
        result, violations, modified = _run_regex_check(text)
        assert result == ComplianceResult.FLAGGED, \
            f"Expected FLAGGED for: '{text[:50]}' but got {result}"
        assert "Compliance Notice" in modified


@test("Compliance — clean financial text PASSES")
def _():
    from src.agents.compliance import _run_regex_check
    from src.state import ComplianceResult

    clean_texts = [
        "Apple Q4 FY2024 revenue was $94.9 billion, up 6% YoY.",
        "Gross margin expanded to 46.2% from 45.2% in the prior year.",
        "Services segment grew 12% year-over-year to $25.3 billion.",
    ]
    for text in clean_texts:
        result, violations, _ = _run_regex_check(text)
        assert result == ComplianceResult.PASSED, \
            f"Expected PASSED for: '{text[:50]}' but got {result} | {violations}"


@test("RAG — RRF fusion ranking is correct")
def _():
    from unittest.mock import MagicMock
    from src.rag_pipeline import RAGPipeline

    # doc_a appears in both lists → should rank first
    dense  = [MagicMock(id="doc_a"), MagicMock(id="doc_b"), MagicMock(id="doc_c")]
    sparse = [MagicMock(id="doc_a"), MagicMock(id="doc_d")]
    fused  = RAGPipeline._rrf_fuse(dense, sparse, k=60)

    assert fused[0][0] == "doc_a", "doc_a in both lists must rank first"
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True), "Scores must be descending"


@test("RAG — RRF handles empty inputs")
def _():
    from src.rag_pipeline import RAGPipeline
    assert RAGPipeline._rrf_fuse([], [])        == []
    assert RAGPipeline._rrf_fuse([], [], k=60)  == []


@test("RAG — RetrievedChunk.to_langchain_doc")
def _():
    from langchain_core.documents import Document
    from src.rag_pipeline import RetrievedChunk

    chunk = RetrievedChunk(
        content  = "Apple gross margin 46.2%",
        score    = 0.87,
        metadata = {"ticker": "AAPL", "doc_type": "earnings_transcript"},
        chunk_id = "abc123",
    )
    doc = chunk.to_langchain_doc()
    assert isinstance(doc, Document)
    assert doc.page_content         == "Apple gross margin 46.2%"
    assert doc.metadata["ticker"]   == "AAPL"
    assert doc.metadata["score"]    == 0.87


@test("Observability — LangfuseTracer no-op when client is None")
def _():
    from unittest.mock import patch
    from src.observability import LangfuseTracer

    with patch("src.observability.get_langfuse_client", return_value=None):
        tracer = LangfuseTracer("test_node", session_id="smoke-001")
        with tracer as t:
            t.update(output={"key": "value"})
        # No exception = pass


@test("Observability — flush() no-op when client is None")
def _():
    from unittest.mock import patch
    from src.observability import flush
    with patch("src.observability.get_langfuse_client", return_value=None):
        flush()   # must not raise


@test("Observability — trace_node decorator preserves return value")
def _():
    from unittest.mock import patch
    from src.observability import trace_node

    @trace_node("smoke_node")
    def dummy(state: dict) -> dict:
        return {"result": 42, "status": "ok"}

    with patch("src.observability.get_langfuse_client", return_value=None):
        result = dummy({"session_id": "s1", "user_query": "q", "node_timings": {}})

    assert result["result"] == 42
    assert result["status"] == "ok"


@test("UI — chat panel NODE_PROGRESS covers all statuses")
def _():
    from ui.components.chat_panel import NODE_PROGRESS
    required = {"fetching", "retrieving", "compliance_check", "synthesizing", "awaiting_hitl", "complete", "error"}
    missing  = required - set(NODE_PROGRESS.keys())
    assert not missing, f"NODE_PROGRESS missing keys: {missing}"
    for k, v in NODE_PROGRESS.items():
        assert isinstance(v, tuple) and len(v) == 2, f"Entry for {k} must be (icon, label) tuple"


@test("UI — eval dashboard thresholds match metric labels")
def _():
    from ui.components.eval_dashboard import THRESHOLDS, METRIC_LABELS, _demo_scores
    assert set(THRESHOLDS.keys()) == set(METRIC_LABELS.keys()), \
        "THRESHOLDS and METRIC_LABELS must have identical keys"

    demo = _demo_scores()
    for metric in THRESHOLDS:
        assert metric in demo, f"Demo scores missing metric: {metric}"
    assert demo["hallucination"] <= THRESHOLDS["hallucination"], \
        "Demo hallucination score must be below threshold"


@test("UI — sidebar PROVIDER_LABELS covers all providers")
def _():
    from ui.components.sidebar import PROVIDER_LABELS, PROVIDER_COLOURS
    for provider in ("gemini", "groq", "ollama"):
        assert provider in PROVIDER_LABELS,  f"Missing label for {provider}"
        assert provider in PROVIDER_COLOURS, f"Missing colour for {provider}"


@test("Eval — golden dataset loads with correct structure")
def _():
    from tests.eval_suite import load_golden_dataset, THRESHOLDS
    dataset = load_golden_dataset()
    assert len(dataset) == 10, f"Expected 10 QA pairs, got {len(dataset)}"

    required_fields = {"question", "ground_truth", "contexts", "ticker", "doc_type"}
    for item in dataset:
        missing = required_fields - set(item.keys())
        assert not missing, f"Item {item.get('id')} missing: {missing}"
        assert isinstance(item["contexts"], list) and len(item["contexts"]) >= 1

    for metric, threshold in THRESHOLDS.items():
        assert 0.0 <= threshold <= 1.0, f"Threshold for {metric} out of range"


@test("Imports — all src modules import without error")
def _():
    import importlib
    modules = [
        "src.config",
        "src.state",
        "src.llm_factory",
        "src.rag_pipeline",
        "src.graph",
        "src.observability",
        "src.agents.fetcher",
        "src.agents.retriever",
        "src.agents.synthesizer",
        "src.agents.compliance",
        "src.agents.compiler",
    ]
    for mod in modules:
        importlib.import_module(mod)


@test("Imports — all ui modules import without error")
def _():
    import importlib
    modules = [
        "ui.components.sidebar",
        "ui.components.chat_panel",
        "ui.components.hitl_panel",
        "ui.components.report_panel",
        "ui.components.eval_dashboard",
    ]
    for mod in modules:
        importlib.import_module(mod)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(verbose: bool = False) -> bool:
    """Run all registered smoke tests and print results table."""
    console.print()
    console.rule("[bold cyan]End-to-End Smoke Test[/bold cyan]")
    console.print(f"  Running [bold]{len(TESTS)}[/bold] smoke tests…\n")

    for test_fn in TESTS:
        test_fn()

    # Results table
    table = Table(show_lines=True, title=f"Smoke Test Results ({len(TESTS)} tests)")
    table.add_column("#",       width=4,  justify="right")
    table.add_column("Test",    style="cyan")
    table.add_column("Result",  width=10, justify="center")
    table.add_column("Detail",  style="dim")

    passed = 0
    failed = 0
    for i, (name, ok, detail) in enumerate(RESULTS, 1):
        if ok:
            passed += 1
            table.add_row(str(i), name, "[green]✅ PASS[/green]", "")
        else:
            failed += 1
            table.add_row(str(i), name, "[red]❌ FAIL[/red]",
                          detail[:120] if not verbose else detail)

    console.print(table)

    colour = "green" if failed == 0 else "red"
    label  = "ALL TESTS PASSED ✅" if failed == 0 else f"{failed} TEST(S) FAILED ❌"
    console.print(
        f"\n[bold {colour}]{label}[/bold {colour}]  "
        f"[dim]({passed} passed, {failed} failed)[/dim]\n"
    )

    if verbose and failed > 0:
        console.print("[bold red]Failures:[/bold red]")
        for name, ok, detail in RESULTS:
            if not ok:
                console.print(f"\n  [red]• {name}[/red]")
                console.print(f"    {detail}")

    return failed == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end smoke test")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full error tracebacks")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    success = run_all(verbose=args.verbose)
    sys.exit(0 if success else 1)