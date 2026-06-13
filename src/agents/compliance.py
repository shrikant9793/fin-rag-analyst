"""
src/agents/compliance.py
========================
Compliance / Guardrail Agent
-----------------------------
Fourth node in the LangGraph graph (runs after retriever, before synthesizer).

Two-layer validation strategy:
  Layer 1 — NeMo Guardrails (LLM-backed rail evaluation via colang rules)
  Layer 2 — Deterministic regex fallback (runs even if NeMo unavailable)

Responsibilities:
  1. Validate that the incoming query is appropriate
  2. Post-validate the synthesizer draft_report before HITL
  3. Set compliance_result: PASSED | FLAGGED | BLOCKED
  4. On FLAGGED: inject disclaimer into draft_report
  5. On BLOCKED: route back to synthesizer for re-generation

Node signature (LangGraph):
    def compliance_node(state: AgentState) -> dict
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.observability import trace_node
from src.state import AgentState, ComplianceResult, WorkflowStatus

# ---------------------------------------------------------------------------
# Deterministic rule patterns (Layer 2)
# ---------------------------------------------------------------------------

BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"\b(strong\s+buy|strong\s+sell)\b",              "Explicit buy/sell rating detected"),
    (r"\b(price\s+target|target\s+price)\s+of\s+\$",   "Price target prediction detected"),
    (r"\bI\s+recommend\s+(buying|selling|investing)\b", "Direct investment recommendation"),
    (r"\byou\s+should\s+(buy|sell|invest|short)\b",     "Direct investment instruction"),
]

FLAGGED_PATTERNS: list[tuple[str, str]] = [
    (r"\b(outperform|underperform|market\s+perform)\b",    "Rating language — disclaimer added"),
    (r"\bwill\s+(rise|fall|increase|decrease)\s+by\s+\d+", "Forward-looking prediction"),
    (r"\b(buy|sell)\s+opportunity\b",                      "Opportunity framing detected"),
    (r"\bupside\s+potential\s+of\s+\d+",                   "Upside target language"),
]

DISCLAIMER_INJECTION = (
    "\n\n> ⚠ **Compliance Notice:** This report has been automatically reviewed. "
    "Certain language was flagged as potentially resembling investment advice. "
    "All content is for informational research purposes only and does not "
    "constitute a recommendation to buy or sell any security."
)


def _run_nemo_rails(text: str, config_path: str) -> tuple[ComplianceResult, list[str]]:
    """
    Layer 1: Run NeMo Guardrails on the given text using colang rail definitions.

    Args:
        text:        Report text or query to validate.
        config_path: Path to the guardrails config directory.

    Returns:
        (ComplianceResult, list_of_violation_notes)
    """
    try:
        from nemoguardrails import LLMRails, RailsConfig
        rails_config = RailsConfig.from_path(config_path)
        rails        = LLMRails(rails_config)
        response     = rails.generate(
            messages=[{"role": "assistant", "content": text}]
        )
        if response != text:
            logger.warning("[compliance] NeMo rail fired — output modified")
            return ComplianceResult.FLAGGED, ["NeMo Guardrails: output modified by rail"]
        return ComplianceResult.PASSED, []
    except ImportError:
        logger.warning("[compliance] nemoguardrails not installed — skipping Layer 1")
        return ComplianceResult.PASSED, []
    except Exception as exc:
        logger.warning(f"[compliance] NeMo error ({exc}) — using regex fallback only")
        return ComplianceResult.PASSED, []


def _run_regex_check(text: str) -> tuple[ComplianceResult, list[str], str]:
    """
    Layer 2: Deterministic pattern matching against BLOCKED and FLAGGED rule sets.

    Args:
        text: Report text to scan.

    Returns:
        (ComplianceResult, list_of_violations, modified_text)
        modified_text has disclaimer appended when FLAGGED; unchanged otherwise.
    """
    violations:    list[str] = []
    modified_text: str       = text

    for pattern, description in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"BLOCKED: {description}")
            logger.error(f"[compliance][regex] {description}")

    if any(v.startswith("BLOCKED") for v in violations):
        return ComplianceResult.BLOCKED, violations, text

    flagged = False
    for pattern, description in FLAGGED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(f"FLAGGED: {description}")
            logger.warning(f"[compliance][regex] {description}")
            flagged = True

    if flagged:
        modified_text = text + DISCLAIMER_INJECTION
        return ComplianceResult.FLAGGED, violations, modified_text

    return ComplianceResult.PASSED, [], text


@trace_node("compliance")
def compliance_node(state: AgentState) -> dict:
    """
    LangGraph node: validate report content against financial compliance rules.

    Runs NeMo Guardrails (Layer 1) then deterministic regex (Layer 2).
    Takes the worst result across both layers.

    On first entry (no draft yet): validates the query intent pre-synthesis.
    On re-entry (draft exists):    validates the draft_report post-synthesis.

    Args:
        state: Current AgentState.

    Returns:
        Partial state update with compliance_result, compliance_notes,
        optionally updated draft_report, compliance_retries, node_timings.
    """
    t0         = time.perf_counter()
    session_id = state.get("session_id", "unknown")
    retries    = state.get("compliance_retries", 0)

    logger.info(f"[compliance] START | session={session_id} | retry={retries}")

    settings      = get_settings()
    guardrail_cfg = settings.guardrails_config
    config_path   = str(Path(guardrail_cfg["config_path"]).resolve())

    draft_report  = state.get("draft_report", "").strip()
    text_to_check = draft_report if draft_report else state.get("query_intent", state["user_query"])

    all_violations: list[str] = []
    final_result               = ComplianceResult.PASSED
    final_text                 = text_to_check
    severity = {
        ComplianceResult.PASSED:  0,
        ComplianceResult.FLAGGED: 1,
        ComplianceResult.BLOCKED: 2,
    }

    if guardrail_cfg.get("provider") == "nemo":
        nemo_result, nemo_violations = _run_nemo_rails(text_to_check, config_path)
        all_violations.extend(nemo_violations)
        if severity[nemo_result] > severity[final_result]:
            final_result = nemo_result

    regex_result, regex_violations, modified_text = _run_regex_check(text_to_check)
    all_violations.extend(regex_violations)

    if severity[regex_result] > severity[final_result]:
        final_result = regex_result

    if regex_result == ComplianceResult.FLAGGED:
        final_text = modified_text

    elapsed = time.perf_counter() - t0

    update: dict = {
        "compliance_result":  final_result,
        "compliance_notes":   all_violations,
        "compliance_retries": retries + 1,
        "status":             WorkflowStatus.COMPLIANCE_CHECK,
        "node_timings": {
            **state.get("node_timings", {}),
            "compliance": round(elapsed, 3),
        },
    }

    if draft_report and final_result == ComplianceResult.FLAGGED:
        update["draft_report"] = final_text
        logger.warning(f"[compliance] FLAGGED — disclaimer injected | {all_violations}")
    elif final_result == ComplianceResult.BLOCKED:
        logger.error(
            f"[compliance] BLOCKED | {all_violations} | "
            f"retry {retries + 1}/{guardrail_cfg['max_retries']}"
        )
    else:
        logger.success(f"[compliance] PASSED | {elapsed:.2f}s")

    return update