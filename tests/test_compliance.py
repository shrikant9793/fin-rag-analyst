"""
tests/test_compliance.py
========================
Unit tests for Day 3 — Compliance Agent and Observability layer.

Run:
    pytest tests/test_compliance.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.state import AgentState, ComplianceResult, WorkflowStatus, create_initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_state() -> AgentState:
    return create_initial_state(
        user_query="What is Apple's gross margin for Q4 2024?",
        session_id="test-compliance-001",
    )


@pytest.fixture
def state_with_clean_draft(clean_state: AgentState) -> AgentState:
    return {
        **clean_state,
        "ticker":       ["AAPL"],
        "query_intent": "Analyse Apple gross margin Q4 2024",
        "draft_report": (
            "## Executive Summary\n"
            "Apple delivered strong Q4 FY2024 results with revenue of $94.9B.\n\n"
            "## Financial Highlights\n"
            "- Revenue: $94.9B (+6% YoY) [AAPL_Q4_2024.txt]\n"
            "- Gross Margin: 46.2% [AAPL_Q4_2024.txt]\n"
            "- Net Income: $14.7B [AAPL_Q4_2024.txt]\n\n"
            "## Risk Factors\nMacroeconomic headwinds in Greater China.\n\n"
            "## Data Sources\n- AAPL_Q4_2024.txt (earnings_transcript)"
        ),
    }


@pytest.fixture
def state_with_blocked_draft(clean_state: AgentState) -> AgentState:
    return {
        **clean_state,
        "draft_report": (
            "## Executive Summary\n"
            "I recommend buying AAPL stock given strong Q4 results.\n\n"
            "## Financial Highlights\n- Revenue: $94.9B\n\n"
            "## Data Sources\n- AAPL_Q4_2024.txt"
        ),
    }


@pytest.fixture
def state_with_flagged_draft(clean_state: AgentState) -> AgentState:
    return {
        **clean_state,
        "draft_report": (
            "## Executive Summary\n"
            "Apple is expected to outperform peers in Q1 2025.\n\n"
            "## Financial Highlights\n- Gross Margin: 46.2%\n\n"
            "## Data Sources\n- AAPL_Q4_2024.txt"
        ),
    }


# ---------------------------------------------------------------------------
# Tests — _run_regex_check (Layer 2)
# ---------------------------------------------------------------------------

class TestRegexCheck:

    def test_clean_text_passes(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, text = _run_regex_check(
            "Apple revenue was $94.9B with gross margin of 46.2%."
        )
        assert result   == ComplianceResult.PASSED
        assert violations == []

    def test_strong_buy_is_blocked(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, _ = _run_regex_check(
            "Based on our analysis, this is a strong buy opportunity."
        )
        assert result == ComplianceResult.BLOCKED
        assert any("BLOCKED" in v for v in violations)

    def test_recommend_buying_is_blocked(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, _ = _run_regex_check(
            "I recommend buying AAPL at current levels."
        )
        assert result == ComplianceResult.BLOCKED

    def test_you_should_sell_is_blocked(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, _ = _run_regex_check(
            "You should sell your position before earnings."
        )
        assert result == ComplianceResult.BLOCKED

    def test_outperform_is_flagged(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, modified = _run_regex_check(
            "Apple is expected to outperform the S&P 500."
        )
        assert result == ComplianceResult.FLAGGED
        assert any("FLAGGED" in v for v in violations)
        assert "Compliance Notice" in modified

    def test_buy_opportunity_is_flagged(self) -> None:
        from src.agents.compliance import _run_regex_check
        result, violations, _ = _run_regex_check(
            "This presents a buy opportunity for long-term investors."
        )
        assert result == ComplianceResult.FLAGGED

    def test_blocked_takes_priority_over_flagged(self) -> None:
        from src.agents.compliance import _run_regex_check
        # Text has both blocked AND flagged patterns
        result, violations, _ = _run_regex_check(
            "Strong buy. Apple will outperform the market."
        )
        # BLOCKED must take priority
        assert result == ComplianceResult.BLOCKED

    def test_flagged_appends_disclaimer(self) -> None:
        from src.agents.compliance import _run_regex_check
        _, _, modified = _run_regex_check(
            "Apple is expected to outperform the S&P 500."
        )
        assert "Compliance Notice" in modified
        assert "informational research purposes only" in modified

    def test_passed_text_unchanged(self) -> None:
        from src.agents.compliance import _run_regex_check
        original = "Revenue was $94.9B with net income of $14.7B."
        _, _, modified = _run_regex_check(original)
        assert modified == original


# ---------------------------------------------------------------------------
# Tests — compliance_node (full node)
# ---------------------------------------------------------------------------

class TestComplianceNode:

    def _mock_settings(self):
        m = MagicMock()
        m.guardrails_config = {
            "provider":    "nemo",
            "config_path": "config/guardrails",
            "max_retries": 2,
        }
        return m

    def test_node_passes_clean_draft(self, state_with_clean_draft: AgentState) -> None:
        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state_with_clean_draft)

        assert result["compliance_result"] == ComplianceResult.PASSED
        assert result["compliance_notes"]  == []
        assert result["compliance_retries"] == 1

    def test_node_blocks_bad_draft(self, state_with_blocked_draft: AgentState) -> None:
        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state_with_blocked_draft)

        assert result["compliance_result"] == ComplianceResult.BLOCKED
        assert len(result["compliance_notes"]) > 0

    def test_node_flags_soft_violation(self, state_with_flagged_draft: AgentState) -> None:
        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state_with_flagged_draft)

        assert result["compliance_result"] == ComplianceResult.FLAGGED
        assert "Compliance Notice" in result["draft_report"]

    def test_node_increments_retries(self, state_with_clean_draft: AgentState) -> None:
        state = {**state_with_clean_draft, "compliance_retries": 1}

        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state)

        assert result["compliance_retries"] == 2

    def test_node_records_timing(self, state_with_clean_draft: AgentState) -> None:
        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state_with_clean_draft)

        assert "compliance" in result["node_timings"]
        assert result["node_timings"]["compliance"] >= 0

    def test_node_validates_query_when_no_draft(self, clean_state: AgentState) -> None:
        """When no draft_report exists, node should validate query_intent instead."""
        state = {**clean_state, "query_intent": "Analyse Apple gross margin data"}

        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.PASSED, [])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state)

        # Clean query — should pass
        assert result["compliance_result"] == ComplianceResult.PASSED

    def test_nemo_flagged_overrides_regex_passed(self, state_with_clean_draft: AgentState) -> None:
        """NeMo FLAGGED result should propagate even if regex passes."""
        with patch("src.agents.compliance.get_settings", return_value=self._mock_settings()), \
             patch("src.agents.compliance._run_nemo_rails",
                   return_value=(ComplianceResult.FLAGGED, ["NeMo rail fired"])), \
             patch("src.observability.get_langfuse_client", return_value=None):

            from src.agents.compliance import compliance_node
            result = compliance_node(state_with_clean_draft)

        assert result["compliance_result"] == ComplianceResult.FLAGGED
        assert "NeMo rail fired" in result["compliance_notes"]


# ---------------------------------------------------------------------------
# Tests — Observability helpers
# ---------------------------------------------------------------------------

class TestObservability:

    def test_get_langfuse_client_returns_none_when_disabled(self) -> None:
        mock_settings = MagicMock()
        mock_settings.observability_config = {
            "provider": "langfuse",
            "langfuse": {"enabled": False, "flush_at": 10, "flush_interval": 5.0},
        }
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host       = "http://localhost:3000"

        with patch("src.observability.get_settings", return_value=mock_settings), \
             patch("src.observability._langfuse_client", None):
            from src.observability import get_langfuse_client
            # Reset cache
            import src.observability as obs_module
            obs_module._langfuse_client = None

            client = get_langfuse_client()
            assert client is None

    def test_langfuse_tracer_no_op_when_client_none(self) -> None:
        """LangfuseTracer must not raise when client is None."""
        from src.observability import LangfuseTracer

        with patch("src.observability.get_langfuse_client", return_value=None):
            tracer = LangfuseTracer("test_node", session_id="sess-001")
            with tracer as t:
                t.update(output={"result": "ok"})
            # No exception = pass

    def test_trace_node_decorator_passes_result_through(self) -> None:
        """@trace_node decorator must not alter the node's return value."""
        from src.observability import trace_node

        @trace_node("test_node")
        def dummy_node(state: dict) -> dict:
            return {"status": "complete", "value": 42}

        with patch("src.observability.get_langfuse_client", return_value=None):
            result = dummy_node({"session_id": "s1", "user_query": "test", "node_timings": {}})

        assert result["status"] == "complete"
        assert result["value"]  == 42

    def test_flush_does_not_raise_when_client_none(self) -> None:
        with patch("src.observability.get_langfuse_client", return_value=None):
            from src.observability import flush
            flush()   # must not raise

    def test_log_eval_scores_does_not_raise_when_client_none(self) -> None:
        with patch("src.observability.get_langfuse_client", return_value=None):
            from src.observability import log_eval_scores
            log_eval_scores("sess-001", {"faithfulness": 0.85, "answer_relevancy": 0.78})