"""
tests/test_ui_components.py
============================
Unit tests for Day 4 — Streamlit UI components.

Streamlit components are tested by mocking `st.*` calls and verifying:
  - Correct session_state reads/writes
  - Correct calls to graph functions (run_graph, resume_graph)
  - Edge cases (empty state, missing fields)

Run:
    pytest tests/test_ui_components.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.state import ComplianceResult, WorkflowStatus, create_initial_state


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_st(monkeypatch):
    """Mock streamlit module so tests don't need a running Streamlit server."""
    st_mock = MagicMock()
    st_mock.session_state = {}
    monkeypatch.setattr("streamlit.session_state", st_mock.session_state)
    return st_mock


@pytest.fixture
def complete_session_state():
    """Realistic session_state after a full approved run."""
    state = create_initial_state(
        user_query="What is Apple gross margin Q4 2024?",
        session_id="test-ui-session-001",
    )
    return {
        "session_id":      "test-ui-session-001",
        "workflow_status": "complete",
        "active_provider": "gemini",
        "current_query":   "What is Apple gross margin Q4 2024?",
        "chat_history": [
            {"role": "user",      "content": "What is Apple gross margin Q4 2024?"},
            {"role": "assistant", "content": "Draft ready.", "is_markdown": True, "node_timings": {}},
        ],
        "paused_state": {
            **state,
            "ticker":            ["AAPL"],
            "draft_report":      "## Executive Summary\nStrong quarter.\n\n## Financial Highlights\n- GM: 46.2%",
            "report_sections":   {"executive_summary": "Strong quarter.", "financial_highlights": "GM: 46.2%"},
            "compliance_result": ComplianceResult.PASSED,
            "compliance_notes":  [],
            "retrieval_metadata": {"chunks_retrieved": 5, "top_rrf_score": 0.87},
            "node_timings":      {"fetcher": 1.2, "retriever": 0.8, "compliance": 0.3, "synthesizer": 3.1},
        },
        "final_report": (
            "# Financial Research Report\n\n"
            "| Field | Value |\n|---|---|\n| **Tickers** | AAPL |\n\n"
            "## Executive Summary\nStrong quarter.\n\n"
            "## Financial Highlights\n- Gross Margin: 46.2%\n\n"
            "---\n> **Compliance Disclaimer**\n> For informational purposes only."
        ),
        "eval_scores":  {},
        "eval_history": [],
    }


@pytest.fixture
def awaiting_hitl_state(complete_session_state):
    """Session state paused at HITL."""
    state = {**complete_session_state}
    state["workflow_status"] = "awaiting_hitl"
    state["final_report"]    = ""
    return state


# ---------------------------------------------------------------------------
# Tests — Sidebar helpers
# ---------------------------------------------------------------------------

class TestSidebarHelpers:

    def test_render_status_badge_idle(self) -> None:
        """_render_status_badge should not raise for any known status."""
        with patch("streamlit.markdown") as mock_md:
            from ui.components.sidebar import _render_status_badge
            for status in ["idle", "running", "awaiting_hitl", "complete", "error"]:
                _render_status_badge(status)
            assert mock_md.called

    def test_reset_session_clears_keys(self) -> None:
        """_reset_session should remove all tracked session keys."""
        import streamlit as st
        st.session_state["session_id"]      = "abc"
        st.session_state["workflow_status"] = "complete"
        st.session_state["chat_history"]    = [{"role": "user", "content": "test"}]

        with patch("streamlit.session_state", st.session_state):
            from ui.components.sidebar import _reset_session
            _reset_session()

        assert "session_id"      not in st.session_state
        assert "workflow_status" not in st.session_state
        assert "chat_history"    not in st.session_state

    def test_update_provider_in_memory(self) -> None:
        """_update_provider_in_memory should update settings._yaml without raising."""
        mock_settings        = MagicMock()
        mock_settings._yaml  = {"llm": {"active": "gemini"}}

        with patch("ui.components.sidebar.get_settings", return_value=mock_settings):
            # Import fresh to avoid cached state
            import importlib
            import ui.components.sidebar as sb
            importlib.reload(sb)
            sb._update_provider_in_memory("groq")

        assert mock_settings._yaml["llm"]["active"] == "groq"


# ---------------------------------------------------------------------------
# Tests — Chat Panel helpers
# ---------------------------------------------------------------------------

class TestChatPanelHelpers:

    def test_node_progress_covers_all_statuses(self) -> None:
        """NODE_PROGRESS dict should cover all key workflow stages."""
        from ui.components.chat_panel import NODE_PROGRESS
        expected = {
            "fetching", "retrieving", "compliance_check",
            "synthesizing", "awaiting_hitl", "complete", "error",
        }
        assert expected.issubset(set(NODE_PROGRESS.keys()))

    def test_node_progress_entries_are_tuples(self) -> None:
        from ui.components.chat_panel import NODE_PROGRESS
        for key, value in NODE_PROGRESS.items():
            assert isinstance(value, tuple), f"{key} should map to a (icon, label) tuple"
            assert len(value) == 2


# ---------------------------------------------------------------------------
# Tests — HITL Panel
# ---------------------------------------------------------------------------

class TestHITLPanel:

    def test_hitl_panel_does_not_render_when_idle(self) -> None:
        """HITL panel should be a no-op when workflow_status is not awaiting_hitl."""
        import streamlit as st
        st.session_state["workflow_status"] = "idle"
        st.session_state["paused_state"]    = {}

        with patch("streamlit.markdown") as mock_md, \
             patch("streamlit.button",   return_value=False):
            from ui.components.hitl_panel import render_hitl_panel
            render_hitl_panel("test-session")

        # No HITL banner should be rendered
        for call_args in mock_md.call_args_list:
            if call_args[0]:
                assert "ANALYST REVIEW REQUIRED" not in str(call_args[0][0])

    def test_hitl_panel_renders_when_awaiting(self, awaiting_hitl_state) -> None:
        """HITL panel should render when status is awaiting_hitl and draft exists."""
        import streamlit as st
        for k, v in awaiting_hitl_state.items():
            st.session_state[k] = v

        rendered_content = []
        with patch("streamlit.markdown", side_effect=lambda x, **kw: rendered_content.append(str(x))), \
             patch("streamlit.tabs",     return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch("streamlit.columns",  return_value=[MagicMock()] * 4), \
             patch("streamlit.button",   return_value=False), \
             patch("streamlit.text_area", return_value=""), \
             patch("streamlit.expander", return_value=MagicMock().__enter__()):
            from ui.components.hitl_panel import render_hitl_panel
            render_hitl_panel("test-ui-session-001")

        all_rendered = " ".join(rendered_content)
        assert "ANALYST REVIEW REQUIRED" in all_rendered

    def test_handle_hitl_decision_approve_calls_resume(self, awaiting_hitl_state) -> None:
        """Approving should call resume_graph with hitl_approved=True."""
        import streamlit as st
        for k, v in awaiting_hitl_state.items():
            st.session_state[k] = v

        mock_final_state = {
            **awaiting_hitl_state["paused_state"],
            "final_report": "# Final Report\nApproved content.",
            "status":       WorkflowStatus.COMPLETE,
            "node_timings": {},
        }

        with patch("ui.components.hitl_panel.resume_graph", return_value=mock_final_state) as mock_resume, \
             patch("ui.components.hitl_panel.flush"), \
             patch("streamlit.spinner", return_value=MagicMock().__enter__()), \
             patch("streamlit.success"), \
             patch("streamlit.rerun"):
            from ui.components.hitl_panel import _handle_hitl_decision
            _handle_hitl_decision(
                session_id="test-ui-session-001",
                approved=True,
                override_notes="",
            )

        mock_resume.assert_called_once_with(
            session_id="test-ui-session-001",
            hitl_approved=True,
            override_notes="",
        )
        assert st.session_state["workflow_status"] == "complete"
        assert st.session_state["final_report"]    == mock_final_state["final_report"]

    def test_handle_hitl_decision_reject_calls_resume(self, awaiting_hitl_state) -> None:
        """Rejecting should call resume_graph with hitl_approved=False + notes."""
        import streamlit as st
        for k, v in awaiting_hitl_state.items():
            st.session_state[k] = v

        new_draft_state = {
            **awaiting_hitl_state["paused_state"],
            "draft_report": "## Executive Summary\nRevised report.",
            "status":       WorkflowStatus.SYNTHESIZING,
            "node_timings": {},
        }

        with patch("ui.components.hitl_panel.resume_graph", return_value=new_draft_state) as mock_resume, \
             patch("ui.components.hitl_panel.flush"), \
             patch("streamlit.spinner", return_value=MagicMock().__enter__()), \
             patch("streamlit.info"), \
             patch("streamlit.rerun"):
            from ui.components.hitl_panel import _handle_hitl_decision
            _handle_hitl_decision(
                session_id="test-ui-session-001",
                approved=False,
                override_notes="Focus on Services margin only.",
            )

        mock_resume.assert_called_once_with(
            session_id="test-ui-session-001",
            hitl_approved=False,
            override_notes="Focus on Services margin only.",
        )
        assert st.session_state["workflow_status"] == "awaiting_hitl"


# ---------------------------------------------------------------------------
# Tests — Report Panel
# ---------------------------------------------------------------------------

class TestReportPanel:

    def test_report_panel_idle_renders_placeholder(self) -> None:
        """When idle with no report, placeholder should render."""
        import streamlit as st
        st.session_state["workflow_status"] = "idle"
        st.session_state["final_report"]    = ""

        rendered = []
        with patch("streamlit.markdown", side_effect=lambda x, **kw: rendered.append(str(x))):
            from ui.components.report_panel import render_report_panel
            render_report_panel("test-session")

        all_rendered = " ".join(rendered)
        assert "No report yet" in all_rendered

    def test_report_panel_renders_final_report(self, complete_session_state) -> None:
        """When complete, final report content should be rendered."""
        import streamlit as st
        for k, v in complete_session_state.items():
            st.session_state[k] = v

        rendered = []
        with patch("streamlit.markdown", side_effect=lambda x, **kw: rendered.append(str(x))), \
             patch("streamlit.columns",  return_value=[MagicMock()] * 3), \
             patch("streamlit.tabs",     return_value=[MagicMock(), MagicMock()]), \
             patch("streamlit.expander", return_value=MagicMock().__enter__()), \
             patch("streamlit.caption"), \
             patch("streamlit.download_button"), \
             patch("streamlit.button", return_value=False):
            from ui.components.report_panel import render_report_panel
            render_report_panel("test-ui-session-001")

        all_rendered = " ".join(rendered)
        assert "Final Report" in all_rendered or "Analyst Approved" in all_rendered


# ---------------------------------------------------------------------------
# Tests — Eval Dashboard helpers
# ---------------------------------------------------------------------------

class TestEvalDashboard:

    def test_thresholds_cover_all_metrics(self) -> None:
        from ui.components.eval_dashboard import THRESHOLDS, METRIC_LABELS
        assert set(THRESHOLDS.keys()) == set(METRIC_LABELS.keys())

    def test_demo_scores_pass_thresholds(self) -> None:
        from ui.components.eval_dashboard import _demo_scores, THRESHOLDS
        scores = _demo_scores()
        for metric, threshold in THRESHOLDS.items():
            if metric == "hallucination":
                assert scores[metric] <= threshold, f"{metric} demo score should be below threshold"
            else:
                assert scores[metric] >= threshold, f"{metric} demo score should meet threshold"

    def test_render_empty_dashboard_does_not_raise(self) -> None:
        with patch("streamlit.markdown"):
            from ui.components.eval_dashboard import _render_empty_dashboard
            _render_empty_dashboard()   # must not raise

    def test_demo_scores_all_present(self) -> None:
        from ui.components.eval_dashboard import _demo_scores, THRESHOLDS
        scores = _demo_scores()
        for metric in THRESHOLDS:
            assert metric in scores, f"Demo scores missing metric: {metric}"