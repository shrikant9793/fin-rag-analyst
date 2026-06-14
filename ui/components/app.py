"""
ui/app.py
=========
Main Streamlit Application Entry Point
----------------------------------------
Financial Market Research Analyst — Multi-Agent RAG System

Layout:
  ┌──────────────┬──────────────────────────────────────────┐
  │   Sidebar    │  Tab 1: 💬 Chat + HITL Approval Panel    │
  │              │  Tab 2: 📄 Final Report                  │
  │  - Provider  │  Tab 3: 📊 Evaluation Dashboard          │
  │  - Session   │                                          │
  │  - Ingest    │                                          │
  └──────────────┴──────────────────────────────────────────┘

Run:
    streamlit run ui/app.py
    streamlit run ui/app.py --server.port 8502
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (works whether run from root or ui/ dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title     = "Financial Analyst — Multi-Agent RAG",
    page_icon      = "📊",
    layout         = "wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/your-org/fin-rag-analyst",
        "Report a bug":"https://github.com/your-org/fin-rag-analyst/issues",
        "About":       "Multi-Agent RAG Financial Analyst v0.1.0",
    },
)

# ---------------------------------------------------------------------------
# Global CSS — dark finance theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Base ── */
    .stApp { background-color: #0B0F14; color: #E2EAF4; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111720;
        border-right: 1px solid #1E2A38;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111720;
        border-radius: 6px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        color: #8A9BB0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E2A38 !important;
        color: #00D4AA !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background-color: #00D4AA;
        color: #0B0F14;
        font-weight: 600;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #00B894;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #111720;
        border: 1px solid #1E2A38;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background-color: #111720;
        border: 1px solid #1E2A38;
        border-radius: 6px;
        padding: 10px;
    }

    /* ── Code blocks ── */
    .stCode { background-color: #0D1520 !important; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background-color: #111720;
        border-radius: 4px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0B0F14; }
    ::-webkit-scrollbar-thumb { background: #1E2A38; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise all session state keys with safe defaults on first load."""
    defaults = {
        "chat_history":    [],
        "workflow_status": "idle",
        "session_id":      "",
        "active_provider": "gemini",
        "paused_state":    {},
        "final_report":    "",
        "eval_scores":     {},
        "eval_history":    [],
        "current_query":   "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    from ui.components.sidebar import render_sidebar
    active_provider, session_id = render_sidebar()

    # ── Top header ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:14px; margin-bottom:8px'>
            <span style='font-size:2.2rem'>📊</span>
            <div>
                <div style='font-size:1.5rem; font-weight:700; color:#E2EAF4; line-height:1.2'>
                    Financial Market Research Analyst
                </div>
                <div style='color:#4A5568; font-size:0.8rem'>
                    LangGraph · Qdrant · {active_provider.title()} · NeMo Guardrails · Langfuse
                    &nbsp;|&nbsp; Session: <code style='color:#00D4AA'>{session_id[:16]}…</code>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_chat, tab_report, tab_eval = st.tabs([
        "💬 Chat & Research",
        "📄 Final Report",
        "📊 Eval Dashboard",
    ])

    # ── Tab 1: Chat + HITL ────────────────────────────────────────────────────
    with tab_chat:
        from ui.components.chat_panel import render_chat_panel
        render_chat_panel(session_id)

        # HITL panel renders directly below chat when awaiting approval
        from ui.components.hitl_panel import render_hitl_panel
        render_hitl_panel(session_id)

    # ── Tab 2: Final Report ───────────────────────────────────────────────────
    with tab_report:
        from ui.components.report_panel import render_report_panel
        render_report_panel(session_id)

    # ── Tab 3: Eval Dashboard ─────────────────────────────────────────────────
    with tab_eval:
        from ui.components.eval_dashboard import render_eval_dashboard
        render_eval_dashboard(session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()