"""
ui/components/sidebar.py
========================
Streamlit Sidebar Component
----------------------------
Renders the left sidebar containing:
  - LLM provider selector (Gemini / Groq / Ollama)
  - Active session info and status badge
  - Quick links to Qdrant and Langfuse dashboards
  - Document ingestion trigger (upload + ingest)
  - Session reset button

Public API:
    from ui.components.sidebar import render_sidebar
    provider, session_id = render_sidebar()
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# Provider colour map
# ---------------------------------------------------------------------------
PROVIDER_COLOURS = {
    "gemini": "#4285F4",
    "groq":   "#F55036",
    "ollama": "#6BBD45",
}

PROVIDER_LABELS = {
    "gemini": "🔵 Gemini Flash (Google AI Studio)",
    "groq":   "🔴 Groq — Llama 3.1 70B",
    "ollama": "🟢 Ollama — Llama 3.1 8B (Local)",
}


# ---------------------------------------------------------------------------
# render_sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str]:
    """
    Render the full sidebar UI.

    Returns:
        (active_provider, session_id)
        active_provider: one of "gemini" | "groq" | "ollama"
        session_id:      current session UUID string
    """
    with st.sidebar:
        # --- Branding ---
        st.markdown(
            """
            <div style='text-align:center; padding: 8px 0 16px'>
                <span style='font-size:2rem'>📊</span><br>
                <strong style='font-size:1.1rem'>Financial Analyst</strong><br>
                <span style='color:#888; font-size:0.75rem'>Multi-Agent RAG System</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # --- LLM Provider Selector ---
        st.markdown("#### ⚙️ LLM Provider")
        provider_options = list(PROVIDER_LABELS.keys())
        provider_index   = provider_options.index(
            st.session_state.get("active_provider", "gemini")
        )

        selected_provider = st.radio(
            label      = "Select provider",
            options    = provider_options,
            index      = provider_index,
            format_func= lambda p: PROVIDER_LABELS[p],
            label_visibility="collapsed",
        )

        # Persist selection
        if selected_provider != st.session_state.get("active_provider"):
            st.session_state["active_provider"] = selected_provider
            # Update config.yaml active provider at runtime (in-memory only)
            _update_provider_in_memory(selected_provider)
            st.toast(f"Provider switched to **{PROVIDER_LABELS[selected_provider]}**", icon="🔄")

        colour = PROVIDER_COLOURS.get(selected_provider, "#888")
        st.markdown(
            f"<div style='background:{colour}22; border-left:3px solid {colour}; "
            f"padding:6px 10px; border-radius:4px; font-size:0.8rem; color:{colour}'>"
            f"Active: {PROVIDER_LABELS[selected_provider]}</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Session Info ---
        st.markdown("#### 🔑 Session")
        session_id = st.session_state.get("session_id", "")
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:8]}"
            st.session_state["session_id"] = session_id

        st.code(session_id, language=None)

        workflow_status = st.session_state.get("workflow_status", "idle")
        _render_status_badge(workflow_status)

        if st.button("🔄 New Session", use_container_width=True):
            _reset_session()
            st.rerun()

        st.divider()

        # --- Document Ingestion ---
        st.markdown("#### 📄 Ingest Documents")
        uploaded_files = st.file_uploader(
            "Upload SEC filings / Earnings transcripts",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            ticker_input = st.text_input(
                "Ticker symbol", placeholder="AAPL", key="ingest_ticker"
            ).upper()
            doc_type = st.selectbox(
                "Document type",
                ["sec_filing", "earnings_transcript", "research_note"],
                key="ingest_doc_type",
            )
            filing_date = st.date_input("Filing date", key="ingest_date")

            if st.button("⬆️ Ingest Documents", use_container_width=True, type="primary"):
                if not ticker_input:
                    st.error("Please enter a ticker symbol.")
                else:
                    _run_ingestion(
                        uploaded_files=uploaded_files,
                        ticker=ticker_input,
                        doc_type=doc_type,
                        filing_date=str(filing_date),
                    )

        st.divider()

        # --- External Links ---
        st.markdown("#### 🔗 Dashboards")
        st.markdown(
            "[📡 Qdrant Dashboard](http://localhost:6333/dashboard)  \n"
            "[📈 Langfuse Traces](http://localhost:3000)  \n"
            "[📚 Project README](https://github.com/your-org/fin-rag-analyst)",
        )

        st.divider()
        st.caption("v0.1.0 · Multi-Agent RAG · Day 4")

    return selected_provider, session_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_status_badge(status: str) -> None:
    """Render a coloured status pill based on workflow status string."""
    colour_map = {
        "idle":           ("#888",    "⬜ Idle"),
        "running":        ("#3B82F6", "🔵 Running"),
        "awaiting_hitl":  ("#F59E0B", "🟡 Awaiting Approval"),
        "hitl_approved":  ("#10B981", "🟢 Approved"),
        "hitl_rejected":  ("#EF4444", "🔴 Rejected — Re-drafting"),
        "complete":       ("#10B981", "✅ Complete"),
        "error":          ("#EF4444", "❌ Error"),
    }
    colour, label = colour_map.get(status, ("#888", f"⬜ {status.title()}"))
    st.markdown(
        f"<div style='background:{colour}22; border:1px solid {colour}; "
        f"padding:4px 10px; border-radius:20px; font-size:0.8rem; "
        f"color:{colour}; text-align:center; margin:4px 0'>{label}</div>",
        unsafe_allow_html=True,
    )


def _update_provider_in_memory(provider: str) -> None:
    """
    Temporarily override the LLM provider in the settings singleton.
    This updates the in-memory config without touching config.yaml on disk.
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        settings._yaml["llm"]["active"] = provider
    except Exception:
        pass   # Non-critical — provider change visible on next page load


def _reset_session() -> None:
    """Clear all session-scoped state keys."""
    keys_to_clear = [
        "session_id", "workflow_status", "paused_state",
        "chat_history", "final_report", "draft_report",
        "eval_scores",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _run_ingestion(
    uploaded_files: list,
    ticker: str,
    doc_type: str,
    filing_date: str,
) -> None:
    """
    Save uploaded files to a temp directory and call RAGPipeline.ingest_documents().
    Shows a Streamlit progress bar during ingestion.
    """
    import tempfile

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    pipeline.ensure_collection()

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        for f in uploaded_files:
            dest = Path(tmp_dir) / f.name
            dest.write_bytes(f.getvalue())
            saved_paths.append(dest)

        doc_metadata = {
            "ticker":       ticker,
            "doc_type":     doc_type,
            "filing_date":  filing_date,
            "company_name": "",
        }

        with st.spinner(f"Ingesting {len(saved_paths)} document(s)…"):
            try:
                total = pipeline.ingest_documents(
                    file_paths=saved_paths,
                    doc_metadata=doc_metadata,
                )
                st.success(f"✅ Ingested {total} chunks for **{ticker}**")
                stats = pipeline.collection_stats()
                st.caption(f"Total collection size: {stats['points_count']} chunks")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")