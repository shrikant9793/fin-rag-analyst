"""
ui/components/chat_panel.py
============================
Chat Panel Component
---------------------
Renders the main analyst chat interface:
  - Message history (user + assistant bubbles)
  - Query input box with submit button
  - Live streaming status updates as the graph runs
  - Intermediate node progress indicators

Public API:
    from ui.components.chat_panel import render_chat_panel
    render_chat_panel(session_id)
"""

from __future__ import annotations

import streamlit as st

from src.state import WorkflowStatus

# ---------------------------------------------------------------------------
# Node progress labels shown during streaming
# ---------------------------------------------------------------------------
NODE_PROGRESS = {
    "fetching":         ("🔍", "Parsing query — extracting ticker & intent…"),
    "retrieving":       ("📚", "Running hybrid search across financial documents…"),
    "compliance_check": ("🛡️",  "Checking compliance rails…"),
    "synthesizing":     ("✍️",  "Synthesising financial research report…"),
    "awaiting_hitl":    ("⏸️",  "Draft ready — awaiting analyst approval…"),
    "hitl_approved":    ("✅",  "Analyst approved — compiling final report…"),
    "hitl_rejected":    ("🔄",  "Analyst requested revision — re-synthesising…"),
    "compiling":        ("📝",  "Finalising report with compliance stamp…"),
    "complete":         ("🎉",  "Report complete!"),
    "error":            ("❌",  "An error occurred. Check logs."),
}


# ---------------------------------------------------------------------------
# render_chat_panel
# ---------------------------------------------------------------------------

def render_chat_panel(session_id: str) -> None:
    """
    Render the full chat panel — history + input + run trigger.

    Reads from st.session_state:
        chat_history, workflow_status, paused_state

    Writes to st.session_state:
        chat_history, workflow_status, paused_state
    """
    st.markdown("### 💬 Research Query")

    # --- Chat history ---
    chat_history: list[dict] = st.session_state.get("chat_history", [])
    _render_history(chat_history)

    # --- Query input ---
    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        user_query = st.text_area(
            label           = "Ask a financial research question",
            placeholder     = "e.g. What is Apple's gross margin trend for Q3 and Q4 FY2024?",
            height          = 80,
            key             = "query_input",
            label_visibility= "collapsed",
        )

    with col_btn:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        submit = st.button(
            "▶ Run",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.get("workflow_status") == "running",
        )

    # --- Submit logic ---
    if submit and user_query.strip():
        _run_query(user_query.strip(), session_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_history(history: list[dict]) -> None:
    """Render chat message bubbles from history list."""
    for msg in history:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        icon    = "🧑‍💼" if role == "user" else "🤖"

        with st.chat_message(role, avatar=icon):
            if role == "assistant" and msg.get("is_markdown", False):
                st.markdown(content)
            else:
                st.write(content)

            # Show node timings if present
            if timings := msg.get("node_timings"):
                with st.expander("⏱ Node timings", expanded=False):
                    cols = st.columns(len(timings))
                    for i, (node, secs) in enumerate(timings.items()):
                        cols[i].metric(node, f"{secs:.2f}s")


def _run_query(user_query: str, session_id: str) -> None:
    """
    Kick off a new graph run for the given query.

    Streams node-level status updates in real time using a Streamlit
    status container, then stores the paused state for the HITL panel.
    """
    from src.graph import run_graph

    # Add user message to history
    history: list[dict] = st.session_state.get("chat_history", [])
    history.append({"role": "user", "content": user_query})
    st.session_state["chat_history"]     = history
    st.session_state["workflow_status"]  = "running"
    st.session_state["current_query"]    = user_query

    # Stream graph execution with live status updates
    with st.status("🚀 Agent pipeline running…", expanded=True) as status_box:
        try:
            _stream_node_updates(status_box)
            paused_state, _ = run_graph(
                user_query = user_query,
                session_id = session_id,
            )

            st.session_state["paused_state"]    = paused_state
            st.session_state["workflow_status"] = "awaiting_hitl"

            # Add assistant "draft ready" message
            draft_len   = len(paused_state.get("draft_report", ""))
            chunks_ret  = paused_state.get("retrieval_metadata", {}).get("chunks_retrieved", 0)
            tickers     = paused_state.get("ticker", [])

            history.append({
                "role":        "assistant",
                "content":     (
                    f"📋 **Draft report ready for your review.**\n\n"
                    f"- **Tickers analysed:** {', '.join(tickers) or 'N/A'}\n"
                    f"- **Chunks retrieved:** {chunks_ret}\n"
                    f"- **Draft length:** {draft_len:,} characters\n\n"
                    f"👇 Scroll to the **HITL Approval Panel** below to review and approve."
                ),
                "is_markdown":  True,
                "node_timings": paused_state.get("node_timings", {}),
            })
            st.session_state["chat_history"] = history

            status_box.update(
                label    = "✅ Draft ready — awaiting analyst approval",
                state    = "complete",
                expanded = False,
            )

        except Exception as exc:
            st.session_state["workflow_status"] = "error"
            status_box.update(label=f"❌ Error: {exc}", state="error")
            history.append({
                "role":    "assistant",
                "content": f"❌ Pipeline error: {exc}",
            })
            st.session_state["chat_history"] = history

    st.rerun()


def _stream_node_updates(status_box) -> None:
    """
    Write live node progress messages into the Streamlit status container.
    Called before run_graph() to display which nodes are about to execute.
    """
    nodes_in_order = [
        "fetching",
        "retrieving",
        "compliance_check",
        "synthesizing",
        "awaiting_hitl",
    ]
    for node_key in nodes_in_order:
        icon, label = NODE_PROGRESS.get(node_key, ("⚙️", node_key))
        status_box.write(f"{icon} {label}")