"""
ui/components/report_panel.py
==============================
Final Report Panel
-------------------
Renders the completed, analyst-approved financial report.
Only visible when workflow_status == "complete".

Features:
  - Full markdown render of the final report
  - Download as .md file
  - Copy-to-clipboard button
  - Report metadata summary strip
  - Node timing breakdown

Public API:
    from ui.components.report_panel import render_report_panel
    render_report_panel(session_id)
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_report_panel(session_id: str) -> None:
    """
    Render the final approved report panel.

    Only renders when workflow_status == "complete" and final_report is set.

    Args:
        session_id: Active session identifier (used for download filename).
    """
    status       = st.session_state.get("workflow_status", "idle")
    final_report = st.session_state.get("final_report", "")

    if status != "complete" or not final_report:
        if status == "idle":
            _render_idle_placeholder()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='background:#052e16; border:2px solid #10B981; border-radius:8px;
                    padding:14px 20px; margin-bottom:16px'>
            <div style='font-size:1.1rem; font-weight:700; color:#10B981'>
                ✅ Final Report — Analyst Approved
            </div>
            <div style='color:#6EE7B7; font-size:0.82rem; margin-top:3px'>
                This report has been reviewed and approved by the analyst.
                It includes a compliance disclaimer and is ready for distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Download strip ────────────────────────────────────────────────────────
    dl_col1, dl_col2, dl_col3 = st.columns([2, 2, 3])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"fin_report_{session_id[:8]}_{timestamp}.md"

    with dl_col1:
        st.download_button(
            label    = "⬇️ Download .md",
            data     = final_report,
            file_name= filename,
            mime     = "text/markdown",
            use_container_width=True,
        )

    with dl_col2:
        # Copy to clipboard via JS injection
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.write(
                f"<script>navigator.clipboard.writeText(`{final_report[:2000]}`);</script>",
                unsafe_allow_html=True,
            )
            st.toast("Report copied to clipboard!", icon="📋")

    with dl_col3:
        report_len = len(final_report)
        word_count = len(final_report.split())
        st.caption(f"📏 {report_len:,} chars · ~{word_count:,} words · Session: `{session_id[:12]}`")

    # ── Timing breakdown ──────────────────────────────────────────────────────
    paused_state = st.session_state.get("paused_state", {})
    timings      = paused_state.get("node_timings", {})
    if timings:
        with st.expander("⏱ Agent Node Timings", expanded=False):
            t_cols = st.columns(len(timings))
            total  = sum(timings.values())
            for i, (node, secs) in enumerate(timings.items()):
                pct = (secs / total * 100) if total > 0 else 0
                t_cols[i].metric(
                    label = node.title(),
                    value = f"{secs:.2f}s",
                    delta = f"{pct:.0f}% of total",
                    delta_color="off",
                )

    # ── Report render ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📄 Final Report")

    tab_render, tab_raw = st.tabs(["📄 Rendered", "📝 Raw Markdown"])

    with tab_render:
        st.markdown(final_report)

    with tab_raw:
        st.code(final_report, language="markdown")


def _render_idle_placeholder() -> None:
    """Show a placeholder card when no report exists yet."""
    st.markdown(
        """
        <div style='border:1px dashed #374151; border-radius:8px; padding:40px;
                    text-align:center; color:#6B7280; margin-top:20px'>
            <div style='font-size:2.5rem; margin-bottom:12px'>📊</div>
            <div style='font-size:1rem; font-weight:600; margin-bottom:6px'>
                No report yet
            </div>
            <div style='font-size:0.85rem'>
                Submit a financial research query in the Chat tab,<br>
                then approve the draft in the HITL panel to generate a final report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )