"""
ui/components/hitl_panel.py
============================
Human-in-the-Loop (HITL) Approval Panel
-----------------------------------------
The most critical UI component — renders when the LangGraph graph has
PAUSED at the `hitl_node` (interrupt_before) and is awaiting an analyst
decision before the final report is compiled.

Layout:
  ┌─────────────────────────────────────────────┐
  │  ★ ANALYST REVIEW REQUIRED                  │
  │  ─────────────────────────────────────────  │
  │  [Draft Report — full markdown render]       │
  │                                             │
  │  Compliance flags (if any)                  │
  │  Retrieval metadata                         │
  │                                             │
  │  [Override Notes text area]                 │
  │                                             │
  │  [ ✅ Approve ]     [ ❌ Reject & Revise ]   │
  └─────────────────────────────────────────────┘

Public API:
    from ui.components.hitl_panel import render_hitl_panel
    render_hitl_panel(session_id)
"""

from __future__ import annotations

import streamlit as st

from src.state import ComplianceResult


def render_hitl_panel(session_id: str) -> None:
    """
    Render the HITL analyst approval panel.

    Only renders when `st.session_state["workflow_status"] == "awaiting_hitl"`.
    Reads the paused state from `st.session_state["paused_state"]`.

    On Approve → calls resume_graph(hitl_approved=True)  → triggers compiler
    On Reject  → calls resume_graph(hitl_approved=False) → triggers re-synthesis

    Args:
        session_id: Active LangGraph thread ID.
    """
    status = st.session_state.get("workflow_status", "idle")

    if status not in ("awaiting_hitl", "hitl_rejected"):
        return

    paused_state: dict = st.session_state.get("paused_state", {})
    draft_report        = paused_state.get("draft_report", "")

    if not draft_report:
        return

    # ── HITL Banner ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style='background:#1a1200; border:2px solid #F59E0B; border-radius:8px;
                    padding:16px 20px; margin-bottom:16px'>
            <div style='font-size:1.2rem; font-weight:700; color:#F59E0B'>
                ★ ANALYST REVIEW REQUIRED
            </div>
            <div style='color:#9CA3AF; font-size:0.85rem; margin-top:4px'>
                The AI has produced a draft report. Review it carefully before approving.
                Your decision resumes or re-routes the agent graph.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metadata strip ────────────────────────────────────────────────────────
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    ret_meta = paused_state.get("retrieval_metadata", {})
    compliance_result = paused_state.get("compliance_result", ComplianceResult.PASSED)
    compliance_notes  = paused_state.get("compliance_notes", [])
    synthesis_retries = paused_state.get("synthesis_retries", 0)
    tickers           = paused_state.get("ticker", [])

    meta_col1.metric("Tickers",         ", ".join(tickers) or "N/A")
    meta_col2.metric("Chunks Retrieved", ret_meta.get("chunks_retrieved", 0))
    meta_col3.metric("Synthesis Retries", synthesis_retries)
    meta_col4.metric("Compliance",       str(compliance_result.value).upper()
                     if hasattr(compliance_result, "value") else str(compliance_result))

    # ── Compliance flags ──────────────────────────────────────────────────────
    if compliance_notes:
        comp_colour = (
            "#EF4444" if any("BLOCKED" in n for n in compliance_notes)
            else "#F59E0B"
        )
        with st.expander("🛡️ Compliance Flags", expanded=True):
            for note in compliance_notes:
                st.markdown(
                    f"<div style='color:{comp_colour}; font-size:0.85rem; "
                    f"padding:2px 0'>⚠ {note}</div>",
                    unsafe_allow_html=True,
                )

    # ── Draft Report ──────────────────────────────────────────────────────────
    st.markdown("#### 📋 Draft Report")

    # Tab view: rendered markdown vs raw text
    tab_render, tab_raw, tab_sections = st.tabs(
        ["📄 Rendered", "📝 Raw Markdown", "🗂 Sections"]
    )

    with tab_render:
        st.markdown(draft_report)

    with tab_raw:
        st.code(draft_report, language="markdown")

    with tab_sections:
        sections: dict = paused_state.get("report_sections", {})
        if sections:
            for section_key, section_content in sections.items():
                label = section_key.replace("_", " ").title()
                with st.expander(f"📌 {label}", expanded=False):
                    st.markdown(section_content or "_No content_")
        else:
            st.info("Sections not parsed — check synthesizer output.")

    # ── Override Notes ────────────────────────────────────────────────────────
    st.markdown("#### 📝 Override Notes *(optional)*")
    override_notes = st.text_area(
        label       = "Analyst corrections or instructions for re-synthesis",
        placeholder = (
            "e.g. 'Focus only on Services segment margin. Remove the China revenue breakdown. "
            "Ensure all figures are cited to the source document.'"
        ),
        height      = 100,
        key         = "override_notes_input",
        help        = "These notes are injected into the synthesizer prompt on rejection.",
    )

    # ── Action Buttons ────────────────────────────────────────────────────────
    st.markdown("")
    btn_col1, btn_spacer, btn_col2 = st.columns([2, 1, 2])

    with btn_col1:
        approve_clicked = st.button(
            "✅ Approve & Compile Final Report",
            use_container_width=True,
            type="primary",
            key="hitl_approve_btn",
        )

    with btn_col2:
        reject_clicked = st.button(
            "❌ Reject — Request Revision",
            use_container_width=True,
            type="secondary",
            key="hitl_reject_btn",
        )

    # ── Handle decisions ──────────────────────────────────────────────────────
    if approve_clicked:
        _handle_hitl_decision(
            session_id     = session_id,
            approved       = True,
            override_notes = override_notes.strip(),
        )

    if reject_clicked:
        if not override_notes.strip():
            st.warning("⚠ Please enter override notes to guide the re-synthesis.")
        else:
            _handle_hitl_decision(
                session_id     = session_id,
                approved       = False,
                override_notes = override_notes.strip(),
            )


# ---------------------------------------------------------------------------
# HITL decision handler
# ---------------------------------------------------------------------------

def _handle_hitl_decision(
    session_id:     str,
    approved:       bool,
    override_notes: str,
) -> None:
    """
    Resume the LangGraph graph with the analyst's decision.

    Calls resume_graph() which:
      - On approval  → runs compiler_node → sets final_report
      - On rejection → re-runs synthesizer_node with override_notes

    Updates session state and triggers st.rerun() to refresh the UI.

    Args:
        session_id:     LangGraph thread ID.
        approved:       True = analyst approved, False = rejected.
        override_notes: Analyst corrections for re-synthesis.
    """
    from src.graph import resume_graph
    from src.observability import flush

    decision_label = "Approving" if approved else "Rejecting"

    with st.spinner(f"{decision_label} draft and resuming graph…"):
        try:
            final_state = resume_graph(
                session_id     = session_id,
                hitl_approved  = approved,
                override_notes = override_notes,
            )
            flush()

            if approved:
                # Graph ran to completion
                st.session_state["workflow_status"] = "complete"
                st.session_state["final_report"]    = final_state.get("final_report", "")
                st.session_state["paused_state"]    = {}

                # Append final report to chat history
                history: list = st.session_state.get("chat_history", [])
                history.append({
                    "role":        "assistant",
                    "content":     (
                        "✅ **Final report compiled and approved.**\n\n"
                        "Scroll to the **📊 Eval Dashboard** tab to see quality metrics,\n"
                        "or download the report below."
                    ),
                    "is_markdown": True,
                    "node_timings": final_state.get("node_timings", {}),
                })
                st.session_state["chat_history"] = history
                st.success("✅ Report approved and compiled!")

            else:
                # Graph re-ran synthesizer — new draft ready for review
                st.session_state["workflow_status"] = "awaiting_hitl"
                st.session_state["paused_state"]    = final_state
                st.info("🔄 Report re-synthesised with your notes. Review the updated draft above.")

        except Exception as exc:
            st.session_state["workflow_status"] = "error"
            st.error(f"❌ Resume failed: {exc}")

    st.rerun()