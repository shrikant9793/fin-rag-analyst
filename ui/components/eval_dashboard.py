"""
ui/components/eval_dashboard.py
================================
Evaluation Dashboard Component
--------------------------------
Visualises Ragas and DeepEval metric scores for the current session.

Features:
  - Gauge charts for Faithfulness, Answer Relevancy, Context Recall
  - DeepEval Hallucination and Contextual Precision scores
  - Threshold pass/fail badges
  - Run eval button (triggers eval_suite on current session)
  - Historical score trend (across sessions stored in session_state)

Public API:
    from ui.components.eval_dashboard import render_eval_dashboard
    render_eval_dashboard(session_id)
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Metric thresholds (mirrored from config.yaml)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "faithfulness":          0.80,
    "answer_relevancy":      0.75,
    "context_recall":        0.70,
    "hallucination":         0.15,   # lower is better
    "contextual_precision":  0.75,
}

METRIC_LABELS = {
    "faithfulness":         "Faithfulness",
    "answer_relevancy":     "Answer Relevancy",
    "context_recall":       "Context Recall",
    "hallucination":        "Hallucination",
    "contextual_precision": "Contextual Precision",
}


# ---------------------------------------------------------------------------
# render_eval_dashboard
# ---------------------------------------------------------------------------

def render_eval_dashboard(session_id: str) -> None:
    """
    Render the evaluation metrics dashboard.

    Args:
        session_id: Current session ID (used to tag eval run).
    """
    st.markdown("### 📊 Evaluation Dashboard")
    st.caption(
        "Ragas + DeepEval metrics measure the quality of RAG retrieval and LLM generation. "
        "Run evaluation after a report is complete to score the session."
    )

    eval_scores: dict = st.session_state.get("eval_scores", {})
    workflow_status   = st.session_state.get("workflow_status", "idle")

    # ── Run Eval Button ───────────────────────────────────────────────────────
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        run_eval = st.button(
            "▶ Run Evaluation",
            use_container_width=True,
            type="primary",
            disabled=(workflow_status not in ("complete",)),
            help="Only available after a report has been approved and compiled.",
        )
    with col_info:
        if workflow_status != "complete":
            st.info("ℹ Complete a report first to enable evaluation.", icon="ℹ️")

    if run_eval:
        eval_scores = _run_eval_suite(session_id)
        st.session_state["eval_scores"] = eval_scores

    # ── Scores Display ────────────────────────────────────────────────────────
    if not eval_scores:
        _render_empty_dashboard()
        return

    st.markdown("---")
    st.markdown("#### Ragas Metrics")
    _render_gauge_row(
        scores = {
            "faithfulness":     eval_scores.get("faithfulness",     0.0),
            "answer_relevancy": eval_scores.get("answer_relevancy", 0.0),
            "context_recall":   eval_scores.get("context_recall",   0.0),
        },
        lower_is_better_keys=set(),
    )

    st.markdown("#### DeepEval Metrics")
    _render_gauge_row(
        scores = {
            "hallucination":       eval_scores.get("hallucination",       0.0),
            "contextual_precision": eval_scores.get("contextual_precision", 0.0),
        },
        lower_is_better_keys={"hallucination"},
    )

    # ── Pass / Fail Summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ✅ Threshold Summary")
    _render_threshold_table(eval_scores)

    # ── Score history sparkline ───────────────────────────────────────────────
    score_history: list[dict] = st.session_state.get("eval_history", [])
    if len(score_history) > 1:
        st.markdown("---")
        st.markdown("#### 📈 Score History")
        _render_score_history(score_history)


# ---------------------------------------------------------------------------
# Gauge chart renderer
# ---------------------------------------------------------------------------

def _render_gauge_row(
    scores: dict[str, float],
    lower_is_better_keys: set[str],
) -> None:
    """Render a row of Plotly gauge charts for given metric scores."""
    cols = st.columns(len(scores))

    for col, (metric_key, score) in zip(cols, scores.items()):
        label     = METRIC_LABELS.get(metric_key, metric_key)
        threshold = THRESHOLDS.get(metric_key, 0.75)
        lower_ok  = metric_key in lower_is_better_keys

        # For hallucination: pass = below threshold
        if lower_ok:
            passed  = score <= threshold
            bar_col = "#10B981" if passed else "#EF4444"
        else:
            passed  = score >= threshold
            bar_col = "#10B981" if passed else "#EF4444"

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = score,
            title = {"text": label, "font": {"size": 13}},
            number= {"suffix": "", "font": {"size": 22}},
            gauge = {
                "axis":      {"range": [0, 1], "tickwidth": 1},
                "bar":       {"color": bar_col},
                "bgcolor":   "#1F2937",
                "steps": [
                    {"range": [0, threshold], "color": "#374151"},
                    {"range": [threshold, 1], "color": "#1F4D3A" if not lower_ok else "#4D1F1F"},
                ],
                "threshold": {
                    "line":  {"color": "#F59E0B", "width": 3},
                    "value": threshold,
                },
            },
        ))
        fig.update_layout(
            height     = 200,
            margin     = dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color = "#E5E7EB",
        )

        with col:
            st.plotly_chart(fig, use_container_width=True)
            badge = "✅ PASS" if passed else "❌ FAIL"
            badge_colour = "#10B981" if passed else "#EF4444"
            st.markdown(
                f"<div style='text-align:center; color:{badge_colour}; "
                f"font-weight:600; font-size:0.85rem'>{badge} "
                f"(threshold: {'≤' if lower_ok else '≥'}{threshold})</div>",
                unsafe_allow_html=True,
            )


def _render_threshold_table(scores: dict[str, float]) -> None:
    """Render a compact pass/fail table for all metrics."""
    rows = []
    all_passed = True
    for metric_key, threshold in THRESHOLDS.items():
        score      = scores.get(metric_key, None)
        lower_ok   = metric_key == "hallucination"
        if score is None:
            continue
        passed     = (score <= threshold) if lower_ok else (score >= threshold)
        all_passed = all_passed and passed
        rows.append({
            "Metric":    METRIC_LABELS.get(metric_key, metric_key),
            "Score":     f"{score:.3f}",
            "Threshold": f"{'≤' if lower_ok else '≥'}{threshold}",
            "Status":    "✅ PASS" if passed else "❌ FAIL",
        })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        overall_colour = "#10B981" if all_passed else "#EF4444"
        overall_label  = "✅ ALL METRICS PASS — Ready for CI/CD gate" if all_passed \
                         else "❌ SOME METRICS BELOW THRESHOLD — Review before deployment"
        st.markdown(
            f"<div style='background:{overall_colour}22; border:1px solid {overall_colour}; "
            f"padding:10px 16px; border-radius:6px; color:{overall_colour}; "
            f"font-weight:600; margin-top:8px'>{overall_label}</div>",
            unsafe_allow_html=True,
        )


def _render_score_history(history: list[dict]) -> None:
    """Render a multi-line sparkline of score history across sessions."""
    import plotly.express as px
    import pandas as pd

    rows = []
    for i, entry in enumerate(history):
        for metric, score in entry.items():
            if metric == "session_id":
                continue
            rows.append({"Run": i + 1, "Metric": METRIC_LABELS.get(metric, metric), "Score": score})

    if rows:
        df  = pd.DataFrame(rows)
        fig = px.line(
            df, x="Run", y="Score", color="Metric",
            markers=True,
            title="Eval Score Trend",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font_color   = "#E5E7EB",
            height       = 280,
            margin       = dict(l=0, r=0, t=40, b=0),
            yaxis        = dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_empty_dashboard() -> None:
    """Placeholder shown when no eval scores exist."""
    st.markdown(
        """
        <div style='border:1px dashed #374151; border-radius:8px; padding:40px;
                    text-align:center; color:#6B7280; margin-top:16px'>
            <div style='font-size:2rem; margin-bottom:12px'>📐</div>
            <div style='font-size:0.95rem; font-weight:600'>No evaluation scores yet</div>
            <div style='font-size:0.82rem; margin-top:6px'>
                Complete and approve a report, then click <strong>▶ Run Evaluation</strong>
                to score Faithfulness, Answer Relevancy, Context Recall,
                Hallucination, and Contextual Precision.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Eval runner (calls eval_suite.py logic in-process)
# ---------------------------------------------------------------------------

def _run_eval_suite(session_id: str) -> dict[str, float]:
    """
    Run Ragas + DeepEval against the current session's RAG output.

    Uses the current session's retrieved_docs and final_report as the
    evaluation dataset (single-sample eval for the live session).

    Returns:
        Dict of metric_name → float scores.
    """
    paused_state = st.session_state.get("paused_state", {})
    final_report = st.session_state.get("final_report", "")
    user_query   = st.session_state.get("current_query", "")

    if not final_report or not user_query:
        st.error("Cannot run eval — no completed report found in session.")
        return {}

    retrieved_docs = paused_state.get("retrieved_docs", [])
    contexts       = [doc.page_content for doc in retrieved_docs]

    with st.spinner("Running Ragas + DeepEval evaluation…"):
        try:
            from tests.eval_suite import run_single_sample_eval
            scores = run_single_sample_eval(
                question   = user_query,
                answer     = final_report,
                contexts   = contexts,
                session_id = session_id,
            )

            # Persist to history
            history = st.session_state.get("eval_history", [])
            history.append({**scores, "session_id": session_id})
            st.session_state["eval_history"] = history

            st.success("✅ Evaluation complete!")
            return scores

        except ImportError:
            st.warning(
                "⚠ eval_suite.py not yet available (Day 5). "
                "Showing demo scores for UI preview."
            )
            return _demo_scores()
        except Exception as exc:
            st.error(f"Eval failed: {exc}")
            return {}


def _demo_scores() -> dict[str, float]:
    """Return placeholder demo scores for UI preview before Day 5."""
    return {
        "faithfulness":          0.87,
        "answer_relevancy":      0.82,
        "context_recall":        0.76,
        "hallucination":         0.09,
        "contextual_precision":  0.81,
    }