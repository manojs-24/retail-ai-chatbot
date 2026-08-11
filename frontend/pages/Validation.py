from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Ensure project root is importable when Streamlit runs the page directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import pandas as pd
import streamlit as st

from frontend.utils.auth import clear_session, get_role, require_login

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Validation — Retail AI",
    page_icon="🔍",
    layout="wide",
)

# Auth guard — any logged-in user may view the validation page
require_login()

# Sidebar
role = get_role()
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.get('full_name', 'User')}")
    st.markdown(f"📧 {st.session_state.get('email', '')}")
    st.markdown(f"🏷️ Role: `{st.session_state.get('role', '')}`")
    st.divider()
    if role == "manager":
        if st.button("🏠 Dashboard", width="stretch"):
            st.switch_page("pages/ManagerDashboard.py")
        if st.button("💬 AI Assistant", width="stretch", type="primary"):
            st.switch_page("pages/manager_chat.py")
    else:
        if st.button("🏠 Dashboard", width="stretch"):
            st.switch_page("pages/CustomerDashboard.py")
        if st.button("💬 AI Shopping Assistant", width="stretch", type="primary"):
            st.switch_page("pages/customer_chat.py")
    st.divider()
    if st.button("🚪 Logout", width="stretch", type="secondary"):
        clear_session()
        st.switch_page("Home.py")
    st.divider()
    st.caption("🔍 AI System Validation")


# Shared helpers
THRESHOLDS = {
    "good": 0.80,
    "needs_improvement": 0.60,
}


def _status_label(score: float) -> str:
    """Return a coloured status label based on project-level thresholds."""
    if score >= THRESHOLDS["good"]:
        return "🟢 Good"
    if score >= THRESHOLDS["needs_improvement"]:
        return "🟡 Needs Improvement"
    return "🔴 Poor"


def _to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# Page header
st.title("🔍 AI System Validation")
st.markdown(
    "Evaluate the **accuracy** and **reliability** of the Retail AI system. "
    "Results are stored in session state and reset on page refresh."
)
st.divider()

# Run Validation button
col_run, col_info = st.columns([1, 4])
with col_run:
    run_clicked = st.button(
        "▶ Run Validation", type="primary", use_container_width=True
    )
with col_info:
    st.info(
        "Runs RAG (RAGAS 0.2.15), SQL intent, and ML forecasting evaluations "
        "against the live system. RAG evaluation calls OpenAI and takes ~3–5 minutes "
        "for 25 questions."
    )

# Run all evaluations when button is clicked
if run_clicked:
    for key in [
        "eval_rag", "eval_rag_error",
        "eval_sql", "eval_sql_error",
        "eval_ml",  "eval_ml_error",
    ]:
        st.session_state.pop(key, None)

    progress = st.progress(0, text="Starting validation…")

    # ── 1. RAG evaluation (RAGAS 0.2.15) ──────────────────────────────
    progress.progress(5, text="Running RAG evaluation (RAGAS 0.2.15)…")
    try:
        from backend.evaluation.rag_evaluation import run_rag_evaluation
        rag_result = run_rag_evaluation()
        st.session_state["eval_rag"] = rag_result
        logger.info("RAG evaluation completed — overall=%.3f", rag_result["overall_score"])
    except Exception as exc:  # noqa: BLE001
        st.session_state["eval_rag_error"] = str(exc)
        logger.exception("RAG evaluation failed")
    progress.progress(50, text="RAG evaluation done.")

    # ── 2. SQL / Intent evaluation ─────────────────────────────────────
    progress.progress(55, text="Running SQL / Intent evaluation…")
    try:
        from backend.evaluation.sql_evaluation import run_sql_evaluation
        sql_result = run_sql_evaluation()
        st.session_state["eval_sql"] = sql_result
        logger.info("SQL evaluation completed")
    except Exception as exc:  # noqa: BLE001
        st.session_state["eval_sql_error"] = str(exc)
        logger.exception("SQL evaluation failed")
    progress.progress(80, text="SQL evaluation done.")

    # ── 3. ML forecasting evaluation ───────────────────────────────────
    progress.progress(85, text="Running ML forecasting evaluation…")
    try:
        from backend.evaluation.ml_evaluation import run_ml_evaluation
        ml_result = run_ml_evaluation()
        st.session_state["eval_ml"] = ml_result
        logger.info("ML evaluation completed")
    except Exception as exc:  # noqa: BLE001
        st.session_state["eval_ml_error"] = str(exc)
        logger.exception("ML evaluation failed")

    progress.progress(100, text="Validation complete.")
    progress.empty()
    st.success("✅ Validation complete — see results below.")



# Retrieve results from session state

rag_result   = st.session_state.get("eval_rag")      # dict | None
rag_error    = st.session_state.get("eval_rag_error") # str | None
sql_result   = st.session_state.get("eval_sql")
sql_error    = st.session_state.get("eval_sql_error")
ml_result    = st.session_state.get("eval_ml")
ml_error     = st.session_state.get("eval_ml_error")



# SECTION 0 — OVERALL SYSTEM HEALTH SUMMARY

st.divider()
st.header("📊 Overall System Validation")

nothing_run = (
    rag_result is None and sql_result is None and ml_result is None
    and not rag_error and not sql_error and not ml_error
)
if nothing_run:
    st.info("Click **▶ Run Validation** to evaluate the system.")
else:
    col_rag_s, col_sql_s, col_ml_s = st.columns(3)

    with col_rag_s:
        st.markdown("#### 🤖 RAG Pipeline")
        if rag_error:
            st.error("❌ Failed")
            st.caption(f"Error: {rag_error[:120]}")
        elif rag_result:
            score = rag_result["overall_score"]
            st.metric(
                "Overall RAG Score",
                f"{score:.3f}",
                delta=_status_label(score),
            )
        else:
            st.info("Not run yet")

    with col_sql_s:
        st.markdown("#### 🗄️ SQL Retrieval")
        if sql_error:
            st.error("❌ Failed")
            st.caption(f"Error: {sql_error[:120]}")
        elif sql_result:
            ia = sql_result.intent_accuracy
            st.metric("Intent Accuracy", f"{ia:.2f}", delta=_status_label(ia))
        else:
            st.info("Not run yet")

    with col_ml_s:
        st.markdown("#### 📈 ML Forecasting")
        if ml_error:
            st.error("❌ Failed")
            st.caption(f"Error: {ml_error[:120]}")
        elif ml_result:
            if not ml_result.sufficient_data:
                st.warning("⚠️ Insufficient data")
            else:
                mape = ml_result.mape
                if mape <= 10:
                    status = "🟢 Good (MAPE ≤ 10%)"
                elif mape <= 25:
                    status = "🟡 Needs Improvement"
                else:
                    status = "🔴 Poor (MAPE > 25%)"
                st.metric("MAPE", f"{mape:.1f}%", delta=status)
        else:
            st.info("Not run yet")



# SECTION 1 — RAG EVALUATION (RAGAS 0.2.15)

st.divider()
st.header("📚 RAG Evaluation")
st.caption(
    "Powered by **RAGAS 0.2.15** · 25 policy-document questions · "
    "Real ChromaDB retrieval + production RAG answer generation"
)

# ── Run RAG only button ────────────────────────────────────────────────────
col_rag_btn, _ = st.columns([1, 4])
with col_rag_btn:
    if st.button("▶ Run RAG Evaluation", use_container_width=True):
        st.session_state.pop("eval_rag", None)
        st.session_state.pop("eval_rag_error", None)
        with st.spinner("Running RAGAS evaluation — this takes a few minutes…"):
            try:
                from backend.evaluation.rag_evaluation import run_rag_evaluation
                rag_result = run_rag_evaluation()
                st.session_state["eval_rag"] = rag_result
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.session_state["eval_rag_error"] = str(exc)
                st.rerun()

if rag_error:
    st.error("❌ RAG evaluation failed.")
    st.code(rag_error)

elif rag_result:
    st.success(
        f"✅ RAGAS evaluation complete — "
        f"{rag_result['evaluated_questions']} questions evaluated"
        + (
            f", {rag_result['failed_questions']} failed"
            if rag_result["failed_questions"]
            else ""
        )
        + "."
    )

    # ── Top metrics row ───────────────────────────────────────────────
    col_os, col_f, col_ar, col_cp, col_cr, col_ac = st.columns(6)
    col_os.metric(
        "Overall RAG Score",
        f"{rag_result['overall_score']:.3f}",
        delta=f"{rag_result['overall_percentage']:.1f}%",
    )
    col_f.metric(
        "Faithfulness",
        f"{rag_result['faithfulness']:.3f}",
        delta=_status_label(rag_result["faithfulness"]),
    )
    col_ar.metric(
        "Answer Relevancy",
        f"{rag_result['answer_relevancy']:.3f}",
        delta=_status_label(rag_result["answer_relevancy"]),
    )
    col_cp.metric(
        "Context Precision",
        f"{rag_result['context_precision']:.3f}",
        delta=_status_label(rag_result["context_precision"]),
    )
    col_cr.metric(
        "Context Recall",
        f"{rag_result['context_recall']:.3f}",
        delta=_status_label(rag_result["context_recall"]),
    )
    col_ac.metric(
        "Answer Correctness",
        f"{rag_result['answer_correctness']:.3f}",
        delta=_status_label(rag_result["answer_correctness"]),
    )

    # ── Bar chart ─────────────────────────────────────────────────────
    st.subheader("📊 Metric Scores")
    chart_df = pd.DataFrame(
        {
            "Score": [
                rag_result["faithfulness"],
                rag_result["answer_relevancy"],
                rag_result["context_precision"],
                rag_result["context_recall"],
                rag_result["answer_correctness"],
            ]
        },
        index=[
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",
            "Answer Correctness",
        ],
    )
    st.bar_chart(chart_df, y="Score", use_container_width=True)

    # ── Detailed results expander ─────────────────────────────────────
    with st.expander("📋 Detailed RAG Results", expanded=False):
        detail_df: pd.DataFrame = rag_result["details"]
        st.dataframe(detail_df, use_container_width=True)
        st.download_button(
            label="⬇️ Download RAG Results (CSV)",
            data=_to_csv(detail_df),
            file_name="rag_evaluation_results.csv",
            mime="text/csv",
            key="dl_rag",
        )

    # ── Low-scoring cases ─────────────────────────────────────────────
    detail_df = rag_result["details"]
    metric_cols = [
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
        "Answer Correctness",
    ]
    existing_metric_cols = [c for c in metric_cols if c in detail_df.columns]

    low_rows = []
    for _, row in detail_df.iterrows():
        for col in existing_metric_cols:
            val = row.get(col, 1.0)
            try:
                if float(val) < THRESHOLDS["needs_improvement"]:
                    low_rows.append({
                        "Question": row.get("Question", ""),
                        "Metric": col,
                        "Score": round(float(val), 3),
                        "Answer": str(row.get("Answer", ""))[:200],
                    })
            except (TypeError, ValueError):
                pass

    if low_rows:
        st.subheader(f"⚠️ Low-Scoring Cases (score < {THRESHOLDS['needs_improvement']})")
        st.caption(
            "These are project-level interpretation thresholds, "
            "not official RAGAS standards."
        )
        st.dataframe(pd.DataFrame(low_rows), use_container_width=True)
    else:
        st.success("🎉 No low-scoring cases — all metrics are above the threshold.")

    # ── Failed cases (if any) ─────────────────────────────────────────
    if rag_result.get("failures"):
        with st.expander(
            f"❌ {len(rag_result['failures'])} failed test case(s)", expanded=False
        ):
            st.dataframe(pd.DataFrame(rag_result["failures"]), use_container_width=True)

else:
    st.info("RAG results will appear here after running validation.")



# SECTION 2 — SQL VALIDATION

st.divider()
st.header("🗄️ Section 2 — SQL Retrieval Validation")

# ── Run SQL only button ────────────────────────────────────────────────────
col_sql_btn, _ = st.columns([1, 4])
with col_sql_btn:
    if st.button("▶ Run SQL Evaluation", use_container_width=True):
        st.session_state.pop("eval_sql", None)
        st.session_state.pop("eval_sql_error", None)
        with st.spinner("Running SQL / Intent evaluation…"):
            try:
                from backend.evaluation.sql_evaluation import run_sql_evaluation
                sql_result = run_sql_evaluation()
                st.session_state["eval_sql"] = sql_result
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.session_state["eval_sql_error"] = str(exc)
                st.rerun()

if sql_error:
    st.error("❌ SQL evaluation failed. Check the evaluation logs.")
    st.code(sql_error)
elif sql_result:
    st.success(
        f"✅ SQL evaluation completed — {sql_result.total_cases} test cases evaluated."
    )

    col_ia, col_ea, col_ra, col_roa = st.columns(4)
    col_ia.metric(
        "Intent Accuracy",
        f"{sql_result.intent_accuracy:.2f}",
        delta=_status_label(sql_result.intent_accuracy),
    )
    col_ea.metric(
        "Entity Accuracy",
        f"{sql_result.entity_accuracy:.2f}",
        delta=_status_label(sql_result.entity_accuracy),
    )
    col_ra.metric(
        "Retrieval Accuracy",
        f"{sql_result.retrieval_accuracy:.2f}",
        delta=_status_label(sql_result.retrieval_accuracy),
    )
    col_roa.metric(
        "Routing Accuracy",
        f"{sql_result.routing_accuracy:.2f}",
        delta=_status_label(sql_result.routing_accuracy),
    )

    if not sql_result.failures_df.empty:
        st.warning(f"⚠️ {len(sql_result.failures_df)} test case(s) failed.")
    else:
        st.success("🎉 All test cases passed!")

    with st.expander("📋 SQL Detailed Results", expanded=False):
        st.subheader("All Test Cases")
        st.dataframe(sql_result.detail_df, use_container_width=True)

        if not sql_result.failures_df.empty:
            st.subheader("❌ Failed Cases")
            failure_cols = [
                "Case ID", "Role", "Query",
                "Expected Intent", "Actual Intent",
                "Expected Product ID", "Actual Product ID",
                "Expected Order ID", "Actual Order ID",
                "Expected User ID", "Actual User ID",
                "Expected Route", "Actual Route",
                "Status",
            ]
            existing = [c for c in failure_cols if c in sql_result.failures_df.columns]
            st.dataframe(sql_result.failures_df[existing], use_container_width=True)

        st.download_button(
            label="⬇️ Download SQL Results (CSV)",
            data=_to_csv(sql_result.detail_df),
            file_name="sql_evaluation_results.csv",
            mime="text/csv",
            key="dl_sql",
        )
else:
    st.info("SQL results will appear here after running validation.")



# SECTION 3 — ML FORECASTING VALIDATION

st.divider()
st.header("📈 Section 3 — ML Forecasting Validation")

# ── Run ML only button ─────────────────────────────────────────────────────
col_ml_btn, _ = st.columns([1, 4])
with col_ml_btn:
    if st.button("▶ Run ML Evaluation", use_container_width=True):
        st.session_state.pop("eval_ml", None)
        st.session_state.pop("eval_ml_error", None)
        with st.spinner("Running ML forecasting evaluation…"):
            try:
                from backend.evaluation.ml_evaluation import run_ml_evaluation
                ml_result = run_ml_evaluation()
                st.session_state["eval_ml"] = ml_result
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.session_state["eval_ml_error"] = str(exc)
                st.rerun()

if ml_error:
    st.error("❌ ML evaluation failed. Check the evaluation logs.")
    st.code(ml_error)
elif ml_result:
    if not ml_result.sufficient_data:
        st.warning(ml_result.warning)
    else:
        st.success(
            f"✅ ML evaluation completed — {ml_result.test_months} test month(s) "
            f"held out from {ml_result.total_months} total months."
        )

        col_mae, col_rmse, col_mape, col_r2 = st.columns(4)
        col_mae.metric("MAE (₹)", f"₹{ml_result.mae:,.2f}")
        col_rmse.metric("RMSE (₹)", f"₹{ml_result.rmse:,.2f}")
        col_mape.metric("MAPE (%)", f"{ml_result.mape:.2f}%")
        col_r2.metric("R²", f"{ml_result.r2:.3f}")

        with st.expander("📋 ML Detailed Results — Actual vs Predicted", expanded=True):
            st.subheader("Actual vs Predicted Revenue")
            st.dataframe(
                ml_result.actuals_vs_predicted_df, use_container_width=True
            )
            if not ml_result.actuals_vs_predicted_df.empty:
                chart_df = ml_result.actuals_vs_predicted_df.set_index("Month")
                st.line_chart(chart_df)

            st.download_button(
                label="⬇️ Download ML Results (CSV)",
                data=_to_csv(ml_result.actuals_vs_predicted_df),
                file_name="ml_evaluation_results.csv",
                mime="text/csv",
                key="dl_ml",
            )
else:
    st.info("ML results will appear here after running validation.")


# Footer
st.divider()
st.caption(
    "Retail AI Validation Page · Results reset on page refresh · "
    "RAG uses RAGAS 0.2.15 with real ChromaDB retrieval · "
    "SQL tests live intent classification · ML uses walk-forward chronological split · "
    "Score thresholds (≥0.80 Good / 0.60–0.79 Needs Improvement / <0.60 Poor) "
    "are project-level guidelines, not official RAGAS standards."
)
