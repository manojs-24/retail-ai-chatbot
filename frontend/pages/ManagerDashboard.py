"""
Sections:
  1. KPI strip          — Revenue, Orders, Customers, Products, Stock, Low-Stock
  2. Charts row         — Monthly Sales Trend | Category Revenue | Top Products
  3. ML Intelligence    — Forecast | Demand | Inventory Risk | Segments | Sentiment
  4. AI Insights        — Rule-based data-driven observations (no LLM)
  5. Data Tables        — Low-Stock Products | Top Customers
  6. AI Assistant CTA   — Prominent button linking to Manager Chatbot

"""

from __future__ import annotations

import sys
import time
from pathlib import Path


# Path + env bootstrap (must run before any backend imports)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import matplotlib
matplotlib.use("Agg")          # headless — no GUI window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from frontend.utils.auth import clear_session, require_role


# Page config (must be the very first Streamlit call)

st.set_page_config(
    page_title="Manager Dashboard — Retail AI",
    page_icon="📊",
    layout="wide",
)


# Guard

require_role("manager")


# Lazy backend import (after sys.path is ready)

from backend.core.database import SessionLocal
from backend.services.dashboard_service import get_dashboard_data


# Custom CSS — tighter KPI cards, coloured badges

st.markdown("""
<style>
.kpi-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #2e3250;
}
.kpi-label { font-size: 0.78rem; color: #9aa0b4; text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 1.65rem; font-weight: 700; color: #f0f2f8; margin: 4px 0; }
.kpi-sub   { font-size: 0.72rem; color: #6c7293; }
.badge-critical { background:#ff4b4b; color:#fff; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.badge-high     { background:#ff9800; color:#fff; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.badge-medium   { background:#ffc107; color:#000; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.badge-ok       { background:#28a745; color:#fff; padding:2px 8px; border-radius:12px; font-size:.75rem; }
.insight-card   { background:#1a1d2e; border-left:4px solid #4c6ef5; padding:10px 16px; border-radius:6px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)



# Sidebar

with st.sidebar:
    st.markdown(f"### 👤 {st.session_state['full_name']}")
    st.markdown(f"📧 `{st.session_state['email']}`")
    st.markdown(f"🏷️ Role: `{st.session_state['role']}`")
    st.divider()
    if st.button("💬 AI Assistant", width='stretch', type="primary", key="sb_chat"):
        st.switch_page("pages/manager_chat.py")
    if st.button("🔍 Validation", width='stretch', key="sb_validation"):
        st.switch_page("pages/Validation.py")
    st.divider()
    if st.button("🔄 Refresh Dashboard", width='stretch', key="sb_refresh"):
        st.cache_data.clear()
        st.rerun()
    if st.button("🚪 Logout", width='stretch', type="secondary", key="sb_logout"):
        clear_session()
        st.switch_page("Home.py")
    st.divider()
    st.caption("📊 Powered by scikit-learn + SQLAlchemy")



# Load dashboard data (cached 5 min)

@st.cache_data(ttl=300, show_spinner=False)
def _load_data() -> dict:
    db = SessionLocal()
    try:
        return get_dashboard_data(db)
    finally:
        db.close()



# Header + AI Assistant CTA

col_title, col_cta = st.columns([3, 1])
with col_title:
    st.title("📊 Store Manager Dashboard")
    st.markdown(
        f"Welcome, **{st.session_state['full_name']}** — "
        "here's your live store intelligence overview."
    )
with col_cta:
    st.markdown("<div style='padding-top:30px'>", unsafe_allow_html=True)
    if st.button("🤖 Open AI Assistant", width='stretch', type="primary", key="top_chat"):
        st.switch_page("pages/manager_chat.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()


# Load data with spinner

with st.spinner("Loading dashboard data…"):
    data = _load_data()

kpis   = data["kpis"]
charts = data["charts"]
tables = data["tables"]
ml     = data["ml"]
insights = data["insights"]



# SECTION 1 — KPI STRIP

st.subheader("📈 Key Performance Indicators")

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _kpi(col, label: str, value: str, sub: str = "") -> None:
    with col:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

_kpi(k1, "Total Revenue",    kpis["total_revenue_fmt"],    f"Avg ₹{kpis['avg_order_value']:,.0f}/order")
_kpi(k2, "Total Orders",     f"{kpis['total_orders']:,}",  "All time")
_kpi(k3, "Customers",        f"{kpis['total_customers']:,}","Active accounts")
_kpi(k4, "Active Products",  f"{kpis['total_products']:,}", "In catalogue")
_kpi(k5, "Inventory Units",  f"{kpis['total_stock']:,}",   "Total stock")
_kpi(k6, "Stock Status",        f"{kpis['low_stock_count']}, {kpis['out_of_stock_count']} ", "Low & out of stock")

st.markdown("<br>", unsafe_allow_html=True)



# SECTION 2 — CHARTS

st.subheader("📉 Sales & Inventory Charts")

chart_c1, chart_c2, chart_c3 = st.columns([2, 1, 1])

# --- Monthly Sales Trend ---
with chart_c1:
    st.markdown("**📅 Monthly Revenue Trend**")
    monthly = charts.get("monthly_sales", [])
    if monthly:
        df_m = pd.DataFrame(monthly)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.plot(df_m["label"], df_m["revenue"] / 1e5, marker="o", color="#4c6ef5",
                linewidth=2, markersize=5)
        ax.fill_between(range(len(df_m)), df_m["revenue"] / 1e5, alpha=0.15, color="#4c6ef5")
        ax.set_xticks(range(len(df_m)))
        ax.set_xticklabels(df_m["label"], rotation=45, ha="right", fontsize=7, color="#9aa0b4")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))
        ax.tick_params(axis="y", colors="#9aa0b4", labelsize=8)
        ax.spines[:].set_color("#2e3250")
        ax.grid(axis="y", color="#2e3250", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No monthly data available.")

# --- Category Revenue ---
with chart_c2:
    st.markdown("**🗂️ Revenue by Category**")
    cat_rev = charts.get("category_revenue", [])
    if cat_rev:
        df_c = pd.DataFrame(cat_rev).sort_values("revenue")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        colours = plt.cm.Blues_r([i / len(df_c) * 0.7 + 0.2 for i in range(len(df_c))])
        bars = ax.barh(df_c["category"], df_c["revenue"] / 1e5, color=colours)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}L"))
        ax.tick_params(axis="x", colors="#9aa0b4", labelsize=7)
        ax.tick_params(axis="y", colors="#9aa0b4", labelsize=8)
        ax.spines[:].set_color("#2e3250")
        ax.grid(axis="x", color="#2e3250", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No category data.")

# --- Top Selling Products ---
with chart_c3:
    st.markdown("**🏆 Top 8 Products (Units Sold)**")
    top_prods = charts.get("top_products", [])[:8]
    if top_prods:
        df_p = pd.DataFrame(top_prods).sort_values("units_sold")
        labels = [n[:22] + "…" if len(n) > 22 else n for n in df_p["product_name"]]
        fig, ax = plt.subplots(figsize=(4, 3.2))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.barh(labels, df_p["units_sold"], color="#51cf66")
        ax.tick_params(axis="x", colors="#9aa0b4", labelsize=7)
        ax.tick_params(axis="y", colors="#9aa0b4", labelsize=7)
        ax.spines[:].set_color("#2e3250")
        ax.grid(axis="x", color="#2e3250", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No product data.")

st.divider()



# SECTION 3 — ML INTELLIGENCE CARDS

st.subheader("🤖 ML Intelligence")

ml_c1, ml_c2, ml_c3, ml_c4, ml_c5 = st.columns(5)

# --- Sales Forecast ---
with ml_c1:
    fc = ml.get("forecast", {})
    trend_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}.get(fc.get("trend", "stable"), "➡️")
    trend_pct = fc.get("trend_pct", 0)
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">30-Day Sales Forecast</div>
        <div class="kpi-value">{fc.get('forecast_30d_fmt', 'N/A')}</div>
        <div class="kpi-sub">{trend_emoji} {abs(trend_pct):.1f}% MoM • {fc.get('model','')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# --- High Demand Products ---
with ml_c2:
    dm = ml.get("demand", {})
    hd = dm.get("high_demand", [])
    top_3 = "<br>".join(
        f"• {p.get('product_name','')[:22]}" for p in hd[:3]
    )
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">High Demand Products</div>
        <div class="kpi-value">{len(hd)}</div>
        <div class="kpi-sub">{top_3}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# --- Inventory Risk ---
with ml_c3:
    ir = ml.get("inventory_risk", {})
    badge_c = ir.get("critical_count", 0)
    badge_h = ir.get("high_count", 0)
    badge_m = ir.get("medium_count", 0)
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Inventory Risk Alerts</div>
        <div class="kpi-value">{badge_c + badge_h + badge_m}</div>
        <div class="kpi-sub">
            <span class="badge-critical">Critical: {badge_c}</span>&nbsp;
            <span class="badge-high">High: {badge_h}</span>&nbsp;
            <span class="badge-medium">Med: {badge_m}</span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

# --- Customer Segments ---
with ml_c4:
    seg = ml.get("segments", {})
    seg_list = seg.get("segments", [])
    seg_lines = "<br>".join(
        f"• {s['segment']}: {int(s['customer_count'])}" for s in seg_list
    )
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Customer Segments</div>
        <div class="kpi-value">{seg.get('total_customers', 0)}</div>
        <div class="kpi-sub">{seg_lines or 'N/A'}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# --- Sentiment Distribution ---
with ml_c5:
    sent = ml.get("sentiment", {})
    dist = sent.get("distribution", {})
    pos = dist.get("Positive", 0)
    neu = dist.get("Neutral", 0)
    neg = dist.get("Negative", 0)
    st.markdown(
        f"""<div class="kpi-card">
        <div class="kpi-label">Review Sentiment</div>
        <div class="kpi-value">{sent.get('positive_pct', 0):.0f}% Positive</div>
        <div class="kpi-sub">
            <span class="badge-ok">+{pos}</span>&nbsp;
            <span style="background:#888;color:#fff;padding:2px 8px;border-radius:12px;font-size:.75rem">~{neu}</span>&nbsp;
            <span class="badge-critical">-{neg}</span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)



# SECTION 4 — AI INSIGHTS

st.subheader("💡 AI Insights")
st.caption("Data-driven business observations — updated every 5 minutes")

for obs in insights:
    st.markdown(
        f'<div class="insight-card">{obs}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()



# SECTION 5 — DATA TABLES

st.subheader("📋 Data Tables")

tab1, tab2, tab3, tab4 = st.tabs(
    ["⚠️ Low Stock Products", "👥 Top Customers",
     "🔥 High Demand", "🚨 Inventory Risk Alerts"]
)

# --- Low Stock Products ---
with tab1:
    ls = tables.get("low_stock_products", [])
    if ls:
        df_ls = pd.DataFrame(ls)
        df_ls["Status"] = df_ls["status"].map(
            lambda s: "🔴 Out of Stock" if s == "Out of Stock" else "🟡 Low Stock"
        )
        st.dataframe(
            df_ls.rename(columns={
                "product_id": "ID", "product_name": "Product", "brand": "Brand",
                "category": "Category", "stock_quantity": "Stock",
                "reorder_level": "Reorder Lvl", "Status": "Status",
            })[["ID", "Product", "Brand", "Category", "Stock", "Reorder Lvl", "Status"]],
            width='stretch', hide_index=True,
        )
    else:
        st.success("✅ No low-stock products at this time.")

# --- Top Customers ---
with tab2:
    tc = tables.get("top_customers", [])
    if tc:
        df_tc = pd.DataFrame(tc)
        df_tc["total_spent_fmt"] = df_tc["total_spent"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(
            df_tc.rename(columns={
                "user_id": "ID", "full_name": "Name", "city": "City",
                "loyalty_level": "Tier", "order_count": "Orders",
                "total_spent_fmt": "Total Spent",
            })[["ID", "Name", "City", "Tier", "Orders", "Total Spent"]],
            width='stretch', hide_index=True,
        )
    else:
        st.info("No customer data available.")

# --- High Demand Products ---
with tab3:
    hd = ml.get("demand", {}).get("high_demand", [])
    if hd:
        df_hd = pd.DataFrame(hd)
        if "revenue" in df_hd.columns:
            df_hd["revenue_fmt"] = df_hd["revenue"].apply(lambda x: f"₹{x:,.0f}")
        cols = [c for c in ["product_id", "product_name", "brand", "category",
                             "units_sold", "revenue_fmt", "demand_score"] if c in df_hd.columns]
        st.dataframe(
            df_hd[cols].rename(columns={
                "product_id": "ID", "product_name": "Product", "brand": "Brand",
                "category": "Category", "units_sold": "Units Sold",
                "revenue_fmt": "Revenue", "demand_score": "Demand Score",
            }),
            width='stretch', hide_index=True,
        )
    else:
        st.info("No high-demand product data.")

# --- Inventory Risk Alerts ---
with tab4:
    alerts = ml.get("inventory_risk", {}).get("risk_alerts", [])
    if alerts:
        df_ir = pd.DataFrame(alerts)
        df_ir["risk_badge"] = df_ir["risk_level"].map({
            "CRITICAL": "🔴 CRITICAL",
            "HIGH": "🟠 HIGH",
            "MEDIUM": "🟡 MEDIUM",
        })
        cols = [c for c in ["product_id", "product_name", "brand", "category",
                             "stock_quantity", "daily_velocity", "days_remaining",
                             "risk_badge"] if c in df_ir.columns]
        st.dataframe(
            df_ir[cols].rename(columns={
                "product_id": "ID", "product_name": "Product", "brand": "Brand",
                "category": "Category", "stock_quantity": "Stock",
                "daily_velocity": "Daily Vel.", "days_remaining": "Days Left",
                "risk_badge": "Risk Level",
            }),
            width='stretch', hide_index=True,
        )
    else:
        st.success("✅ No inventory risk alerts.")

st.divider()



# SECTION 6 — BOTTOM AI ASSISTANT CTA

cta_l, cta_c, cta_r = st.columns([1, 2, 1])
with cta_c:
    st.markdown(
        "<h4 style='text-align:center;'>Need deeper insights?</h4>"
        "<p style='text-align:center;color:#9aa0b4;'>Ask your AI assistant anything about "
        "inventory, sales, customers, or forecasts.</p>",
        unsafe_allow_html=True,
    )
    if st.button(
        "🤖 Open AI Assistant",
        width='stretch',
        type="primary",
        key="bottom_chat",
    ):
        st.switch_page("pages/manager_chat.py")

st.divider()
st.caption(
    f"📊 Retail AI System · v1.0.0 · "
    f"Data refreshes every 5 minutes · "
    f"ML models: LinearRegression, RandomForest, KMeans, GradientBoosting"
)
