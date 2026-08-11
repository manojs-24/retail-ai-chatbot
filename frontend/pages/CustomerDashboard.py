from __future__ import annotations

import sys
from pathlib import Path

# Path + env bootstrap
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st

from frontend.utils.auth import clear_session, require_role


# Page-level helpers  (defined first so they are in scope everywhere below)

def _fmt_inr(value: float) -> str:
    """Format a float as a compact Indian Rupee string (e.g. ₹1.5 L, ₹2.3 Cr)."""
    if value >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"₹{value / 1_00_000:.1f} L"
    return f"₹{value:,.0f}"


def _loyalty_rate(level: str) -> str:
    """Return the reward-points multiplier label for a loyalty tier."""
    rates = {"Bronze": "1×", "Silver": "1.5×", "Gold": "2×", "Platinum": "3×"}
    return rates.get(level, "1×")


# Page config (must be the very first Streamlit call)
st.set_page_config(
    page_title="My Dashboard — Retail AI",
    page_icon="🛍️",
    layout="wide",
)

# Guard
require_role("customer")

# Lazy backend imports
from backend.core.database import SessionLocal
from backend.services.customer_dashboard_service import get_dashboard_data
from backend.services.recommendation_service import get_recommended_products

# Custom CSS
st.markdown("""
<style>
.kpi-card {
    background: #1a1d2e;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    border: 1px solid #2e3250;
}
.kpi-label { font-size: 0.76rem; color: #9aa0b4; text-transform: uppercase; letter-spacing: .05em; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #f0f2f8; margin: 4px 0 2px; }
.kpi-sub   { font-size: 0.70rem; color: #6c7293; }

.welcome-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #23264a 100%);
    border-radius: 12px;
    padding: 20px 28px;
    border: 1px solid #3a3f6e;
    margin-bottom: 6px;
}
.loyalty-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.loyalty-Platinum { background:#a855f7; color:#fff; }
.loyalty-Gold     { background:#f59e0b; color:#000; }
.loyalty-Silver   { background:#94a3b8; color:#000; }
.loyalty-Bronze   { background:#b45309; color:#fff; }

.insight-card {
    background: #1a1d2e;
    border-left: 4px solid #6366f1;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}
.rec-card {
    background: #1a1d2e;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #2e3250;
    height: 100%;
}
.rec-name  { font-weight: 600; font-size: 0.88rem; color: #e2e8f0; }
.rec-brand { font-size: 0.75rem; color: #9aa0b4; margin-top: 2px; }
.rec-price { font-size: 1.05rem; font-weight: 700; color: #34d399; margin: 6px 0 2px; }
.rec-badge { background:#ef4444; color:#fff; font-size:0.70rem; padding:2px 6px; border-radius:8px; }
.status-Delivered   { color: #34d399; }
.status-Shipped     { color: #60a5fa; }
.status-Processing  { color: #f59e0b; }
.status-Cancelled   { color: #f87171; }
</style>
""", unsafe_allow_html=True)


# Data loader (cached 5 min per user_id)
@st.cache_data(ttl=300, show_spinner=False)
def _load_data(user_id: str) -> dict:
    db = SessionLocal()
    try:
        return get_dashboard_data(db, user_id)
    finally:
        db.close()


@st.cache_data(ttl=300, show_spinner=False)
def _load_recs(user_id: str) -> dict:
    db = SessionLocal()
    try:
        return get_recommended_products(db, user_id, limit=6)
    finally:
        db.close()


# Load data
_UID: str = st.session_state["user_id"]

with st.spinner("Loading your dashboard…"):
    data = _load_data(_UID)
    recs_data = _load_recs(_UID)

if not data:
    st.error("Failed to load dashboard data. Please try logging in again.")
    st.stop()

profile  = data["profile"]
kpis     = data["kpis"]
charts   = data["charts"]
tables   = data["tables"]
insights = data["insights"]
recs     = recs_data.get("recommendations", [])


# Sidebar
with st.sidebar:
    st.markdown(f"### 👤 {profile['full_name']}")
    st.markdown(f"📧 `{profile['email']}`")
    loyalty = profile["loyalty_level"]
    st.markdown(
        f'<span class="loyalty-badge loyalty-{loyalty}">{loyalty} Member</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("💬 AI Shopping Assistant", width='stretch', type="primary", key="sb_chat"):
        st.switch_page("pages/customer_chat.py")
    st.divider()
    if st.button("🔄 Refresh", width='stretch', key="sb_refresh"):
        st.cache_data.clear()
        st.rerun()
    if st.button("🚪 Logout", width='stretch', type="secondary", key="sb_logout"):
        clear_session()
        st.switch_page("Home.py")
    st.divider()
    st.caption("🛍️ Powered by Retail AI")



# HEADER + CTA

col_title, col_cta = st.columns([3, 1])
with col_title:
    st.title(f"🛍️ Welcome back, {profile['full_name']}!")
    st.markdown("Here's your personalised shopping intelligence.")
with col_cta:
    st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
    if st.button("💬 Open AI Shopping Assistant", width='stretch',
                 type="primary", key="top_chat"):
        st.switch_page("pages/customer_chat.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()



# SECTION 1 — WELCOME CARD

w1, w2, w3, w4 = st.columns(4)

def _welcome_tile(col, icon: str, label: str, value: str) -> None:
    with col:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value" style="font-size:1.1rem">{value}</div>
            </div>""",
            unsafe_allow_html=True,
        )

_welcome_tile(w1, "🏆", "Loyalty Level",
              f'<span class="loyalty-badge loyalty-{loyalty}">{loyalty}</span>')
_welcome_tile(w2, "🛍️", "Favourite Category", profile["preferred_category"])
_welcome_tile(w3, "📅", "Member Since",       profile["join_date"])
_welcome_tile(w4, "📍", "Location",            f"{profile['city']}, {profile['state']}")

st.markdown("<br>", unsafe_allow_html=True)



# SECTION 2 — KPI CARDS

st.subheader("📊 Your Shopping Summary")

k1, k2, k3, k4 = st.columns(4)

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

_kpi(k1, "🛒 Total Orders",      str(kpis["total_orders"]),     "All time")
_kpi(k2, "💸 Total Spent",        kpis["total_spent_fmt"],       f"Avg {_fmt_inr(kpis['avg_order_value'])}/order")
_kpi(k3, "⭐ Reward Points",      f"{kpis['reward_points']:,}",  f"{loyalty} × {_loyalty_rate(loyalty)} multiplier")
_kpi(k4, "❤️ Wishlist",           "Coming Soon",                 "Feature in progress")

st.markdown("<br>", unsafe_allow_html=True)



# SECTION 3 — CHARTS

st.subheader("📈 Spending Analytics")

ch1, ch2, ch3 = st.columns([2, 1.2, 1])

# --- Monthly Spending Trend ---
with ch1:
    st.markdown("**📅 Monthly Spending Trend**")
    monthly = charts.get("monthly_spending", [])
    if monthly:
        df_m = pd.DataFrame(monthly)
        # Show last 12 months max
        df_m = df_m.tail(12)
        fig, ax = plt.subplots(figsize=(7, 3.0))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.plot(df_m["label"], df_m["spent"] / 1_000, marker="o",
                color="#6366f1", linewidth=2, markersize=5)
        ax.fill_between(range(len(df_m)), df_m["spent"] / 1_000,
                        alpha=0.12, color="#6366f1")
        ax.set_xticks(range(len(df_m)))
        ax.set_xticklabels(df_m["label"], rotation=45, ha="right",
                           fontsize=7, color="#9aa0b4")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
        ax.tick_params(axis="y", colors="#9aa0b4", labelsize=8)
        ax.spines[:].set_color("#2e3250")
        ax.grid(axis="y", color="#2e3250", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No spending data yet.")

# --- Category Pie Chart ---
with ch2:
    st.markdown("**🗂️ Purchases by Category**")
    cat_data = charts.get("category_spending", [])
    if cat_data:
        df_c = pd.DataFrame(cat_data)
        fig, ax = plt.subplots(figsize=(4, 3.0))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        colours = plt.cm.Set3.colors[:len(df_c)]
        wedges, texts, autotexts = ax.pie(
            df_c["spent"],
            labels=df_c["category"],
            autopct="%1.0f%%",
            startangle=90,
            colors=colours,
            textprops={"color": "#c8cde0", "fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(6)
            at.set_color("#f0f2f8")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No category data yet.")

# --- Order Status Bar Chart ---
with ch3:
    st.markdown("**📦 Order Status**")
    status_data = charts.get("order_status", [])
    if status_data:
        df_s = pd.DataFrame(status_data).sort_values("count", ascending=False)
        _status_colours = {
            "Delivered":  "#34d399",
            "Shipped":    "#60a5fa",
            "Processing": "#f59e0b",
            "Cancelled":  "#f87171",
        }
        colours = [_status_colours.get(s, "#9aa0b4") for s in df_s["status"]]
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.bar(df_s["status"], df_s["count"], color=colours)
        ax.tick_params(axis="x", colors="#9aa0b4", labelsize=7)
        ax.tick_params(axis="y", colors="#9aa0b4", labelsize=8)
        ax.spines[:].set_color("#2e3250")
        ax.grid(axis="y", color="#2e3250", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No order data yet.")

st.divider()



# SECTION 4 — AI SHOPPING INSIGHTS

st.subheader("💡 Your Shopping Insights")
st.caption("Personalised observations based on your purchase history")

for obs in insights:
    st.markdown(
        f'<div class="insight-card">{obs}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()



# SECTION 5 — TABLES

st.subheader("📋 Your Orders & Reviews")

tab1, tab2, tab3 = st.tabs(["🕐 Recent Orders", "🚚 Active Orders", "⭐ Recent Reviews"])

# --- Recent Orders ---
with tab1:
    recent = tables.get("recent_orders", [])
    if recent:
        df_r = pd.DataFrame(recent)
        df_r["total_amount"] = df_r["total_amount"].apply(
            lambda x: f"₹{x:,.0f}" if x else "₹0"
        )
        df_r["delivery_status"] = df_r["delivery_status"].apply(
            lambda s: f"{'✅' if s=='Delivered' else '🚚' if s=='Shipped' else '⏳' if s=='Processing' else '❌'} {s}"
        )
        st.dataframe(
            df_r.rename(columns={
                "order_id": "Order ID", "order_date": "Date",
                "total_amount": "Amount", "payment_method": "Payment",
                "payment_status": "Pay Status", "delivery_status": "Status",
                "delivery_date": "Delivered",
            })[["Order ID", "Date", "Amount", "Payment", "Pay Status", "Status", "Delivered"]],
            width='stretch', hide_index=True,
        )
    else:
        st.info("No orders placed yet.")

# --- Active Orders ---
with tab2:
    active = tables.get("active_orders", [])
    if active:
        df_a = pd.DataFrame(active)
        df_a["total_amount"] = df_a["total_amount"].apply(
            lambda x: f"₹{x:,.0f}" if x else "₹0"
        )
        df_a["delivery_status"] = df_a["delivery_status"].apply(
            lambda s: f"{'🚚' if s=='Shipped' else '⏳'} {s}"
        )
        st.dataframe(
            df_a.rename(columns={
                "order_id": "Order ID", "order_date": "Ordered On",
                "total_amount": "Amount", "delivery_status": "Status",
                "payment_method": "Payment",
            })[["Order ID", "Ordered On", "Amount", "Status", "Payment"]],
            width='stretch', hide_index=True,
        )
    else:
        st.success("✅ No active orders — all caught up!")

# --- Recent Reviews ---
with tab3:
    reviews = tables.get("recent_reviews", [])
    if reviews:
        for rev in reviews:
            sentiment_icon = {
                "Positive": "💚", "Neutral": "🟡", "Negative": "🔴"
            }.get(rev.get("sentiment", ""), "⭐")
            stars = "⭐" * int(rev.get("rating", 0))
            st.markdown(
                f"**{rev.get('product_name', '')}** — *{rev.get('brand', '')}*  \n"
                f"{stars} {sentiment_icon} &nbsp; `{rev.get('review_date', '')}`  \n"
                f"_{rev.get('text', '')}_"
            )
            st.markdown("---")
    else:
        st.info("You haven't written any reviews yet.")

st.divider()



# SECTION 6 — PRODUCT RECOMMENDATIONS

st.subheader("🎯 Recommended For You")
pref_cat = recs_data.get("preferred_category", "")
strategy_desc = recs_data.get("strategy_description", "")
st.caption(f"📌 {strategy_desc}")

if recs:
    # 3 per row × 2 rows = 6 products
    rows_of_3 = [recs[i:i+3] for i in range(0, len(recs), 3)]
    for row in rows_of_3:
        cols = st.columns(3)
        for col, prod in zip(cols, row):
            with col:
                disc = prod.get("discount_percentage", 0)
                badge_html = (
                    f'<span class="rec-badge">-{disc:.0f}%</span> '
                    if disc > 0 else ""
                )
                rating = prod.get("rating", 0)
                stars = "⭐" * int(round(rating))
                st.markdown(
                    f"""<div class="rec-card">
                    <div class="rec-name">{prod.get("product_name","")}</div>
                    <div class="rec-brand">{prod.get("brand","")} · {prod.get("category","")}</div>
                    <div class="rec-price">₹{prod.get('final_price', 0):,.0f} {badge_html}</div>
                    <div style="font-size:0.72rem;color:#9aa0b4">{stars} ({prod.get('total_reviews',0)} reviews)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)
else:
    st.info("Recommendations will appear here once you start shopping!")

st.divider()



# SECTION 7 — BOTTOM AI ASSISTANT CTA

cta_l, cta_c, cta_r = st.columns([1, 2, 1])
with cta_c:
    st.markdown(
        "<h4 style='text-align:center;'>Need help finding products?</h4>"
        "<p style='text-align:center;color:#9aa0b4;'>Ask our AI assistant about policies, "
        "products, your orders, or get personalised recommendations.</p>",
        unsafe_allow_html=True,
    )
    if st.button(
        "💬 Open AI Shopping Assistant",
        width='stretch',
        type="primary",
        key="bottom_chat",
    ):
        st.switch_page("pages/customer_chat.py")

st.divider()
st.caption(
    "🛍️ Retail AI System · v1.0.0 · "
    "Dashboard data refreshes every 5 minutes"
)


