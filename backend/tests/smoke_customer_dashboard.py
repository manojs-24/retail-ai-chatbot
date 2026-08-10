"""Smoke test — Customer Dashboard services (no LLM)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from backend.core.database import SessionLocal
from backend.services.customer_dashboard_service import (
    get_dashboard_data, get_customer_summary, get_reward_summary,
)
from backend.services.shopping_summary_service import (
    get_spending_by_month, get_spending_by_category,
    get_trending_products, get_brand_summary,
)
from backend.services.recommendation_service import get_recommended_products

print("Service imports: OK")

db = SessionLocal()
UID = "U0195"

# --- customer_dashboard_service ---
data = get_dashboard_data(db, UID)
p = data["profile"]
k = data["kpis"]
print(f"Profile  : {p['full_name']} | {p['loyalty_level']} | pref={p['preferred_category']}")
print(f"KPIs     : orders={k['total_orders']} spent={k['total_spent_fmt']} points={k['reward_points']}")
print(f"Savings  : {k['total_savings_fmt']}")
print(f"Monthly chart  : {len(data['charts']['monthly_spending'])} pts")
print(f"Category chart : {len(data['charts']['category_spending'])} pts")
print(f"Status chart   : {len(data['charts']['order_status'])} pts")
print(f"Recent orders  : {len(data['tables']['recent_orders'])}")
print(f"Active orders  : {len(data['tables']['active_orders'])}")
print(f"Recent reviews : {len(data['tables']['recent_reviews'])}")
print(f"Insights ({len(data['insights'])}):")
for ins in data["insights"]:
    clean = ins.replace("**", "")
    print(f"  - {clean[:90]}")

# --- shopping_summary_service ---
monthly = get_spending_by_month(db, UID)
print(f"Monthly spending : {len(monthly)} pts")
cat = get_spending_by_category(db, UID)
print(f"Categories       : {[c['category'] for c in cat]}")
brands = get_brand_summary(db, UID, limit=3)
print(f"Top brands       : {[b['brand'] for b in brands]}")
trending = get_trending_products(db, limit=6)
print(f"Trending products: {[t['product_name'][:20] for t in trending]}")

# --- recommendation_service ---
recs = get_recommended_products(db, UID, limit=6)
print(f"Recs strategy    : {recs['strategy']} | pref_cat={recs['preferred_category']}")
print(f"Recs count       : {recs['count']}")
for r in recs["recommendations"]:
    print(f"  {r['product_name'][:30]:32} | {r['category']:15} | {r['final_price']}")

db.close()
print("\n=== ALL BACKEND CHECKS PASSED ===")
