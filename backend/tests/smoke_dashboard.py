"""Smoke test — Dashboard Service + all ML modules (no LLM)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from backend.ml import (
    sales_forecast, demand_prediction, customer_segmentation,
    inventory_prediction, sentiment_analysis, product_performance,
)
print("ML module imports: OK")

from backend.core.database import SessionLocal
from backend.services.dashboard_service import get_dashboard_data

db = SessionLocal()
data = get_dashboard_data(db)
db.close()

k = data["kpis"]
print(f"  Revenue          : {k['total_revenue_fmt']}")
print(f"  Orders           : {k['total_orders']}")
print(f"  Customers        : {k['total_customers']}")
print(f"  Products         : {k['total_products']}")
print(f"  Total stock      : {k['total_stock']}")
print(f"  Low stock        : {k['low_stock_count']}")

f = data["ml"]["forecast"]
print(f"  30d Forecast     : {f['forecast_30d_fmt']}  trend={f['trend']}")

d = data["ml"]["demand"]
print(f"  High demand prods: {len(d['high_demand'])}  model={d['model']}")

ir = data["ml"]["inventory_risk"]
print(f"  Inv risk alerts  : critical={ir['critical_count']} high={ir['high_count']} medium={ir['medium_count']}")

seg = data["ml"]["segments"]
print(f"  Segments         : {[s['segment'] for s in seg['segments']]}")

sent = data["ml"]["sentiment"]
print(f"  Sentiment        : {sent['distribution']}  positive={sent['positive_pct']}%")

perf = data["ml"]["performance"]
print(f"  Top performer    : {perf['top_performers'][0]['product_name']}")

print(f"  Monthly chart pts: {len(data['charts']['monthly_sales'])}")
print(f"  Category revenue : {[c['category'] for c in data['charts']['category_revenue']]}")
print(f"  Low stock table  : {len(data['tables']['low_stock_products'])} rows")
print(f"  Top customers    : {len(data['tables']['top_customers'])} rows")
print(f"  Insights         : {len(data['insights'])} observations")
for i, obs in enumerate(data["insights"], 1):
    clean = obs.replace("**", "")
    print(f"    {i}. {clean[:100]}")

print("\n=== ALL BACKEND CHECKS PASSED ===")
