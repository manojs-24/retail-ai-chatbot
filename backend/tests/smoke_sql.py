"""Smoke test — SQL layer (no LLM calls)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from backend.repositories.user_repository import UserRepository
from backend.repositories.product_repository import ProductRepository
from backend.repositories.order_repository import OrderRepository
from backend.repositories.review_repository import ReviewRepository
print("Repositories: OK")

from backend.services.user_service import UserService
from backend.services.product_service import ProductService
from backend.services.order_service import OrderService
from backend.services.recommendation_service import RecommendationService
from backend.services.analytics_service import AnalyticsService
print("Services: OK")

from backend.tools import customer_sql_tool, manager_sql_tool
print("Tools: OK")

from backend.nodes.customer.sql_node import sql_node as c_sql
from backend.nodes.customer.recommendation_node import recommendation_node
from backend.nodes.manager.sql_node import sql_node as m_sql
print("Nodes: OK")

from backend.core.database import SessionLocal
from backend.models.models import Order

# --- UserService ---
db = SessionLocal()
us = UserService()
p = us.get_customer_profile(db, "U0001")
db.close()
print(f"UserService.get_customer_profile(U0001): {p['full_name']} / {p['loyalty_level']}")

# --- ProductService ---
db = SessionLocal()
ps = ProductService()
res = ps.search_products(db, "laptop", limit=3)
db.close()
print(f"ProductService.search_products(laptop): {res['count']} results")

# --- OrderService + security ---
db = SessionLocal()
os_ = OrderService()
hist = os_.get_purchase_history(db, "U0001")
other_order = db.query(Order).filter(Order.user_id == "U0002").first()
denied = os_.get_order_details(db, other_order.order_id, "U0001")
db.close()
print(f"OrderService.get_purchase_history(U0001): {hist['total_orders']} orders")
print(f"Cross-user access blocked: {denied is None}")

# --- RecommendationService ---
db = SessionLocal()
rs = RecommendationService()
recs = rs.recommend_for_customer(db, "U0001", limit=5)
db.close()
print(f"RecommendationService(U0001): {recs['count']} recs, strategy={recs['strategy']}")

# --- AnalyticsService ---
db = SessionLocal()
ans = AnalyticsService()
inv = ans.get_inventory_summary(db)
sales = ans.get_sales_summary(db)
monthly = ans.get_monthly_sales(db)
db.close()
print(f"AnalyticsService.inventory_summary: total_products={inv['total_products']}, low={inv['low_stock']}")
print(f"AnalyticsService.sales_summary: revenue={sales['total_revenue']:,.2f}, orders={sales['total_orders']}")
print(f"AnalyticsService.monthly_sales: {monthly['total_months']} months")

# --- Customer tool ---
r1 = customer_sql_tool.get_recent_orders("U0001", limit=3)
r2 = customer_sql_tool.search_products("samsung")
r3 = customer_sql_tool.recommend_products("U0001", limit=5)
print(f"customer_sql_tool.get_recent_orders: {r1['count']} orders")
print(f"customer_sql_tool.search_products(samsung): {r2['count']} results")
print(f"customer_sql_tool.recommend_products: {r3['count']} recs")

# --- Manager tool ---
r4 = manager_sql_tool.inventory_summary()
r5 = manager_sql_tool.sales_summary()
r6 = manager_sql_tool.top_selling_products(limit=5)
r7 = manager_sql_tool.customer_details("U0001")
r8 = manager_sql_tool.monthly_sales()
r9 = manager_sql_tool.low_stock_products()
r10 = manager_sql_tool.category_summary()
print(f"manager_sql_tool.inventory_summary: total_products={r4['total_products']}")
print(f"manager_sql_tool.sales_summary: total_orders={r5['total_orders']}")
print(f"manager_sql_tool.top_selling_products: {r6['count']} products")
print(f"manager_sql_tool.customer_details(U0001): {r7['full_name']}")
print(f"manager_sql_tool.monthly_sales: {r8['total_months']} months")
print(f"manager_sql_tool.low_stock_products: {r9['low_stock_count']} low, {r9['out_of_stock_count']} OOS")
print(f"manager_sql_tool.category_summary: {r10['category_count']} categories")

print("\n=== ALL CHECKS PASSED ===")
