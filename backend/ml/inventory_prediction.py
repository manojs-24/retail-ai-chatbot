"""
Inventory Prediction Module
============================
Identifies products at risk of stockout within the next 30 days by
computing a daily sales velocity and projecting days-of-stock-remaining.

Risk classification
-------------------
- CRITICAL  : projected stockout within 7 days
- HIGH      : projected stockout within 15 days
- MEDIUM    : projected stockout within 30 days
- OK        : sufficient stock for > 30 days

Output contract
---------------
``run(products, order_items)`` → dict with:
    - ``risk_alerts``   : list[dict] — products at CRITICAL / HIGH / MEDIUM risk
    - ``critical_count``: int
    - ``high_count``    : int
    - ``medium_count``  : int
    - ``model``         : str
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_DAYS_HISTORY = 90  # rolling window for velocity calculation


def run(products: list[dict], order_items: list[dict]) -> dict:
    """
    Compute inventory risk for all active products.

    Args:
        products:    List of dicts with ``product_id``, ``product_name``,
                     ``category``, ``brand``, ``stock_quantity``,
                     ``reorder_level``.
        order_items: List of dicts with ``product_id``, ``quantity``,
                     ``order_date`` (ISO string or date object).

    Returns:
        Inventory risk dict (see module docstring).
    """
    if not products:
        return _empty_result()

    prod_df = pd.DataFrame(products)
    if prod_df.empty:
        return _empty_result()

    alerts = []

    if order_items:
        items_df = pd.DataFrame(order_items)
        items_df["order_date"] = pd.to_datetime(items_df["order_date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=_DAYS_HISTORY)
        recent = items_df[items_df["order_date"] >= cutoff]
        velocity = (
            recent.groupby("product_id")["quantity"].sum() / _DAYS_HISTORY
        )
    else:
        velocity = pd.Series(dtype=float)

    for _, row in prod_df.iterrows():
        pid = row["product_id"]
        stock = float(row.get("stock_quantity", 0))
        daily_vel = float(velocity.get(pid, 0.0))

        if daily_vel > 0:
            days_remaining = stock / daily_vel
        else:
            # No recent sales — use a conservative velocity estimate
            days_remaining = float("inf")

        risk = _classify_risk(days_remaining, stock, row.get("reorder_level", 0))
        if risk != "OK":
            alerts.append({
                "product_id": pid,
                "product_name": row.get("product_name", ""),
                "brand": row.get("brand", ""),
                "category": row.get("category", ""),
                "stock_quantity": int(stock),
                "reorder_level": int(row.get("reorder_level", 0)),
                "daily_velocity": round(daily_vel, 2),
                "days_remaining": round(days_remaining, 1) if days_remaining != float("inf") else 999,
                "risk_level": risk,
            })

    alerts.sort(key=lambda x: (
        {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["risk_level"], 3),
        x["days_remaining"],
    ))

    return {
        "risk_alerts": alerts,
        "critical_count": sum(1 for a in alerts if a["risk_level"] == "CRITICAL"),
        "high_count": sum(1 for a in alerts if a["risk_level"] == "HIGH"),
        "medium_count": sum(1 for a in alerts if a["risk_level"] == "MEDIUM"),
        "model": "velocity_projection",
    }


def _classify_risk(days_remaining: float, stock: float, reorder_level: float) -> str:
    if stock == 0:
        return "CRITICAL"
    if days_remaining <= 7:
        return "CRITICAL"
    if days_remaining <= 15:
        return "HIGH"
    if days_remaining <= 30 or stock <= reorder_level:
        return "MEDIUM"
    return "OK"


def _empty_result() -> dict:
    return {
        "risk_alerts": [],
        "critical_count": 0,
        "high_count": 0,
        "medium_count": 0,
        "model": "velocity_projection",
    }
