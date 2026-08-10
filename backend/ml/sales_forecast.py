"""
Sales Forecast Module
=====================
Produces a 30-day rolling sales forecast using a Linear Regression model
trained on historical monthly revenue.

No LLM is involved.  Pure scikit-learn + Pandas.

Output contract
---------------
``run(monthly_data)`` → dict with:
    - ``forecast_30d``         : float  — predicted revenue for the next 30 days
    - ``forecast_30d_fmt``     : str    — human-readable Indian number format
    - ``trend``                : str    — "up" | "down" | "stable"
    - ``trend_pct``            : float  — month-over-month change %
    - ``last_12_months``       : list[dict] — label + actual revenue for chart
    - ``model``                : str    — model name used
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def run(monthly_data: list[dict]) -> dict:
    """
    Forecast next-30-day revenue from historical monthly sales data.

    Args:
        monthly_data: List of dicts with keys ``year``, ``month``, ``revenue``.
                      Returned by :meth:`~backend.repositories.order_repository
                      .OrderRepository.monthly_sales` (newest-first order).

    Returns:
        Forecast dict (see module docstring for full contract).
    """
    if not monthly_data:
        return _empty_result()

    # Reverse to chronological order and build a DataFrame.
    df = pd.DataFrame(sorted(monthly_data, key=lambda r: (r["year"], r["month"])))
    df["period_idx"] = range(len(df))
    df["label"] = df.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
    )

    # Need at least 3 data points to fit a line.
    if len(df) < 3:
        return _empty_result()

    X = df[["period_idx"]].values
    y = df["revenue"].values

    model = LinearRegression()
    model.fit(X, y)

    next_idx = len(df)
    forecast_monthly: float = float(model.predict([[next_idx]])[0])
    # Scale monthly forecast to 30-day window.
    forecast_30d = max(forecast_monthly * (30 / 30.44), 0)

    # Trend: compare last 2 actual months.
    last_rev = float(df["revenue"].iloc[-1])
    prev_rev = float(df["revenue"].iloc[-2]) if len(df) >= 2 else last_rev
    trend_pct = ((last_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
    trend = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "stable")

    # Last 12 months for chart.
    last_12 = df.tail(12)[["label", "revenue"]].to_dict(orient="records")

    return {
        "forecast_30d": round(forecast_30d, 2),
        "forecast_30d_fmt": _fmt_inr(forecast_30d),
        "trend": trend,
        "trend_pct": round(trend_pct, 1),
        "last_12_months": last_12,
        "model": "LinearRegression",
    }


def _fmt_inr(value: float) -> str:
    if value >= 1_00_00_000:
        return f"₹{value/1_00_00_000:.1f} Cr"
    if value >= 1_00_000:
        return f"₹{value/1_00_000:.1f} L"
    return f"₹{value:,.0f}"


def _empty_result() -> dict:
    return {
        "forecast_30d": 0.0,
        "forecast_30d_fmt": "₹0",
        "trend": "stable",
        "trend_pct": 0.0,
        "last_12_months": [],
        "model": "LinearRegression",
    }
