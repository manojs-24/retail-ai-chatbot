"""
Sales Forecast Module
=====================
Produces a 30-day rolling sales forecast using a Polynomial Regression
model (degree 2) trained on historical monthly revenue.

Why Polynomial Regression over plain Linear Regression
------------------------------------------------------
Linear regression fits a straight line, so it cannot capture growth that
accelerates, levels off, or dips and recovers.  A degree-2 polynomial adds a
curvature term (period_idx²) which handles these common retail revenue shapes
while remaining entirely within scikit-learn — no new dependencies required.

Auto-degree fallback
--------------------
- n >= 5 months → degree 2  (default — captures trend curvature)
- n == 3 or 4   → degree 1  (falls back to linear to avoid overfitting on
                              very small samples where a curve would just
                              chase noise)

No LLM is involved.  Pure scikit-learn + Pandas.

Output contract
---------------
``run(monthly_data)`` → dict with:
    - ``forecast_30d``         : float  — predicted revenue for the next 30 days
    - ``forecast_30d_fmt``     : str    — human-readable Indian number format
    - ``trend``                : str    — "up" | "down" | "stable"
    - ``trend_pct``            : float  — month-over-month change %
    - ``last_12_months``       : list[dict] — label + actual revenue for chart
    - ``model``                : str    — model name used (includes degree)
    - ``poly_degree``          : int    — polynomial degree actually used
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

# Minimum data points required to safely use degree-2 polynomial.
# With fewer points the extra curvature term overfits to noise.
_MIN_POINTS_FOR_DEGREE2 = 5


def _make_poly_pipeline(degree: int) -> make_pipeline:
    """Return a fitted-ready sklearn Pipeline: PolynomialFeatures → LinearRegression."""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=True),
        LinearRegression(),
    )


def run(monthly_data: list[dict]) -> dict:
    """
    Forecast next-30-day revenue from historical monthly sales data using
    Polynomial Regression (degree 2, with auto fallback to degree 1).

    Args:
        monthly_data: List of dicts with keys ``year``, ``month``, ``revenue``.
                      Returned by :meth:`~backend.repositories.order_repository
                      .OrderRepository.monthly_sales` (newest-first order).

    Returns:
        Forecast dict (see module docstring for full contract).
    """
    if not monthly_data:
        return _empty_result()

    # Sort to chronological order and build a DataFrame.
    df = pd.DataFrame(sorted(monthly_data, key=lambda r: (r["year"], r["month"])))
    df["period_idx"] = range(len(df))
    df["label"] = df.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
    )

    n = len(df)

    # Need at least 3 data points to fit any model.
    if n < 3:
        return _empty_result()

    # Auto-select polynomial degree based on available data.
    # Degree 2 needs at least _MIN_POINTS_FOR_DEGREE2 points to be meaningful.
    degree = 2 if n >= _MIN_POINTS_FOR_DEGREE2 else 1
    logger.debug(
        "sales_forecast.run — n=%d months, using polynomial degree=%d", n, degree
    )

    X = df[["period_idx"]].values   # shape (n, 1) — 2D required by sklearn pipeline
    y = df["revenue"].values        # shape (n,)

    model = _make_poly_pipeline(degree)
    model.fit(X, y)

    # Predict the next period (index = n, one step beyond the last known month).
    next_idx = np.array([[n]])                          # shape (1, 1)
    forecast_monthly: float = float(model.predict(next_idx)[0])

    # Revenue cannot be negative — clamp before scaling.
    # Scale from calendar-month average (30.44 days) to a 30-day window.
    forecast_30d = max(forecast_monthly * (30 / 30.44), 0.0)

    # Trend direction: compare the two most recent actual months.
    last_rev = float(df["revenue"].iloc[-1])
    prev_rev = float(df["revenue"].iloc[-2]) if n >= 2 else last_rev
    trend_pct = ((last_rev - prev_rev) / prev_rev * 100) if prev_rev else 0.0
    trend = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "stable")

    # Last 12 months for the dashboard chart.
    last_12 = df.tail(12)[["label", "revenue"]].to_dict(orient="records")

    model_name = f"PolynomialRegression(degree={degree})"
    logger.info(
        "sales_forecast.run — model=%s | forecast_30d=%.2f | trend=%s",
        model_name, forecast_30d, trend,
    )

    return {
        "forecast_30d":     round(forecast_30d, 2),
        "forecast_30d_fmt": _fmt_inr(forecast_30d),
        "trend":            trend,
        "trend_pct":        round(trend_pct, 1),
        "last_12_months":   last_12,
        "model":            model_name,
        "poly_degree":      degree,
    }


def _fmt_inr(value: float) -> str:
    if value >= 1_00_00_000:
        return f"₹{value/1_00_00_000:.1f} Cr"
    if value >= 1_00_000:
        return f"₹{value/1_00_000:.1f} L"
    return f"₹{value:,.0f}"


def _empty_result() -> dict:
    return {
        "forecast_30d":     0.0,
        "forecast_30d_fmt": "₹0",
        "trend":            "stable",
        "trend_pct":        0.0,
        "last_12_months":   [],
        "model":            "PolynomialRegression(degree=2)",
        "poly_degree":      2,
    }
