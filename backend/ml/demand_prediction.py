"""
Demand Prediction Module
========================
Identifies high-demand products by scoring each active product on a
composite demand signal derived from:

- units sold (normalised)
- revenue generated (normalised)
- current rating (normalised)
- stock velocity = units_sold / max(stock_quantity, 1)

A RandomForestRegressor is trained to predict the demand score, so the
ranking can later be extended with richer features (recency, seasonality, etc.).

Output contract
---------------
``run(products_df)`` → dict with:
    - ``high_demand``   : list[dict] — top-N products with demand_score
    - ``model``         : str
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

_TOP_N = 8


def run(products: list[dict]) -> dict:
    """
    Predict demand scores for all products and return the top-N.

    Args:
        products: List of dicts with keys ``product_id``, ``product_name``,
                  ``category``, ``brand``, ``units_sold``, ``revenue``,
                  ``final_price``, ``rating`` (use ``total_sold`` and
                  ``stock_quantity`` from full product data).
                  Compatible with the output of
                  :meth:`~backend.repositories.product_repository
                  .ProductRepository.top_selling_with_revenue`.

    Returns:
        Demand prediction dict (see module docstring).
    """
    if len(products) < 3:
        return {"high_demand": products[:_TOP_N], "model": "rule_based"}

    df = pd.DataFrame(products)

    # Ensure required columns exist with safe defaults.
    for col in ["units_sold", "revenue", "final_price"]:
        if col not in df.columns:
            df[col] = 0
    if "rating" not in df.columns:
        df["rating"] = 3.0
    if "stock_quantity" not in df.columns:
        df["stock_quantity"] = 1

    df["stock_velocity"] = df["units_sold"] / df["stock_quantity"].clip(lower=1)

    features = ["units_sold", "revenue", "final_price", "stock_velocity"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features].fillna(0))
    scaled_df = pd.DataFrame(scaled, columns=features)

    # Composite demand score (simple weighted sum as target).
    target = (
        0.4 * scaled_df["units_sold"]
        + 0.35 * scaled_df["revenue"]
        + 0.15 * scaled_df["stock_velocity"]
        + 0.10 * (MinMaxScaler().fit_transform(df[["rating"]].fillna(3)).flatten())
    )

    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(scaled, target)
    df["demand_score"] = model.predict(scaled)

    top = (
        df.sort_values("demand_score", ascending=False)
        .head(_TOP_N)
        .round({"demand_score": 3})
    )

    cols = [c for c in ["product_id", "product_name", "brand", "category",
                         "units_sold", "revenue", "demand_score"] if c in top.columns]
    return {
        "high_demand": top[cols].to_dict(orient="records"),
        "model": "RandomForestRegressor",
    }
