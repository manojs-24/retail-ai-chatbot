"""
Customer Segmentation Module
============================
Segments customers into distinct groups using K-Means clustering on
RFM (Recency, Frequency, Monetary) features derived from their order history.

RFM features
------------
- R (Recency)   : days since last order (lower = more recent = better)
- F (Frequency) : total number of orders
- M (Monetary)  : total amount spent

Clusters are relabelled with business-friendly names based on centroid
characteristics: Champion, Loyal, At-Risk, New Customer, etc.

Output contract
---------------
``run(orders_df, reference_date)`` → dict with:
    - ``segments``      : list[dict] — segment name + customer count + avg_spend
    - ``summary``       : dict       — counts per segment
    - ``model``         : str
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_N_CLUSTERS = 4

_SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk",
    3: "New Customers",
}


def run(orders: list[dict], customers: list[dict]) -> dict:
    """
    Segment customers using K-Means on RFM features.

    Args:
        orders:    List of dicts with keys ``user_id``, ``order_date``,
                   ``total_amount``.  All orders in the database.
        customers: List of dicts with key ``user_id``.

    Returns:
        Segmentation dict (see module docstring).
    """
    if not orders or not customers:
        return _empty_result()

    df = pd.DataFrame(orders)
    df["order_date"] = pd.to_datetime(df["order_date"])
    ref_date = pd.Timestamp(date.today())

    rfm = df.groupby("user_id").agg(
        recency=("order_date", lambda x: (ref_date - x.max()).days),
        frequency=("user_id", "count"),
        monetary=("total_amount", "sum"),
    ).reset_index()

    if len(rfm) < _N_CLUSTERS:
        return _empty_result()

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    km = KMeans(n_clusters=_N_CLUSTERS, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X)

    # Label clusters: sort centroids by (low recency + high freq + high monetary).
    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=["recency", "frequency", "monetary"],
    )
    centers["score"] = (
        -centers["recency"] + centers["frequency"] + centers["monetary"] / 1000
    )
    rank = centers["score"].rank(ascending=False).astype(int) - 1
    label_map = {old: _SEGMENT_LABELS.get(new, f"Segment {new}") for old, new in rank.items()}
    rfm["segment"] = rfm["cluster"].map(label_map)

    summary = (
        rfm.groupby("segment")
        .agg(
            customer_count=("user_id", "count"),
            avg_spend=("monetary", "mean"),
            avg_orders=("frequency", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "segments": summary,
        "total_customers": len(rfm),
        "model": "KMeans (RFM)",
    }


def _empty_result() -> dict:
    return {
        "segments": [],
        "total_customers": 0,
        "model": "KMeans (RFM)",
    }
