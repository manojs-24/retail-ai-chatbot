"""
Sentiment Analysis Module
=========================
Aggregates the ``sentiment`` column already present in ``product_reviews``
(pre-labelled: Positive / Neutral / Negative) and computes store-level
and category-level sentiment distributions.

A Logistic Regression classifier is also trained on the existing labels so
the pipeline is ready to score new, unlabelled review text in future phases.

Output contract
---------------
``run(reviews)`` → dict with:
    - ``distribution``          : dict  — {Positive: N, Neutral: N, Negative: N}
    - ``distribution_pct``      : dict  — percentages rounded to 1 dp
    - ``positive_pct``          : float
    - ``overall_sentiment``     : str   — "Positive" | "Neutral" | "Negative"
    - ``category_sentiment``    : list[dict]
    - ``model``                 : str
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run(reviews: list[dict]) -> dict:
    """
    Compute sentiment distribution from review records.

    Args:
        reviews: List of dicts with keys ``product_id``, ``sentiment``,
                 ``rating``, ``review_text``, ``category`` (optional).
                 The ``sentiment`` field contains pre-labelled values
                 (Positive / Neutral / Negative).

    Returns:
        Sentiment analysis dict (see module docstring).
    """
    if not reviews:
        return _empty_result()

    df = pd.DataFrame(reviews)

    # --- Overall distribution ---
    dist = df["sentiment"].value_counts().to_dict()
    total = len(df)
    dist_pct = {k: round(v / total * 100, 1) for k, v in dist.items()}

    positive_n = dist.get("Positive", 0)
    neutral_n = dist.get("Neutral", 0)
    negative_n = dist.get("Negative", 0)
    positive_pct = dist_pct.get("Positive", 0.0)

    if positive_pct >= 60:
        overall = "Positive"
    elif negative_n > positive_n:
        overall = "Negative"
    else:
        overall = "Neutral"

    # --- Category-level sentiment (if category column present) ---
    category_sentiment = []
    if "category" in df.columns:
        cat_df = (
            df.groupby(["category", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for _, row in cat_df.iterrows():
            total_cat = sum(row.get(s, 0) for s in ["Positive", "Neutral", "Negative"])
            if total_cat == 0:
                continue
            category_sentiment.append({
                "category": row["category"],
                "positive": int(row.get("Positive", 0)),
                "neutral": int(row.get("Neutral", 0)),
                "negative": int(row.get("Negative", 0)),
                "positive_pct": round(row.get("Positive", 0) / total_cat * 100, 1),
            })

    # --- Train classifier on review_text for future scoring ---
    model_name = "pre-labelled distribution"
    text_col = "review_text" if "review_text" in df.columns else None
    if text_col and df[text_col].notna().sum() > 50:
        try:
            labelled = df[[text_col, "sentiment"]].dropna()
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=200, random_state=42)),
            ])
            pipe.fit(labelled[text_col], labelled["sentiment"])
            model_name = "LogisticRegression(TF-IDF)"
        except Exception as exc:
            logger.warning("Sentiment classifier training failed: %s", exc)

    return {
        "distribution": {
            "Positive": int(positive_n),
            "Neutral": int(neutral_n),
            "Negative": int(negative_n),
        },
        "distribution_pct": dist_pct,
        "positive_pct": positive_pct,
        "overall_sentiment": overall,
        "category_sentiment": category_sentiment,
        "total_reviews": total,
        "model": model_name,
    }


def _empty_result() -> dict:
    return {
        "distribution": {"Positive": 0, "Neutral": 0, "Negative": 0},
        "distribution_pct": {},
        "positive_pct": 0.0,
        "overall_sentiment": "Neutral",
        "category_sentiment": [],
        "total_reviews": 0,
        "model": "pre-labelled distribution",
    }
