from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from backend.ml.sales_forecast import _MIN_POINTS_FOR_DEGREE2, _make_poly_pipeline
from backend.tools.manager_sql_tool import monthly_sales

logger = logging.getLogger(__name__)

# Minimum months of data needed for meaningful evaluation
MIN_MONTHS_FOR_EVAL = 6

# Number of held-out months used as the test set
TEST_MONTHS = 3


@dataclass
class MLEvalResult:
    """Result of the ML forecasting evaluation."""
    mae: float
    rmse: float
    mape: float
    r2: float
    sufficient_data: bool
    total_months: int
    train_months: int
    test_months: int
    model_name: str = "PolynomialRegression(degree=2)"
    warning: str = ""
    actuals_vs_predicted_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def _compute_metrics(
    actuals: list[float],
    predictions: list[float],
) -> tuple[float, float, float, float]:

    n = len(actuals)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    errors = [a - p for a, p in zip(actuals, predictions)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

    # MAPE — guard against zero actuals
    mape_terms = [abs(e / a) * 100 for e, a in zip(errors, actuals) if a != 0]
    mape = sum(mape_terms) / len(mape_terms) if mape_terms else 0.0

    # R²
    mean_actual = sum(actuals) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum(e ** 2 for e in errors)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return mae, rmse, mape, r2


def run_ml_evaluation() -> MLEvalResult:

    logger.info("ML evaluation — fetching monthly sales data")
    monthly_result = monthly_sales()
    monthly_data: list[dict] = monthly_result.get("months", [])

    total_months = len(monthly_data)
    logger.info("ML evaluation — total months available: %d", total_months)

    # Validate sufficient data
    if total_months < MIN_MONTHS_FOR_EVAL:
        warning = (
            f"⚠️ Insufficient historical data for reliable forecasting. "
            f"Found {total_months} month(s) of data; "
            f"at least {MIN_MONTHS_FOR_EVAL} required. "
            f"Populate more sales data and re-run the evaluation."
        )
        logger.warning("ML evaluation — insufficient data: %d months", total_months)
        return MLEvalResult(
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            r2=0.0,
            sufficient_data=False,
            total_months=total_months,
            train_months=0,
            test_months=0,
            warning=warning,
        )

    # Sort chronologically (DB returns newest-first)
    df = pd.DataFrame(sorted(monthly_data, key=lambda r: (r["year"], r["month"])))
    df["period_idx"] = range(len(df))
    df["label"] = df.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
    )

    actual_test_months = min(TEST_MONTHS, total_months - 3)  # keep at least 3 for training
    train_months = total_months - actual_test_months

    actuals: list[float] = []
    predictions: list[float] = []
    labels: list[str] = []
    model_name_used: str = ""

    # Walk-forward loop — mirrors what production sales_forecast.run() does
    for test_idx in range(train_months, total_months):
        train_df = df.iloc[:test_idx]           # only data available before this month
        n_train = len(train_df)

        X_train = train_df[["period_idx"]].values   # shape (n_train, 1)
        y_train = train_df["revenue"].values         # shape (n_train,)

        # Use the SAME degree selection rule as production
        degree = 2 if n_train >= _MIN_POINTS_FOR_DEGREE2 else 1
        model = _make_poly_pipeline(degree)
        model.fit(X_train, y_train)

        # Predict the test month using its period_idx
        test_period = df.iloc[test_idx]["period_idx"]
        predicted = float(model.predict([[test_period]])[0])
        actual = float(df.iloc[test_idx]["revenue"])

        actuals.append(actual)
        predictions.append(max(predicted, 0.0))   # revenue cannot be negative
        labels.append(str(df.iloc[test_idx]["label"]))

        # Record the model name from the last iteration (most data → most representative)
        model_name_used = f"PolynomialRegression(degree={degree})"

    mae, rmse, mape, r2 = _compute_metrics(actuals, predictions)

    # Build the Actual vs Predicted DataFrame for chart rendering
    avp_df = pd.DataFrame({
        "Month":                  labels,
        "Actual Revenue (₹)":    [round(a, 2) for a in actuals],
        "Predicted Revenue (₹)": [round(p, 2) for p in predictions],
    })

    logger.info(
        "ML evaluation — done | model=%s | mae=%.2f | rmse=%.2f | "
        "mape=%.2f%% | r2=%.3f | test_months=%d",
        model_name_used, mae, rmse, mape, r2, actual_test_months,
    )

    return MLEvalResult(
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2),
        r2=round(r2, 3),
        sufficient_data=True,
        total_months=total_months,
        train_months=train_months,
        test_months=actual_test_months,
        model_name=model_name_used,
        actuals_vs_predicted_df=avp_df,
    )
