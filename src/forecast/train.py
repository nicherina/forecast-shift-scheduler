"""Train a visitors forecaster and predict the last N days for scheduling.

Deliberately simple and honest: one gradient-boosting model across all stores,
time-based split (the *schedule needs a real forecast of the last unseen days*,
so we hold out the tail), WMAPE as the headline error metric.

Model: sklearn HistGradientBoostingRegressor — no external heavy dependency.
A LightGBM upgrade is on the roadmap (README).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURES, build_features


def wmape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1e-9))


def train_and_forecast(
    visitors: pd.DataFrame,
    holidays: set | None = None,
    forecast_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Returns (forecast_df, metrics).

    forecast_df columns: store_id, date, visitors_pred — for the last
    `forecast_days` dates in the input.
    """
    df = build_features(visitors, holidays)
    cutoff = df["date"].max() - pd.Timedelta(days=forecast_days)

    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08)
    model.fit(train[FEATURES], train["visitors"])

    out = test[["store_id", "date"]].copy()
    out["visitors_pred"] = np.clip(model.predict(test[FEATURES]), 0, None)

    metrics = {
        "wmape": wmape(test["visitors"], out["visitors_pred"]),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_days": int(test["date"].nunique()),
    }
    return out, metrics
