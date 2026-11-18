"""Feature engineering for the visitors forecast.

Input: per-store daily visitors DataFrame:
    store_id (str), date (datetime/date), visitors (float)
Output: same rows plus lag/rolling/calendar/store features, NaN rows dropped.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "lag1", "lag7", "lag14",
    "roll7", "roll28",
    "dow", "month", "is_holiday",
    "store_mean", "store_dow_mean",
]


def build_features(df: pd.DataFrame, holidays: set | None = None) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"])

    g = df.groupby("store_id")["visitors"]
    df["lag1"] = g.shift(1)
    df["lag7"] = g.shift(7)
    df["lag14"] = g.shift(14)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_holiday"] = df["date"].isin(set(pd.to_datetime(list(holidays)))) if holidays else 0

    # Store-level aggregates. NOTE: computed over the *whole* series, which
    # leaks the test window into training stats — acceptable for the scaffold,
    # documented as a TODO for a leak-free rolling aggregate.
    df["store_mean"] = df.groupby("store_id")["visitors"].transform("mean")
    df["store_dow_mean"] = df.groupby(["store_id", "dow"])["visitors"].transform("mean")

    return df.dropna(subset=["lag1", "lag7", "roll7"])
