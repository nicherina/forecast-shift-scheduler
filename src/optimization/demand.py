"""Convert forecasted visitors into per-slot staffing requirements.

Chain of (stated) assumptions:
    visitors(day, hour)  = visitors(day) * HOURLY_PROFILE[h]
    required(role, h)    = ceil(visitors(hour) * ROLE_VISITOR_SHARE[r]
                                / VISITORS_PER_STAFF_PER_SLOT[r])
                         capped at MAX_STAFF_PER_SLOT
"""
import math

import numpy as np
import pandas as pd

from .. import config


def _normalised_profile() -> dict[int, float]:
    arr = np.array(config.HOURLY_PROFILE, dtype=float)
    arr /= arr.sum()
    return {h: float(w) for h, w in zip(config.OPEN_HOURS, arr)}


def visitors_to_demand(visitors_per_day: pd.Series) -> pd.DataFrame:
    """visitors_per_day: index = Timestamp (one location), value = expected visitors.

    Returns DataFrame: date, hour, role, required.
    """
    profile = _normalised_profile()
    rows = []
    for date, visitors in visitors_per_day.items():
        for hour, share in profile.items():
            slot_visitors = visitors * share
            for role in config.ROLES:
                served = config.ROLE_VISITOR_SHARE[role] * slot_visitors
                need = math.ceil(served / config.VISITORS_PER_STAFF_PER_SLOT[role])
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "role": role,
                    "required": min(need, config.MAX_STAFF_PER_SLOT),
                })
    return pd.DataFrame(rows)


def forecast_to_demand(forecast: pd.DataFrame, store_id: str) -> pd.DataFrame:
    """forecast: store_id, date, visitors_pred. Picks the given store."""
    sel = forecast.loc[forecast["store_id"] == store_id]
    if sel.empty:
        raise ValueError(f"store {store_id!r} not in forecast")
    series = sel.set_index(pd.to_datetime(sel["date"]))["visitors_pred"].sort_index()
    return visitors_to_demand(series)
