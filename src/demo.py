"""End-to-end demo run WITHOUT any data download:
synthetic visitors -> forecast -> staffing requirements -> MIP schedule,
compared against the greedy baseline.

    python -m src.demo              # auto solver (Gurobi if available, else CP-SAT)
    python -m src.demo --solver cpsat --days 60 --store store_03

For the real-data variant see scripts/download_data.py and the README.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import N_EMPLOYEES_DEFAULT, FORECAST_DAYS_DEFAULT
from src.forecast.train import train_and_forecast
from src.optimization.demand import forecast_to_demand
from src.optimization.employees import make_employee_pool
from src.optimization.solve import solve
from src.evaluation.evaluate import compare, save_report


def make_synthetic_visitors(n_stores: int = 10, days: int = 100,
                            seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    dow_mult = np.array([0.8, 0.9, 1.0, 1.0, 1.15, 1.45, 1.55])  # Mon..Sun
    rows = []
    for s in range(n_stores):
        base = rng.uniform(80, 220)
        store_mult = 1 + 0.15 * np.sin(np.linspace(0, 2 * np.pi, days))
        visitors = base * np.array([dow_mult[d.dayofweek] for d in dates]) \
            * store_mult * rng.normal(1, 0.1, days)
        visitors = np.clip(visitors, 10, None)
        for d, v in zip(dates, visitors):
            rows.append({"store_id": f"store_{s:02d}", "date": d, "visitors": round(v)})
    return pd.DataFrame(rows)


def run(source: str = "synthetic", solver: str = "auto",
        days: int = 100, store_id: str | None = None,
        n_employees: int = N_EMPLOYEES_DEFAULT,
        forecast_days: int = FORECAST_DAYS_DEFAULT) -> None:
    if source == "synthetic":
        visitors = make_synthetic_visitors(days=days)
        holidays: set = set()
    elif source == "kaggle":
        visitors = pd.read_csv("data/processed/visitors.csv",
                               dtype={"store_id": str})
        hz = pd.read_csv("data/processed/holidays.csv")
        holidays = set(pd.to_datetime(hz["date"]))
    else:
        raise ValueError(source)

    forecast, metrics = train_and_forecast(visitors, holidays, forecast_days)
    print(f"forecast WMAPE: {metrics['wmape']:.3f} "
          f"({metrics['test_rows']} test rows over {metrics['test_days']} days)")

    store = store_id or forecast["store_id"].iloc[0]
    demand = forecast_to_demand(forecast, store)
    print(f"planning store={store}: {len(dates_avail := sorted(demand['date'].unique()))} days "
          f"x {demand['hour'].nunique()} hours x {demand['role'].nunique()} roles")

    dates_iso = sorted(demand["date"].unique())
    employees = make_employee_pool(n_employees, dates_iso)
    result = solve(demand, employees, solver=solver)
    print(f"solver={result['solver']} status={result['status']} "
          f"runtime={result['runtime_s']}s")

    table = compare(demand, employees, result)
    print("\n=== MIP vs greedy baseline (same KPI definitions) ===")
    print(table.to_string())

    out_dir = save_report(table, result["schedule"], "reports")
    print(f"\nreports saved under {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    p.add_argument("--solver", choices=["auto", "gurobi", "cpsat"], default="auto")
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--store", default=None)
    args = p.parse_args()
    run(args.source, args.solver, days=args.days, store_id=args.store)
